from __future__ import annotations

import argparse
import csv
import datetime as dt
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from src.config import ExperimentConfig
from src.controllers.pid import PIDController
from src.controllers.pid_tuning import PIDGains, tune_pid
from src.controllers.zero import ZeroController
from src.data.build_dataset import build_historical_datasets, load_historical_dataset
from src.eval.logscore import summarize_logscores
from src.noise_lib.noise_w1_w6 import NoiseModel, create_noise_library
from src.predictors.common import DatasetsByStep, StepDataset
from src.predictors.kde import KDEPredictor
from src.predictors.nominal import NominalPredictor
from src.predictors.noise_drpp import NoiseDRPPPredictor
from src.predictors.oracle import OraclePredictor
from src.predictors.wdrpp import WDRPPLowerPredictor, WDRPPUpperPredictor, gamma0_value
from src.radius.wasserstein_radius import epsilon_n_theorem34
from src.systems.system_c import nominal_drift, true_next_state
from src.viz.plot_per_step_scores import plot_score_curves


def _make_controller(control_mode: str, pid_gains: PIDGains, cfg: ExperimentConfig):
    if control_mode == "zero":
        return ZeroController()
    if control_mode == "pid":
        return PIDController(
            kp=pid_gains.kp,
            ki=pid_gains.ki,
            kd=pid_gains.kd,
            u_max=cfg.u_max,
        )
    raise ValueError(f"Unknown control mode: {control_mode}")


def _write_experiment_config(path: Path, cfg: ExperimentConfig, pid_gains: PIDGains, epsilon_n: float) -> None:
    lines = [
        "system: systemC",
        "steps: 32",
        "step_index: 0..31",
        f"x0: {cfg.x0}",
        f"dataset_seed_master: {cfg.dataset_seed_master}",
        f"eval_seed_master: {cfg.eval_seed_master}",
        f"n_main: {cfg.n_main}",
        f"m_monte_carlo: {cfg.m_monte_carlo}",
        f"u_max: {cfg.u_max}",
        f"nominal_noise_mean: {cfg.nominal_noise_mean}",
        f"nominal_noise_var: {cfg.nominal_noise_var}",
        "control_modes: [zero, pid]",
        f"pid_gains: {{kp: {pid_gains.kp:.8f}, ki: {pid_gains.ki:.8f}, kd: {pid_gains.kd:.8f}}}",
        f"beta: {cfg.radius.beta}",
        f"a: {cfg.radius.a}",
        f"c1: {cfg.radius.c1}",
        f"c2: {cfg.radius.c2}",
        f"epsilon_N: {epsilon_n}",
        f"wdrpp_solver_mode: {cfg.wdrpp_solver_mode}",
        f"wdrpp_lse_integration: {cfg.wdrpp_lse_integration}",
        f"wdrpp_lse_mc_samples: {cfg.wdrpp_lse_mc_samples}",
        f"wdrpp_lse_mc_seed: {cfg.wdrpp_lse_mc_seed}",
        f"adversary_enabled: {cfg.adversary_enabled}",
        f"adversary_noise_id: {cfg.adversary_noise_id}",
        f"adversary_source_noise_ids: [{', '.join(cfg.adversary_source_noise_ids)}]",
        f"adversary_grid_size: {cfg.adversary_grid_size}",
        f"adversary_source_samples: {cfg.adversary_source_samples}",
        f"adversary_support_scale: {cfg.adversary_support_scale}",
        "gamma0: min(0.3*||z||_2, 5) * dt^2, dt=1",
        "w4_rule: truncated_t_df3_then_variance_normalized",
        "oracle_rule: true noise distribution only",
        "kde_rule: robust_silverman_per_step",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_requirement_trace(path: Path) -> None:
    lines = [
        "# Requirement Trace",
        "",
        "1. System C only; steps are fixed to k=0..31 (32 steps).",
        "2. Control modes: zero + PID (LQR excluded this round).",
        "3. PID uses fixed tuned gains and saturation u in [-4,4] with anti-windup.",
        "4. Noise library W1-W6 uses unified variance scale (var=1).",
        "5. W4 uses truncated Student-t(df=3) then variance normalization.",
        "6. Oracle uses true noise distribution only.",
        "7. Nominal baseline uses fixed Gaussian N(nominal_drift, 1).",
        "8. gamma0(z)=min(0.3||z||_2, 5).",
        "9. Wasserstein radius follows Theorem 3.4 / Eq.(8) with beta=0.05.",
        "10. W-DRPP solver mode is configurable: exact or additive-LSE relaxation.",
        "11. Training and evaluation trajectories use different random seed masters (no seed overlap).",
        "12. W7_adversary is a full-step one-step stress test: pooled W1-W6 residuals per control and step, common adversary targeted at WDRPP upper, no Oracle.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_pooled_datasets(
    cfg: ExperimentConfig,
    control_mode: str,
    source_noise_ids: List[str],
) -> DatasetsByStep:
    """
    Pool step-wise datasets across a configurable list of source noises.

    This is intentionally isolated so W7 can later be redefined by changing
    source_noise_ids or by replacing this function with a different data source.
    """
    if not source_noise_ids:
        raise ValueError("source_noise_ids must be non-empty.")

    datasets_by_noise = [
        load_historical_dataset(cfg=cfg, control_mode=control_mode, noise_id=noise_id, n=cfg.n_main)
        for noise_id in source_noise_ids
    ]
    pooled: DatasetsByStep = {}
    for k in range(cfg.steps):
        pooled[k] = StepDataset(
            x_k=np.concatenate([datasets[k].x_k for datasets in datasets_by_noise]),
            u_k=np.concatenate([datasets[k].u_k for datasets in datasets_by_noise]),
            x_next=np.concatenate([datasets[k].x_next for datasets in datasets_by_noise]),
            w_hat=np.concatenate([datasets[k].w_hat for datasets in datasets_by_noise]),
        )
    return pooled


def _select_adversary_sources(w_hat: np.ndarray, max_points: int) -> np.ndarray:
    values = np.asarray(w_hat, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot build adversary from empty residual samples.")
    if max_points <= 0 or values.size <= max_points:
        return np.sort(values)

    order = np.argsort(values)
    idx = np.linspace(0, values.size - 1, int(max_points), dtype=int)
    return values[order[idx]]


def _build_adversary_grid(
    w_hat_full: np.ndarray,
    source_points: np.ndarray,
    epsilon: float,
    r_current: float,
    cfg: ExperimentConfig,
) -> np.ndarray:
    values = np.asarray(w_hat_full, dtype=float)
    sigma = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    support_pad = float(cfg.adversary_support_scale) * (sigma + float(epsilon) + float(r_current))
    if support_pad <= 0.0:
        support_pad = max(float(epsilon), 1e-3)

    left = float(np.min(values)) - support_pad
    right = float(np.max(values)) + support_pad
    if right <= left:
        right = left + max(2.0 * support_pad, 1e-3)

    grid_size = max(int(cfg.adversary_grid_size), 20)
    base_grid = np.linspace(left, right, grid_size)
    clipped_sources = np.clip(np.asarray(source_points, dtype=float), left, right)
    grid = np.unique(np.concatenate([base_grid, clipped_sources, np.array([left, right])]))
    return np.sort(grid)


def _solve_adversary_transport_lp(
    source_points: np.ndarray,
    grid: np.ndarray,
    losses: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, float, float]:
    """
    Solve the discrete one-step Wasserstein adversary LP.

    Returns:
        q_adv: destination probability mass on grid
        objective_value: adversarial expected log-score
        transport_cost: realized transport cost
    """
    source = np.asarray(source_points, dtype=float).reshape(-1)
    y = np.asarray(grid, dtype=float).reshape(-1)
    loss = np.asarray(losses, dtype=float).reshape(-1)
    if source.size == 0 or y.size == 0:
        raise ValueError("source_points and grid must be non-empty.")
    if loss.size != y.size:
        raise ValueError("losses must have the same length as grid.")

    n = source.size
    m = y.size
    num_vars = n * m

    c = np.tile(loss, n)
    row_idx = np.repeat(np.arange(n), m)
    col_idx = np.arange(num_vars)
    a_eq = csr_matrix((np.ones(num_vars), (row_idx, col_idx)), shape=(n, num_vars))
    b_eq = np.full(n, 1.0 / n)

    transport = np.abs(y[None, :] - source[:, None]).reshape(1, -1)
    a_ub = csr_matrix(transport)
    b_ub = np.array([float(epsilon)])

    opt = linprog(
        c=c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=(0.0, None),
        method="highs",
    )
    if not opt.success:
        raise RuntimeError(f"Adversary LP failed: {opt.message}")

    pi = np.asarray(opt.x, dtype=float).reshape(n, m)
    q_adv = pi.sum(axis=0)
    transport_cost = float(np.sum(pi * transport.reshape(n, m)))
    objective_value = float(np.dot(q_adv, loss))
    return q_adv, objective_value, transport_cost


def _weighted_logscore_on_grid(
    pred,
    step: int,
    x_eval: float,
    u_eval: float,
    grid: np.ndarray,
    weights: np.ndarray,
) -> float:
    x_nom = nominal_drift(x_eval, u_eval)
    scores = np.array(
        [pred.logpdf(step=step, x_next=float(x_nom + w), x_k=x_eval, u_k=u_eval) for w in grid],
        dtype=float,
    )
    return float(np.dot(weights, scores))


def _run_w7_adversary_scenario(
    cfg: ExperimentConfig,
    control_mode: str,
    figures_dir: Path,
    per_step_rows: List[List[object]],
    summary_rows: List[List[object]],
) -> None:
    pooled = _load_pooled_datasets(
        cfg=cfg,
        control_mode=control_mode,
        source_noise_ids=list(cfg.adversary_source_noise_ids),
    )
    pool_n = int(len(pooled[0].w_hat))
    epsilon_pool = epsilon_n_theorem34(pool_n, cfg.radius)

    predictors = [
        NominalPredictor(cfg),
        NoiseDRPPPredictor(pooled, cfg),
        KDEPredictor(pooled, cfg),
        WDRPPUpperPredictor(pooled, epsilon_pool, cfg),
        WDRPPLowerPredictor(pooled, epsilon_pool, cfg),
    ]
    target_pred = next(p for p in predictors if p.name == "wdrpp_upper")

    steps = list(range(cfg.steps))
    method_to_mean: Dict[str, np.ndarray] = {p.name: np.zeros(cfg.steps, dtype=float) for p in predictors}
    method_to_q025: Dict[str, np.ndarray] = {p.name: np.zeros(cfg.steps, dtype=float) for p in predictors}
    method_to_q975: Dict[str, np.ndarray] = {p.name: np.zeros(cfg.steps, dtype=float) for p in predictors}

    for k in steps:
        data = pooled[k]
        x_eval = float(np.mean(data.x_k))
        u_eval = float(np.mean(data.u_k))
        r_current = float(np.sqrt(max(gamma0_value(x_eval, u_eval), 0.0)))

        source_points = _select_adversary_sources(
            w_hat=data.w_hat,
            max_points=int(cfg.adversary_source_samples),
        )
        grid = _build_adversary_grid(
            w_hat_full=data.w_hat,
            source_points=source_points,
            epsilon=epsilon_pool,
            r_current=r_current,
            cfg=cfg,
        )
        target_losses = np.array(
            [
                target_pred.logpdf(
                    step=k,
                    x_next=float(nominal_drift(x_eval, u_eval) + w),
                    x_k=x_eval,
                    u_k=u_eval,
                )
                for w in grid
            ],
            dtype=float,
        )
        q_adv, _, _ = _solve_adversary_transport_lp(
            source_points=source_points,
            grid=grid,
            losses=target_losses,
            epsilon=epsilon_pool,
        )

        for pred in predictors:
            score = _weighted_logscore_on_grid(
                pred=pred,
                step=k,
                x_eval=x_eval,
                u_eval=u_eval,
                grid=grid,
                weights=q_adv,
            )
            method_to_mean[pred.name][k] = score
            method_to_q025[pred.name][k] = score
            method_to_q975[pred.name][k] = score
            per_step_rows.append(
                [
                    control_mode,
                    cfg.adversary_noise_id,
                    k,
                    pred.name,
                    score,
                    0.0,
                    score,
                    score,
                ]
            )

    for pred in predictors:
        overall = float(np.mean(method_to_mean[pred.name]))
        summary_rows.append([control_mode, cfg.adversary_noise_id, pred.name, overall])

    fig_path = figures_dir / f"score_curves_control_{control_mode}_noise_{cfg.adversary_noise_id}.png"
    plot_score_curves(
        steps=steps,
        method_to_mean=method_to_mean,
        method_to_q025=method_to_q025,
        method_to_q975=method_to_q975,
        title=(
            f"System C | control={control_mode} | noise={cfg.adversary_noise_id} "
            f"| pooled={'+'.join(cfg.adversary_source_noise_ids)} | N={pool_n}"
        ),
        output_path=fig_path,
        y_q_low=cfg.plot_y_q_low,
        y_q_high=cfg.plot_y_q_high,
        y_padding_ratio=cfg.plot_y_padding_ratio,
    )


def run_experiment(cfg: ExperimentConfig, skip_dataset_build: bool = False) -> Path:
    noise_lib = create_noise_library(cfg.w4_truncation_L)
    pid_gains = tune_pid(cfg)

    if not skip_dataset_build:
        build_historical_datasets(cfg=cfg, noise_lib=noise_lib, pid_gains=pid_gains)

    epsilon_n = epsilon_n_theorem34(cfg.n_main, cfg.radius)

    run_id = dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = cfg.results_root / run_id
    figures_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    _write_experiment_config(run_dir / "experiment_config.yaml", cfg, pid_gains, epsilon_n)
    _write_requirement_trace(run_dir / "requirement_trace.md")

    per_step_rows: List[List[object]] = []
    summary_rows: List[List[object]] = []
    steps = list(range(cfg.steps))

    for control_mode in cfg.control_modes:
        for noise_id in cfg.noise_ids:
            datasets = load_historical_dataset(cfg=cfg, control_mode=control_mode, noise_id=noise_id, n=cfg.n_main)
            noise_model: NoiseModel = noise_lib[noise_id]

            predictors = [
                NominalPredictor(cfg),
                NoiseDRPPPredictor(datasets, cfg),
                KDEPredictor(datasets, cfg),
                OraclePredictor(noise_model, cfg),
                WDRPPUpperPredictor(datasets, epsilon_n, cfg),
                WDRPPLowerPredictor(datasets, epsilon_n, cfg),
            ]

            score_buckets: Dict[str, List[List[float]]] = {
                p.name: [[] for _ in range(cfg.steps)] for p in predictors
            }

            control_idx = cfg.control_modes.index(control_mode)
            noise_idx = cfg.noise_ids.index(noise_id)

            for m in range(cfg.m_monte_carlo):
                seed_m = (
                    cfg.eval_seed_master
                    + m
                    + control_idx * 1_000_000
                    + noise_idx * 100_000
                )
                rng = np.random.default_rng(seed_m)
                ctrl = _make_controller(control_mode, pid_gains, cfg)
                x = cfg.x0
                for k in range(cfg.steps):
                    u = float(ctrl.control(x, k))
                    w = float(noise_model.sample(rng, 1)[0])
                    x_next = float(true_next_state(x, u, w))

                    for pred in predictors:
                        score = pred.logpdf(step=k, x_next=x_next, x_k=x, u_k=u)
                        score_buckets[pred.name][k].append(float(score))

                    x = x_next

            method_to_mean: Dict[str, np.ndarray] = {}
            method_to_q025: Dict[str, np.ndarray] = {}
            method_to_q975: Dict[str, np.ndarray] = {}
            for pred in predictors:
                means = np.zeros(cfg.steps, dtype=float)
                stds = np.zeros(cfg.steps, dtype=float)
                q025s = np.zeros(cfg.steps, dtype=float)
                q975s = np.zeros(cfg.steps, dtype=float)
                for k in range(cfg.steps):
                    score_arr = np.asarray(score_buckets[pred.name][k], dtype=float)
                    mean_k, std_k = summarize_logscores(score_arr)
                    q025_k = float(np.quantile(score_arr, 0.025))
                    q975_k = float(np.quantile(score_arr, 0.975))
                    means[k] = mean_k
                    stds[k] = std_k
                    q025s[k] = q025_k
                    q975s[k] = q975_k
                    per_step_rows.append(
                        [
                            control_mode,
                            noise_id,
                            k,
                            pred.name,
                            float(mean_k),
                            float(std_k),
                            q025_k,
                            q975_k,
                        ]
                    )

                overall = float(np.mean(means))
                summary_rows.append([control_mode, noise_id, pred.name, overall])
                method_to_mean[pred.name] = means
                method_to_q025[pred.name] = q025s
                method_to_q975[pred.name] = q975s

            fig_path = figures_dir / f"score_curves_control_{control_mode}_noise_{noise_id}.png"
            plot_score_curves(
                steps=steps,
                method_to_mean=method_to_mean,
                method_to_q025=method_to_q025,
                method_to_q975=method_to_q975,
                title=f"System C | control={control_mode} | noise={noise_id} | N={cfg.n_main} | M={cfg.m_monte_carlo}",
                output_path=fig_path,
                y_q_low=cfg.plot_y_q_low,
                y_q_high=cfg.plot_y_q_high,
                y_padding_ratio=cfg.plot_y_padding_ratio,
            )

        if cfg.adversary_enabled:
            _run_w7_adversary_scenario(
                cfg=cfg,
                control_mode=control_mode,
                figures_dir=figures_dir,
                per_step_rows=per_step_rows,
                summary_rows=summary_rows,
            )

    with (run_dir / "per_step_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "control_mode",
                "noise_id",
                "step",
                "method",
                "mean_logscore",
                "std_logscore",
                "q025_logscore",
                "q975_logscore",
            ]
        )
        writer.writerows(per_step_rows)

    with (run_dir / "summary_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["control_mode", "noise_id", "method", "overall_mean_logscore"])
        writer.writerows(summary_rows)

    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 1D W-DRPP experiment-2 on system C.")
    parser.add_argument("--m", type=int, default=1000, help="Monte Carlo repeats.")
    parser.add_argument("--steps", type=int, default=32, help="Prediction steps (should be 32).")
    parser.add_argument("--n-main", type=int, default=100, help="Main training sample size for experiment 2.")
    parser.add_argument("--beta", type=float, default=0.05, help="Confidence parameter for epsilon_N(beta).")
    parser.add_argument("--a", type=float, default=1.5, help="Light-tail exponent a (>1).")
    parser.add_argument("--c1", type=float, default=2.0, help="Theorem constant c1.")
    parser.add_argument("--c2", type=float, default=1.0, help="Theorem constant c2.")
    parser.add_argument("--dataset-seed-master", type=int, default=20260412, help="Master seed for training datasets.")
    parser.add_argument("--eval-seed-master", type=int, default=30260412, help="Master seed for evaluation trajectories.")
    parser.add_argument("--datasets-root", type=str, default="datasets_1d_wdrpp", help="Dataset root directory.")
    parser.add_argument("--results-root", type=str, default="results_1d_wdrpp", help="Results root directory.")
    parser.add_argument(
        "--wdrpp-solver-mode",
        type=str,
        default="exact",
        choices=["exact", "lse"],
        help="W-DRPP solver mode for both upper/lower predictors.",
    )
    parser.add_argument(
        "--wdrpp-lse-integration",
        type=str,
        default="closed_form",
        choices=["closed_form", "mc"],
        help="Integration method used by LSE solver.",
    )
    parser.add_argument(
        "--wdrpp-lse-mc-samples",
        type=int,
        default=2000,
        help="MC sample count for LSE integration when --wdrpp-lse-integration=mc.",
    )
    parser.add_argument(
        "--wdrpp-lse-mc-seed",
        type=int,
        default=20260501,
        help="Base random seed for LSE MC integration.",
    )
    parser.add_argument("--skip-adversary", action="store_true", help="Skip W7 adversary stress-test plots.")
    parser.add_argument(
        "--adversary-source-noises",
        type=str,
        default="W1,W2,W3,W4,W5,W6",
        help="Comma-separated source noises pooled to build W7_adversary.",
    )
    parser.add_argument(
        "--adversary-grid-size",
        type=int,
        default=400,
        help="Grid size for the one-step adversary transport LP.",
    )
    parser.add_argument(
        "--adversary-source-samples",
        type=int,
        default=120,
        help="Maximum pooled residual source points used in each adversary LP.",
    )
    parser.add_argument(
        "--adversary-support-scale",
        type=float,
        default=2.0,
        help="Support padding scale c_B for W7_adversary.",
    )
    parser.add_argument("--skip-dataset-build", action="store_true", help="Skip historical dataset generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = ExperimentConfig()
    radius_cfg = replace(base_cfg.radius, beta=float(args.beta), a=float(args.a), c1=float(args.c1), c2=float(args.c2))
    adversary_source_noise_ids = [
        token.strip()
        for token in str(args.adversary_source_noises).split(",")
        if token.strip()
    ]
    cfg = replace(
        base_cfg,
        m_monte_carlo=int(args.m),
        steps=int(args.steps),
        n_main=int(args.n_main),
        dataset_seed_master=int(args.dataset_seed_master),
        eval_seed_master=int(args.eval_seed_master),
        datasets_root=Path(args.datasets_root),
        results_root=Path(args.results_root),
        wdrpp_solver_mode=str(args.wdrpp_solver_mode),
        wdrpp_lse_integration=str(args.wdrpp_lse_integration),
        wdrpp_lse_mc_samples=int(args.wdrpp_lse_mc_samples),
        wdrpp_lse_mc_seed=int(args.wdrpp_lse_mc_seed),
        adversary_enabled=not bool(args.skip_adversary),
        adversary_source_noise_ids=adversary_source_noise_ids,
        adversary_grid_size=int(args.adversary_grid_size),
        adversary_source_samples=int(args.adversary_source_samples),
        adversary_support_scale=float(args.adversary_support_scale),
        radius=radius_cfg,
    )
    if cfg.steps != 32:
        raise ValueError("This experiment design fixes steps to 32 (k=0..31).")
    run_dir = run_experiment(cfg=cfg, skip_dataset_build=bool(args.skip_dataset_build))
    print(f"Experiment finished. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()

