from __future__ import annotations

import argparse
import csv
import datetime as dt
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.config import ExperimentConfig
from src.controllers.pid import PIDController
from src.controllers.pid_tuning import PIDGains, tune_pid
from src.controllers.zero import ZeroController
from src.data.build_dataset import build_historical_datasets, load_historical_dataset
from src.eval.logscore import summarize_logscores
from src.noise_lib.noise_w1_w6 import NoiseModel, create_noise_library
from src.predictors.kde import KDEPredictor
from src.predictors.nominal import NominalPredictor
from src.predictors.noise_drpp import NoiseDRPPPredictor
from src.predictors.oracle import OraclePredictor
from src.predictors.wdrpp import WDRPPLowerPredictor, WDRPPUpperPredictor
from src.radius.wasserstein_radius import epsilon_n_theorem34
from src.systems.system_c import true_next_state
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
        "10. W-DRPP solver reuses src/solvers/drpp_1d_exact_solver.py directly.",
        "11. Training and evaluation trajectories use different random seed masters (no seed overlap).",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
            for pred in predictors:
                means = np.zeros(cfg.steps, dtype=float)
                stds = np.zeros(cfg.steps, dtype=float)
                for k in range(cfg.steps):
                    mean_k, std_k = summarize_logscores(np.asarray(score_buckets[pred.name][k], dtype=float))
                    means[k] = mean_k
                    stds[k] = std_k
                    per_step_rows.append(
                        [control_mode, noise_id, k, pred.name, float(mean_k), float(std_k)]
                    )

                overall = float(np.mean(means))
                summary_rows.append([control_mode, noise_id, pred.name, overall])
                method_to_mean[pred.name] = means

            fig_path = figures_dir / f"score_curves_control_{control_mode}_noise_{noise_id}.png"
            plot_score_curves(
                steps=steps,
                method_to_mean=method_to_mean,
                title=f"System C | control={control_mode} | noise={noise_id} | N={cfg.n_main} | M={cfg.m_monte_carlo}",
                output_path=fig_path,
                y_q_low=cfg.plot_y_q_low,
                y_q_high=cfg.plot_y_q_high,
                y_padding_ratio=cfg.plot_y_padding_ratio,
            )

    with (run_dir / "per_step_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["control_mode", "noise_id", "step", "method", "mean_logscore", "std_logscore"])
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
    parser.add_argument("--skip-dataset-build", action="store_true", help="Skip historical dataset generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = ExperimentConfig()
    radius_cfg = replace(base_cfg.radius, beta=float(args.beta), a=float(args.a), c1=float(args.c1), c2=float(args.c2))
    cfg = replace(
        base_cfg,
        m_monte_carlo=int(args.m),
        steps=int(args.steps),
        n_main=int(args.n_main),
        dataset_seed_master=int(args.dataset_seed_master),
        eval_seed_master=int(args.eval_seed_master),
        datasets_root=Path(args.datasets_root),
        results_root=Path(args.results_root),
        radius=radius_cfg,
    )
    if cfg.steps != 32:
        raise ValueError("This experiment design fixes steps to 32 (k=0..31).")
    run_dir = run_experiment(cfg=cfg, skip_dataset_build=bool(args.skip_dataset_build))
    print(f"Experiment finished. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()

