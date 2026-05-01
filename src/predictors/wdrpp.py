from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np

from src.solvers.drpp_1d_exact_solver import DRPP1DSolution, solve_drpp_1d_exact
from src.solvers.drpp_lse_solver import DRPPLSESolution, solve_drpp_lse
from src.config import ExperimentConfig
from src.predictors.common import DatasetsByStep, PredictorBase, residual_to_nominal


def gamma0_value(x: float, u: float) -> float:
    """按 DRPP 约定的 gamma0(z)。"""
    z_norm = float(np.sqrt(x * x + u * u))
    return min(0.3 * z_norm, 5.0)


DRPPSolution = Union[DRPP1DSolution, DRPPLSESolution]


def _solve_by_mode(
    centers: np.ndarray,
    epsilon: float,
    radii: list[float] | None,
    cfg: ExperimentConfig,
    seed_offset: int,
) -> DRPPSolution:
    mode = cfg.wdrpp_solver_mode.strip().lower()
    if mode == "exact":
        return solve_drpp_1d_exact(
            centers=centers.tolist(),
            epsilon=epsilon,
            radii=radii,
            maxiter=500,
            tol=1e-9,
        )
    if mode == "lse":
        return solve_drpp_lse(
            centers=centers.tolist(),
            epsilon=epsilon,
            radii=radii,
            integration_method=cfg.wdrpp_lse_integration,
            mc_samples=int(cfg.wdrpp_lse_mc_samples),
            mc_seed=int(cfg.wdrpp_lse_mc_seed + seed_offset),
            maxiter=int(cfg.wdrpp_lse_maxiter),
            tol=float(cfg.wdrpp_lse_tol),
        )
    raise ValueError(f"Unknown wdrpp_solver_mode: {cfg.wdrpp_solver_mode}")


class WDRPPUpperPredictor(PredictorBase):
    """W-DRPP 上界（尖顶核）。"""

    def __init__(self, datasets: DatasetsByStep, epsilon: float, cfg: ExperimentConfig) -> None:
        self.name = "wdrpp_upper"
        self.floor = cfg.logpdf_floor
        self.sol_by_step: Dict[int, DRPPSolution] = {}
        for k, data in datasets.items():
            sol = _solve_by_mode(
                centers=data.w_hat,
                epsilon=epsilon,
                radii=None,
                cfg=cfg,
                seed_offset=10_000 * int(k),
            )
            self.sol_by_step[k] = sol

    def logpdf(self, step: int, x_next: float, x_k: float, u_k: float) -> float:
        residual = residual_to_nominal(x_next=x_next, x_k=x_k, u_k=u_k)
        out = self.sol_by_step[step].log_pdf(residual)
        return max(float(out), self.floor)


class WDRPPLowerPredictor(PredictorBase):
    """W-DRPP 下界（平顶核）。"""

    def __init__(self, datasets: DatasetsByStep, epsilon: float, cfg: ExperimentConfig) -> None:
        self.name = "wdrpp_lower"
        self.floor = cfg.logpdf_floor
        self.epsilon = epsilon
        self.quant = max(cfg.lower_r_quant, 1e-3)
        self.cfg = cfg

        self.w_by_step: Dict[int, np.ndarray] = {}
        self.base_r_by_step: Dict[int, np.ndarray] = {}
        for k, data in datasets.items():
            self.w_by_step[k] = data.w_hat
            gamma_hat = np.array([gamma0_value(x, u) for x, u in zip(data.x_k, data.u_k)], dtype=float)
            self.base_r_by_step[k] = np.sqrt(np.maximum(gamma_hat, 0.0))

        self.cache: Dict[Tuple[int, float], DRPPSolution] = {}

    def _quantize(self, r: float) -> float:
        return round(r / self.quant) * self.quant

    def _get_solution(self, step: int, r_current: float) -> DRPPSolution:
        rq = self._quantize(r_current)
        key = (step, rq)
        if key not in self.cache:
            radii = (self.base_r_by_step[step] + rq).tolist()
            seed_offset = int(step * 1_000_000 + round(rq / self.quant))
            sol = _solve_by_mode(
                centers=self.w_by_step[step],
                epsilon=self.epsilon,
                radii=radii,
                cfg=self.cfg,
                seed_offset=seed_offset,
            )
            self.cache[key] = sol
        return self.cache[key]

    def logpdf(self, step: int, x_next: float, x_k: float, u_k: float) -> float:
        r_current = float(np.sqrt(max(gamma0_value(x_k, u_k), 0.0)))
        sol = self._get_solution(step=step, r_current=r_current)
        residual = residual_to_nominal(x_next=x_next, x_k=x_k, u_k=u_k)
        out = sol.log_pdf(residual)
        return max(float(out), self.floor)

