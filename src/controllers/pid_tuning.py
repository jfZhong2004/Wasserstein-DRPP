from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
from scipy.optimize import minimize

from src.config import ExperimentConfig
from src.controllers.pid import PIDController
from src.systems.system_c import nominal_drift


@dataclass(frozen=True)
class PIDGains:
    kp: float
    ki: float
    kd: float


def _frange(start: float, stop: float, step: float) -> Iterable[float]:
    n = int(round((stop - start) / step))
    for i in range(n + 1):
        yield start + i * step


def _pid_objective(gains: Tuple[float, float, float], cfg: ExperimentConfig) -> float:
    kp, ki, kd = gains
    if kp < cfg.pid_tuning.kp_min or kp > cfg.pid_tuning.kp_max:
        return 1e12
    if ki < cfg.pid_tuning.ki_min or ki > cfg.pid_tuning.ki_max:
        return 1e12
    if kd < cfg.pid_tuning.kd_min or kd > cfg.pid_tuning.kd_max:
        return 1e12

    ctrl = PIDController(kp=kp, ki=ki, kd=kd, u_max=cfg.u_max)
    x = cfg.x0
    total = 0.0
    for k in range(cfg.steps):
        u = ctrl.control(x=x, k=k)
        x_next = nominal_drift(x, u)
        total += x_next * x_next + cfg.pid_tuning.objective_u_weight * (u * u)
        x = x_next
    return total


def tune_pid(cfg: ExperimentConfig) -> PIDGains:
    """
    两阶段 PID 自动整定：
    1) 网格粗搜
    2) Nelder-Mead 局部细化
    """
    best = (0.0, 0.0, 0.0)
    best_obj = float("inf")

    for kp in _frange(cfg.pid_tuning.kp_min, cfg.pid_tuning.kp_max, cfg.pid_tuning.kp_step):
        for ki in _frange(cfg.pid_tuning.ki_min, cfg.pid_tuning.ki_max, cfg.pid_tuning.ki_step):
            for kd in _frange(cfg.pid_tuning.kd_min, cfg.pid_tuning.kd_max, cfg.pid_tuning.kd_step):
                obj = _pid_objective((kp, ki, kd), cfg)
                if obj < best_obj:
                    best_obj = obj
                    best = (kp, ki, kd)

    x0 = np.array(best, dtype=float)

    def local_objective(x: np.ndarray) -> float:
        return _pid_objective((float(x[0]), float(x[1]), float(x[2])), cfg)

    res = minimize(
        local_objective,
        x0=x0,
        method="Nelder-Mead",
        options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-5, "disp": False},
    )

    kp, ki, kd = (float(v) for v in res.x)
    kp = min(max(kp, cfg.pid_tuning.kp_min), cfg.pid_tuning.kp_max)
    ki = min(max(ki, cfg.pid_tuning.ki_min), cfg.pid_tuning.ki_max)
    kd = min(max(kd, cfg.pid_tuning.kd_min), cfg.pid_tuning.kd_max)
    return PIDGains(kp=kp, ki=ki, kd=kd)

