from __future__ import annotations

import math
from typing import Dict

import numpy as np

from src.config import ExperimentConfig
from src.predictors.common import DatasetsByStep, PredictorBase, safe_log
from src.systems.system_c import nominal_drift


class NoiseDRPPPredictor(PredictorBase):
    """
    Moment-DRPP / Noise-DRPP（单峰高斯）。
    """

    def __init__(self, datasets: DatasetsByStep, cfg: ExperimentConfig) -> None:
        self.name = "noise_drpp"
        self.floor = cfg.logpdf_floor
        self.gamma2 = cfg.gamma2_noise_drpp
        self.mu_by_step: Dict[int, float] = {}
        self.var_by_step: Dict[int, float] = {}

        for k, data in datasets.items():
            mu = float(np.mean(data.w_hat))
            var = float(np.var(data.w_hat, ddof=1)) if data.w_hat.size > 1 else 1e-4
            self.mu_by_step[k] = mu
            self.var_by_step[k] = max(var, 1e-6)

    def logpdf(self, step: int, x_next: float, x_k: float, u_k: float) -> float:
        mu = self.mu_by_step[step]
        var = self.var_by_step[step] * self.gamma2
        sigma = math.sqrt(max(var, 1e-12))
        mean = nominal_drift(x_k, u_k) + mu
        z = (x_next - mean) / sigma
        logp = -0.5 * z * z - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)
        return max(float(logp), self.floor)

