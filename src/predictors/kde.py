from __future__ import annotations

from typing import Dict

import numpy as np

from src.config import ExperimentConfig
from src.predictors.common import DatasetsByStep, PredictorBase, gaussian_mixture_logpdf
from src.systems.system_c import nominal_drift


def robust_silverman_bandwidth(samples: np.ndarray) -> float:
    n = samples.size
    if n <= 1:
        return 1e-3
    sigma = float(np.std(samples, ddof=1))
    q75, q25 = np.percentile(samples, [75, 25])
    iqr = float(q75 - q25)
    robust_scale = min(sigma, iqr / 1.34) if iqr > 0 else sigma
    if robust_scale <= 1e-12:
        return 1e-3
    return max(0.9 * robust_scale * (n ** (-1.0 / 5.0)), 1e-3)


class KDEPredictor(PredictorBase):
    """Gaussian KDE baseline。"""

    def __init__(self, datasets: DatasetsByStep, cfg: ExperimentConfig) -> None:
        self.name = "kde"
        self.floor = cfg.logpdf_floor
        self.w_by_step: Dict[int, np.ndarray] = {}
        self.h_by_step: Dict[int, float] = {}
        for k, data in datasets.items():
            self.w_by_step[k] = data.w_hat
            self.h_by_step[k] = robust_silverman_bandwidth(data.w_hat)

    def logpdf(self, step: int, x_next: float, x_k: float, u_k: float) -> float:
        centers = nominal_drift(x_k, u_k) + self.w_by_step[step]
        h = self.h_by_step[step]
        return gaussian_mixture_logpdf(x_next, centers, h, self.floor)

