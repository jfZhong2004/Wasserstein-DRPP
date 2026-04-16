from __future__ import annotations

import math

from src.config import ExperimentConfig
from src.predictors.common import PredictorBase
from src.systems.system_c import nominal_drift


class NominalPredictor(PredictorBase):
    """
    Nominal baseline:
    p_k(x) = N(nominal_drift(x_k,u_k) + mu0, sigma0^2)
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.name = "nominal"
        self.floor = cfg.logpdf_floor
        self.mu0 = float(cfg.nominal_noise_mean)
        self.var0 = max(float(cfg.nominal_noise_var), 1e-12)
        self.sigma0 = math.sqrt(self.var0)

    def logpdf(self, step: int, x_next: float, x_k: float, u_k: float) -> float:  # noqa: ARG002
        mean = nominal_drift(x_k, u_k) + self.mu0
        z = (x_next - mean) / self.sigma0
        logp = -0.5 * z * z - math.log(self.sigma0) - 0.5 * math.log(2.0 * math.pi)
        return max(float(logp), self.floor)

