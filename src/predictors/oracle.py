from __future__ import annotations

import numpy as np

from src.config import ExperimentConfig
from src.noise_lib.noise_w1_w6 import NoiseModel
from src.predictors.common import PredictorBase, safe_log
from src.systems.system_c import true_drift


class OraclePredictor(PredictorBase):
    """Oracle：直接使用真实噪声分布。"""

    def __init__(self, noise_model: NoiseModel, cfg: ExperimentConfig) -> None:
        self.name = "oracle"
        self.noise_model = noise_model
        self.floor = cfg.logpdf_floor

    def logpdf(self, step: int, x_next: float, x_k: float, u_k: float) -> float:  # noqa: ARG002
        residual = x_next - true_drift(x_k, u_k)
        density = float(self.noise_model.pdf(np.array([residual]))[0])
        return safe_log(density, self.floor)

