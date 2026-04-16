from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.special import logsumexp

from src.systems.system_c import nominal_drift


@dataclass(frozen=True)
class StepDataset:
    x_k: np.ndarray
    u_k: np.ndarray
    x_next: np.ndarray
    w_hat: np.ndarray


def safe_log(value: float, floor: float) -> float:
    """安全对数，避免 -inf。"""
    if value <= 0.0 or not np.isfinite(value):
        return floor
    out = float(np.log(value))
    return max(out, floor)


def gaussian_mixture_logpdf(x: float, centers: np.ndarray, bandwidth: float, floor: float) -> float:
    """一维等权高斯混合 logpdf。"""
    if centers.size == 0:
        return floor
    h = max(float(bandwidth), 1e-8)
    z = -0.5 * ((x - centers) / h) ** 2 - np.log(h) - 0.5 * np.log(2.0 * np.pi)
    return max(float(logsumexp(z) - np.log(centers.size)), floor)


class PredictorBase:
    name: str

    def logpdf(self, step: int, x_next: float, x_k: float, u_k: float) -> float:
        raise NotImplementedError


def residual_to_nominal(x_next: float, x_k: float, u_k: float) -> float:
    """把状态空间点评价转成噪声空间点评价。"""
    return x_next - nominal_drift(x_k, u_k)


DatasetsByStep = Dict[int, StepDataset]

