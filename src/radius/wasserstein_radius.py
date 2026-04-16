from __future__ import annotations

import math

from src.config import RadiusConfig


def epsilon_n_theorem34(n: int, cfg: RadiusConfig) -> float:
    """
    按 Theorem 3.4 / Eq. (8) 计算 epsilon_N(beta)。
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if not (0.0 < cfg.beta < 1.0):
        raise ValueError("beta must be in (0,1).")
    if cfg.a <= 1.0:
        raise ValueError("a must be > 1.")
    if cfg.c1 <= cfg.beta:
        raise ValueError("c1 must be greater than beta so log(c1/beta)>0.")
    if cfg.c2 <= 0.0:
        raise ValueError("c2 must be positive.")

    log_term = math.log(cfg.c1 / cfg.beta)
    threshold = log_term / cfg.c2
    t = log_term / (cfg.c2 * n)

    if n >= threshold:
        exponent = 1.0 / max(cfg.m, 2)
    else:
        exponent = 1.0 / cfg.a
    return t**exponent

