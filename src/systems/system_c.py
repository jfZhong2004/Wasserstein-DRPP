from __future__ import annotations

import math


def true_drift(x: float, u: float) -> float:
    """系统 C 的真实无噪声漂移项。"""
    return math.sin(x) + 0.5 * u


def nominal_drift(x: float, u: float) -> float:
    """系统 C 的标称模型（sin 的三阶泰勒截断）。"""
    return x - (x**3) / 6.0 + 0.5 * u


def true_next_state(x: float, u: float, w: float) -> float:
    """真实系统一步推进。"""
    return true_drift(x, u) + w

