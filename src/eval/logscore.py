from __future__ import annotations

import numpy as np


def summarize_logscores(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=1) if arr.size > 1 else 0.0)

