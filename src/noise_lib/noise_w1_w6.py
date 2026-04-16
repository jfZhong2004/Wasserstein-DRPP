from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
from scipy.stats import norm, t


SQRT_2PI = np.sqrt(2.0 * np.pi)


@dataclass(frozen=True)
class NoiseModel:
    name: str
    sample_fn: Callable[[np.random.Generator, int], np.ndarray]
    pdf_fn: Callable[[np.ndarray], np.ndarray]
    description: str

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return self.sample_fn(rng, size)

    def pdf(self, x: np.ndarray | float) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        return self.pdf_fn(arr)


def _normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * SQRT_2PI)


def create_noise_library(w4_truncation_L: float) -> Dict[str, NoiseModel]:
    """
    创建 W1-W6 噪声模型，统一保证方差为 1。
    """
    lib: Dict[str, NoiseModel] = {}

    # W1: N(0,1)
    lib["W1"] = NoiseModel(
        name="W1",
        sample_fn=lambda rng, size: rng.normal(loc=0.0, scale=1.0, size=size),
        pdf_fn=lambda x: _normal_pdf(x, mu=0.0, sigma=1.0),
        description="Standard Gaussian N(0,1).",
    )

    # W2 base variance = 4.25, scale to variance 1.
    s2 = np.sqrt(4.25)

    def sample_w2(rng: np.random.Generator, size: int) -> np.ndarray:
        comp = rng.random(size) < 0.5
        z = np.empty(size, dtype=float)
        z[comp] = rng.normal(-2.0, 0.5, comp.sum())
        z[~comp] = rng.normal(2.0, 0.5, (~comp).sum())
        return z / s2

    def pdf_w2(x: np.ndarray) -> np.ndarray:
        y = x * s2
        return (0.5 * _normal_pdf(y, -2.0, 0.5) + 0.5 * _normal_pdf(y, 2.0, 0.5)) * s2

    lib["W2"] = NoiseModel("W2", sample_w2, pdf_w2, "Bimodal Gaussian mixture (variance normalized).")

    # W3 base variance = 2.617, scale to variance 1.
    s3 = np.sqrt(2.617)

    def sample_w3(rng: np.random.Generator, size: int) -> np.ndarray:
        comp = rng.random(size) < 0.3
        z = np.empty(size, dtype=float)
        z[comp] = rng.normal(-1.0, 0.3, comp.sum())
        z[~comp] = rng.normal(2.0, 1.0, (~comp).sum())
        return z / s3

    def pdf_w3(x: np.ndarray) -> np.ndarray:
        y = x * s3
        return (0.3 * _normal_pdf(y, -1.0, 0.3) + 0.7 * _normal_pdf(y, 2.0, 1.0)) * s3

    lib["W3"] = NoiseModel("W3", sample_w3, pdf_w3, "Asymmetric Gaussian mixture (variance normalized).")

    # W4: truncated Student-t(df=3), then scaled to variance 1.
    df = 3.0
    L = float(w4_truncation_L)
    z_norm = t.cdf(L, df=df) - t.cdf(-L, df=df)
    second_moment = t.expect(lambda u: u * u, args=(df,), lb=-L, ub=L) / z_norm
    s4 = np.sqrt(second_moment)

    def sample_w4(rng: np.random.Generator, size: int) -> np.ndarray:
        out = np.empty(size, dtype=float)
        filled = 0
        while filled < size:
            # 拒绝采样生成截断 t 分布，再做方差标准化。
            proposal = rng.standard_t(df=df, size=max(256, (size - filled) * 3))
            accepted = proposal[np.abs(proposal) <= L]
            if accepted.size == 0:
                continue
            take = min(size - filled, accepted.size)
            out[filled : filled + take] = accepted[:take] / s4
            filled += take
        return out

    def pdf_w4(x: np.ndarray) -> np.ndarray:
        y = x * s4
        base = np.where(np.abs(y) <= L, t.pdf(y, df=df) / z_norm, 0.0)
        return base * s4

    lib["W4"] = NoiseModel(
        "W4",
        sample_w4,
        pdf_w4,
        f"Truncated Student-t(df=3, |x|<={L}) with variance normalization.",
    )

    # W5: Uniform[-sqrt(3), sqrt(3)] already has variance 1.
    a5 = np.sqrt(3.0)

    def sample_w5(rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.uniform(-a5, a5, size=size)

    def pdf_w5(x: np.ndarray) -> np.ndarray:
        return np.where(np.abs(x) <= a5, 1.0 / (2.0 * a5), 0.0)

    lib["W5"] = NoiseModel("W5", sample_w5, pdf_w5, "Uniform[-sqrt(3), sqrt(3)].")

    # W6 base variance = 6.09, scale to variance 1.
    s6 = np.sqrt(6.09)

    def sample_w6(rng: np.random.Generator, size: int) -> np.ndarray:
        comp = rng.integers(0, 3, size=size)
        z = np.empty(size, dtype=float)
        z[comp == 0] = rng.normal(-3.0, 0.3, np.sum(comp == 0))
        z[comp == 1] = rng.normal(0.0, 0.3, np.sum(comp == 1))
        z[comp == 2] = rng.normal(3.0, 0.3, np.sum(comp == 2))
        return z / s6

    def pdf_w6(x: np.ndarray) -> np.ndarray:
        y = x * s6
        base = (
            _normal_pdf(y, -3.0, 0.3)
            + _normal_pdf(y, 0.0, 0.3)
            + _normal_pdf(y, 3.0, 0.3)
        ) / 3.0
        return base * s6

    lib["W6"] = NoiseModel("W6", sample_w6, pdf_w6, "Three-peak Gaussian mixture (variance normalized).")

    return lib

