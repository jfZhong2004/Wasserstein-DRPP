from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "This solver requires scipy. Please install it with: pip install scipy"
    ) from exc


EPS = 1e-12
LOG_MIN_LAM = -14.0
LOG_MAX_LAM = 14.0


def _safe_exp(v: float) -> float:
    if v > 700.0:
        return float("inf")
    if v < -745.0:
        return 0.0
    return math.exp(v)


def _normalize_centers(centers: Sequence[float] | Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(centers, dtype=float)
    if arr.ndim == 1:
        if arr.size == 0:
            raise ValueError("centers must be non-empty.")
        return arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError("centers must be a non-empty 1D/2D array-like.")
    return arr


def _normalize_radii(radii: Optional[Sequence[float]], n: int) -> np.ndarray:
    if radii is None:
        out = np.zeros(n, dtype=float)
    else:
        out = np.asarray(radii, dtype=float)
        if out.shape != (n,):
            raise ValueError("radii must have the same length as centers.")
    if np.any(out < 0.0):
        raise ValueError("radii must be non-negative.")
    return out


def _sphere_area(d: int) -> float:
    return 2.0 * math.pi ** (d / 2.0) / math.gamma(d / 2.0)


def _unit_ball_volume(d: int) -> float:
    return math.pi ** (d / 2.0) / math.gamma(d / 2.0 + 1.0)


def _laplace_integral_closed_form(lam: float, d: int) -> float:
    c_d = 2.0 * math.pi ** (d / 2.0) * math.gamma(d) / math.gamma(d / 2.0)
    return c_d / (lam**d)


def _hinge_integral_closed_form(lam: float, r: float, d: int) -> float:
    s_d_minus_1 = _sphere_area(d)
    v_d = _unit_ball_volume(d)
    out = v_d * (r**d)
    poly = 0.0
    for k in range(d):
        coeff = math.comb(d - 1, k) * math.factorial(k) * (r ** (d - 1 - k))
        poly += coeff / (lam ** (k + 1))
    out += s_d_minus_1 * poly
    return out


def _precompute_mc_proposal(
    d: int,
    n_samples: int,
    seed: int,
    scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=0.0, scale=scale, size=(n_samples, d))
    sq_norm = np.sum(x * x, axis=1)
    log_q = -0.5 * d * math.log(2.0 * math.pi * scale * scale) - sq_norm / (2.0 * scale * scale)
    norm = np.sqrt(sq_norm)
    return norm, log_q


def _integrals_mc(
    lam: float,
    radii: np.ndarray,
    norm_samples: np.ndarray,
    log_q_samples: np.ndarray,
) -> np.ndarray:
    if lam <= 0.0:
        raise ValueError("lambda must be positive.")
    out = np.zeros_like(radii, dtype=float)
    for idx, r in enumerate(radii):
        phi = np.maximum(norm_samples - r, 0.0)
        log_w = -lam * phi - log_q_samples
        out[idx] = _safe_exp(float(logsumexp(log_w) - math.log(norm_samples.size)))
    return out


@dataclass
class DRPPLSESolution:
    lambda_star: float
    s_star: np.ndarray
    centers: np.ndarray
    radii: np.ndarray
    epsilon: float
    dimension: int
    model_type: str  # "tent" or "flat_top"
    integration_method: str  # "closed_form" or "mc"
    objective_value: float
    constraint_value: float
    success: bool
    message: str
    iterations: int

    def _point(self, x: float | Sequence[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if self.dimension == 1:
            if arr.ndim == 0:
                return arr.reshape(1)
            if arr.ndim == 1 and arr.size == 1:
                return arr
            raise ValueError("For dimension=1, x must be a scalar.")
        if arr.ndim != 1 or arr.size != self.dimension:
            raise ValueError(f"x must be a vector of length {self.dimension}.")
        return arr

    def log_pdf(self, x: float | Sequence[float] | np.ndarray) -> float:
        point = self._point(x)
        diff = self.centers - point[None, :]
        dist = np.linalg.norm(diff, axis=1)
        if self.model_type == "tent":
            phi = dist
        else:
            phi = np.maximum(dist - self.radii, 0.0)
        vals = self.s_star - self.lambda_star * phi
        return float(np.max(vals))

    def pdf(self, x: float | Sequence[float] | np.ndarray) -> float:
        return _safe_exp(self.log_pdf(x))

    def to_dict(self) -> Dict[str, object]:
        return {
            "lambda_star": self.lambda_star,
            "s_star": self.s_star.tolist(),
            "centers": self.centers.tolist(),
            "radii": self.radii.tolist(),
            "epsilon": self.epsilon,
            "dimension": self.dimension,
            "model_type": self.model_type,
            "integration_method": self.integration_method,
            "objective_value": self.objective_value,
            "constraint_value": self.constraint_value,
            "success": self.success,
            "message": self.message,
            "iterations": self.iterations,
        }


def solve_drpp_lse(
    centers: Sequence[float] | Sequence[Sequence[float]],
    epsilon: float,
    radii: Optional[Sequence[float]] = None,
    integration_method: str = "closed_form",
    mc_samples: int = 2000,
    mc_seed: int = 20260501,
    maxiter: int = 300,
    tol: float = 1e-8,
) -> DRPPLSESolution:
    """
    Solve additive-LSE relaxed DRPP:

      max_{lambda>=0,s}  -lambda*epsilon + mean(s_i)
      s.t.               barG(lambda,s) <= 1
      barG = integral sum_i exp(s_i - lambda*phi_i(x)) dx

    Supports d>=1 inputs (centers can be scalars or vectors).
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    centers_arr = _normalize_centers(centers)
    n, d = centers_arr.shape
    radii_arr = _normalize_radii(radii, n)
    model_type = "flat_top" if np.any(radii_arr > 0.0) else "tent"

    method = integration_method.strip().lower()
    if method not in {"closed_form", "mc"}:
        raise ValueError("integration_method must be 'closed_form' or 'mc'.")
    if method == "mc" and mc_samples < 200:
        raise ValueError("mc_samples must be at least 200 for stability.")

    norm_samples: Optional[np.ndarray] = None
    log_q_samples: Optional[np.ndarray] = None
    if method == "mc":
        scale = max(1.0, d / max(epsilon, 1e-6), float(np.max(radii_arr)) + 1.0)
        norm_samples, log_q_samples = _precompute_mc_proposal(
            d=d,
            n_samples=int(mc_samples),
            seed=int(mc_seed),
            scale=scale,
        )

    def compute_integrals(lam: float) -> np.ndarray:
        if lam <= 0.0:
            raise ValueError("lambda must be positive.")
        if method == "closed_form":
            if model_type == "tent":
                val = _laplace_integral_closed_form(lam=lam, d=d)
                return np.full(n, val, dtype=float)
            return np.array(
                [_hinge_integral_closed_form(lam=lam, r=float(r), d=d) for r in radii_arr],
                dtype=float,
            )
        return _integrals_mc(
            lam=lam,
            radii=radii_arr,
            norm_samples=norm_samples if norm_samples is not None else np.zeros(1),
            log_q_samples=log_q_samples if log_q_samples is not None else np.zeros(1),
        )

    def objective_of_log_lambda(log_lam: float) -> float:
        lam = math.exp(float(log_lam))
        integrals = compute_integrals(lam)
        if np.any(~np.isfinite(integrals)) or np.any(integrals <= EPS):
            return float("inf")
        # minimize -J = lambda*epsilon + mean(log(N*I_i(lambda)))
        return lam * epsilon + float(np.mean(np.log(n * integrals)))

    opt = minimize_scalar(
        objective_of_log_lambda,
        bounds=(LOG_MIN_LAM, LOG_MAX_LAM),
        method="bounded",
        options={"xatol": tol, "maxiter": int(maxiter)},
    )

    log_lam_star = float(opt.x)
    lam_star = math.exp(log_lam_star)
    i_star = compute_integrals(lam_star)
    if np.any(i_star <= EPS):
        raise RuntimeError("LSE solver produced non-positive integral values.")

    s_star = -np.log(n * i_star)
    constraint_value = float(np.sum(np.exp(s_star) * i_star))
    objective_value = -lam_star * epsilon + float(np.mean(s_star))

    return DRPPLSESolution(
        lambda_star=lam_star,
        s_star=s_star,
        centers=centers_arr,
        radii=radii_arr,
        epsilon=float(epsilon),
        dimension=int(d),
        model_type=model_type,
        integration_method=method,
        objective_value=float(objective_value),
        constraint_value=constraint_value,
        success=bool(opt.success),
        message=str(opt.message),
        iterations=int(getattr(opt, "nit", -1)),
    )

