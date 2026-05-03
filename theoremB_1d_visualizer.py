from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.solvers.drpp_1d_exact_solver import solve_drpp_1d_exact

try:
    from scipy.optimize import minimize
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "This visualizer requires scipy. Please install it with: pip install scipy"
    ) from exc

# 运行指令：
# streamlit run theoremB_1d_visualizer.py


@dataclass(frozen=True)
class RadiusParams:
    beta: float = 0.05
    m: int = 1
    a: float = 1.5
    c1: float = 2.0
    c2: float = 1.0


@dataclass
class SmoothLowerApproxSolution:
    lambda_star: float
    s_star: List[float]
    objective_value: float
    constraint_value: float
    success: bool
    message: str
    iterations: int


def epsilon_n_theorem34(n: int, p: RadiusParams) -> float:
    if n <= 0:
        raise ValueError("n must be positive.")
    if not (0.0 < p.beta < 1.0):
        raise ValueError("beta must be in (0,1).")
    if p.a <= 1.0:
        raise ValueError("a must be > 1.")
    if p.c1 <= p.beta:
        raise ValueError("c1 must be greater than beta so log(c1/beta)>0.")
    if p.c2 <= 0.0:
        raise ValueError("c2 must be positive.")

    log_term = math.log(p.c1 / p.beta)
    threshold = log_term / p.c2
    t = log_term / (p.c2 * n)
    exponent = 1.0 / max(p.m, 2) if n >= threshold else 1.0 / p.a
    return t**exponent


def phi_flat_top_1d(x: np.ndarray, c: float, r: float) -> np.ndarray:
    return np.maximum(0.0, np.abs(x - c) - r)


def gamma0_value(x: float, u: float) -> float:
    """
    与项目主实验一致的 gamma0(z), z=(x,u):
      gamma0(z) = min(0.3 * ||z||_2, 5), ||z||_2 = sqrt(x^2+u^2)
    """
    z_norm = math.sqrt(x * x + u * u)
    return min(0.3 * z_norm, 5.0)


EXP_UPPER = 700.0
EXP_LOWER = -745.0


def exp_clipped(v: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(v, EXP_LOWER, EXP_UPPER))


def _safe_exp_scalar(v: float) -> float:
    if v > EXP_UPPER:
        return float("inf")
    if v < EXP_LOWER:
        return 0.0
    return math.exp(v)


def _logsumexp_1d(v: np.ndarray) -> float:
    vmax = float(np.max(v))
    return vmax + math.log(float(np.sum(np.exp(v - vmax))))


def lower_smooth_density_values(
    x: np.ndarray,
    centers: Sequence[float],
    radii: Sequence[float],
    lam: float,
    s: Sequence[float],
    tau: float,
) -> np.ndarray:
    if lam <= 0.0:
        raise ValueError("lambda must be positive.")
    if tau <= 0.0:
        raise ValueError("tau must be positive.")
    c = np.asarray(centers, dtype=float)
    r = np.asarray(radii, dtype=float)
    s_arr = np.asarray(s, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    if not (len(c) == len(r) == len(s_arr)):
        raise ValueError("centers/radii/s must have same length.")
    if np.any(r < 0.0):
        raise ValueError("radii must be non-negative.")

    u = np.empty((len(c), x_arr.size), dtype=float)
    for i, (ci, ri, si) in enumerate(zip(c, r, s_arr)):
        u[i, :] = tau * (si - lam * phi_flat_top_1d(x_arr, ci, ri))
    umax = np.max(u, axis=0)
    lse = umax + np.log(np.sum(np.exp(u - umax), axis=0))
    m_plus = lse / tau
    return np.exp(np.clip(m_plus, EXP_LOWER, EXP_UPPER))


def evaluate_lower_smooth_constraint(
    centers: Sequence[float],
    radii: Sequence[float],
    lam: float,
    s: Sequence[float],
    tau: float,
    interior_grid_points: int = 1201,
) -> float:
    if lam <= 0.0:
        raise ValueError("lambda must be positive.")
    if tau <= 0.0:
        raise ValueError("tau must be positive.")
    c = np.asarray(centers, dtype=float)
    r = np.asarray(radii, dtype=float)
    s_arr = np.asarray(s, dtype=float)
    if not (len(c) == len(r) == len(s_arr)):
        raise ValueError("centers/radii/s must have same length.")

    x_left = float(np.min(c - r))
    x_right = float(np.max(c + r))

    left_vals = tau * (s_arr - lam * (c - r))
    right_vals = tau * (s_arr + lam * (c + r))
    log_k_left = _logsumexp_1d(left_vals) / tau
    log_k_right = _logsumexp_1d(right_vals) / tau

    left_tail = _safe_exp_scalar(log_k_left + lam * x_left - math.log(lam))
    right_tail = _safe_exp_scalar(log_k_right - lam * x_right - math.log(lam))

    interior = 0.0
    if x_right > x_left:
        m = max(int(interior_grid_points), 401)
        x_inner = np.linspace(x_left, x_right, m)
        y_inner = lower_smooth_density_values(
            x=x_inner,
            centers=c,
            radii=r,
            lam=lam,
            s=s_arr,
            tau=tau,
        )
        interior = float(np.trapz(y_inner, x_inner))

    return left_tail + interior + right_tail


def solve_lower_smooth_1d(
    centers: Sequence[float],
    radii: Sequence[float],
    epsilon: float,
    tau: float,
    initial_lambda: float,
    initial_s: Sequence[float],
    interior_grid_points: int = 1201,
    maxiter: int = 400,
    tol: float = 1e-8,
) -> SmoothLowerApproxSolution:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if tau <= 0.0:
        raise ValueError("tau must be positive.")
    if len(centers) == 0:
        raise ValueError("centers must be non-empty.")
    if not (len(centers) == len(radii) == len(initial_s)):
        raise ValueError("centers/radii/initial_s must have same length.")

    n = len(centers)
    x0 = np.array([max(float(initial_lambda), 1e-6)] + [float(v) for v in initial_s], dtype=float)
    c = [float(v) for v in centers]
    r = [float(v) for v in radii]

    def objective(x: np.ndarray) -> float:
        lam = max(float(x[0]), 1e-8)
        s_vals = x[1:]
        return lam * epsilon - float(np.mean(s_vals))

    def objective_grad(_: np.ndarray) -> np.ndarray:
        return np.array([epsilon] + [-(1.0 / n)] * n, dtype=float)

    def constraint_fun(x: np.ndarray) -> float:
        lam = max(float(x[0]), 1e-8)
        s_vals = x[1:]
        g_val = evaluate_lower_smooth_constraint(
            centers=c,
            radii=r,
            lam=lam,
            s=s_vals,
            tau=tau,
            interior_grid_points=interior_grid_points,
        )
        return 1.0 - g_val

    opt = minimize(
        fun=objective,
        x0=x0,
        method="SLSQP",
        jac=objective_grad,
        bounds=[(1e-8, None)] + [(None, None)] * n,
        constraints=[{"type": "ineq", "fun": constraint_fun}],
        options={"maxiter": maxiter, "ftol": tol, "disp": False},
    )

    lam_star = max(float(opt.x[0]), 1e-8)
    s_star = [float(v) for v in opt.x[1:]]
    g_star = evaluate_lower_smooth_constraint(
        centers=c,
        radii=r,
        lam=lam_star,
        s=s_star,
        tau=tau,
        interior_grid_points=interior_grid_points,
    )
    j_star = -lam_star * epsilon + sum(s_star) / n

    return SmoothLowerApproxSolution(
        lambda_star=lam_star,
        s_star=s_star,
        objective_value=j_star,
        constraint_value=g_star,
        success=bool(opt.success),
        message=str(opt.message),
        iterations=int(getattr(opt, "nit", -1)),
    )


def lse_relaxed_params_1d_flat_top(epsilon: float, radii: List[float]) -> Tuple[float, List[float]]:
    """
    定理2在 d=1 的 LSE 加性松弛闭式/KKT形式：
      I_i(lambda) = 2 R_i + 2 / lambda
      s_i(lambda) = -log(N * I_i(lambda))
      epsilon = (1/N) * sum_i [ 1 / (lambda * (1 + lambda R_i)) ]
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if len(radii) == 0:
        raise ValueError("radii must be non-empty.")
    if any(r < 0.0 for r in radii):
        raise ValueError("radii must be non-negative.")

    n = len(radii)

    def g(lam: float) -> float:
        return sum(1.0 / (lam * (1.0 + lam * r)) for r in radii) / n

    lo = 1e-10
    hi = max(1.0, 1.0 / max(epsilon, 1e-8))
    while g(hi) > epsilon:
        hi *= 2.0
        if hi > 1e8:
            break

    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if g(mid) > epsilon:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)

    s = [-math.log(n * (2.0 * r + 2.0 / lam)) for r in radii]
    return lam, s


def build_exact_plot(sol, centers: List[float], radii: List[float]) -> plt.Figure:
    lam = max(sol.lambda_star, 1e-8)
    cmin = min(c - r for c, r in zip(centers, radii))
    cmax = max(c + r for c, r in zip(centers, radii))
    pad = max(2.0, 6.0 / lam)
    x = np.linspace(cmin - pad, cmax + pad, 1200)
    log_y = np.array([sol.log_pdf(float(xx)) for xx in x])
    y = exp_clipped(log_y)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, linewidth=2.0, label=r"$p^\star(x)$")
    ax.scatter(centers, np.zeros_like(centers), marker="x", s=80, label="samples")

    for i, (si, ci, ri) in enumerate(zip(sol.s_star, centers, radii), start=1):
        yi = exp_clipped(si - lam * phi_flat_top_1d(x, ci, ri))
        ax.plot(x, yi, linestyle="--", alpha=0.45, linewidth=1.0, label=f"kernel {i}")

    ax.set_title("Theorem B exact density (flat-top kernels)")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    return fig


def build_lse_plot(centers: List[float], radii: List[float], epsilon: float) -> Tuple[plt.Figure, float, List[float], float]:
    lam_lse, s_lse = lse_relaxed_params_1d_flat_top(epsilon=epsilon, radii=radii)
    cmin = min(c - r for c, r in zip(centers, radii))
    cmax = max(c + r for c, r in zip(centers, radii))
    pad = max(2.0, 6.0 / lam_lse)
    x = np.linspace(cmin - pad, cmax + pad, 1200)

    y = np.zeros_like(x)
    for si, ci, ri in zip(s_lse, centers, radii):
        y += exp_clipped(si - lam_lse * phi_flat_top_1d(x, ci, ri))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, linewidth=2.0, color="tab:orange", label=r"$q_{\mathrm{LSE}}(x)$")
    ax.scatter(centers, np.zeros_like(centers), marker="x", s=80, label="samples")
    for i, (si, ci, ri) in enumerate(zip(s_lse, centers, radii), start=1):
        yi = exp_clipped(si - lam_lse * phi_flat_top_1d(x, ci, ri))
        ax.plot(x, yi, linestyle="--", alpha=0.45, linewidth=1.0, label=f"term {i}")

    ax.set_title(r"Theorem B LSE-relaxed density: $q_{\mathrm{LSE}}(x)=\sum_i e^{s_i^{\mathrm{LSE}}-\lambda^{\mathrm{LSE}}[|x-c_i|-R_i]_+}$")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)

    j_lse = -lam_lse * epsilon + sum(s_lse) / len(s_lse)
    return fig, lam_lse, s_lse, j_lse


def build_tau_smooth_lower_plot(
    centers: List[float],
    radii: List[float],
    sol: SmoothLowerApproxSolution,
    tau: float,
) -> plt.Figure:
    lam = max(sol.lambda_star, 1e-8)
    cmin = min(c - r for c, r in zip(centers, radii))
    cmax = max(c + r for c, r in zip(centers, radii))
    pad = max(2.0, 6.0 / lam)
    x = np.linspace(cmin - pad, cmax + pad, 1200)
    y = lower_smooth_density_values(
        x=x,
        centers=centers,
        radii=radii,
        lam=lam,
        s=sol.s_star,
        tau=tau,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, linewidth=2.0, color="tab:green", label=r"$q_{\tau,+}(x)$")
    ax.scatter(centers, np.zeros_like(centers), marker="x", s=80, label="samples")
    ax.set_title(r"Theorem B tau-smooth lower-bound approximation: $q_{\tau,+}(x)$")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    return fig


def main() -> None:
    st.set_page_config(page_title="Theorem B 1D Visualizer", layout="wide")
    st.title("定理B（一维）交互可视化：平顶核最优预测密度")
    st.caption("设定：一维、名义映射为恒等映射、样本输入值=输出值、beta=0.05。")

    with st.sidebar:
        st.header("参数设置")
        n = st.slider("样本个数 N", min_value=1, max_value=10, value=3, step=1)
        slider_min = st.number_input("样本滑动条最小值", value=-5.0, step=0.5)
        slider_max = st.number_input("样本滑动条最大值", value=5.0, step=0.5)
        if slider_max <= slider_min:
            st.error("滑动条上界必须大于下界。")
            return

        st.markdown("**当前点 $z=(x_k,u_k)$（用于计算 $R_i$）**")
        x_current = st.slider("当前状态 x_k", min_value=float(slider_min), max_value=float(slider_max), value=0.0, step=0.01)
        u_current = st.slider("当前控制 u_k", min_value=-4.0, max_value=4.0, value=0.0, step=0.01)
        st.caption("样本点采用 $\\hat z_{k,i}=(\\hat x_i,0)$（即样本控制默认 0）。")

        centers: List[float] = []
        for i in range(n):
            ci = st.slider(
                f"样本值 x[{i+1}] (= y[{i+1}])",
                min_value=float(slider_min),
                max_value=float(slider_max),
                value=float(-1.0 + i),
                step=0.01,
                key=f"center_{i}",
            )
            centers.append(float(ci))

        st.markdown("---")
        st.markdown("**Theorem 3.4 半径参数**")
        beta = 0.05
        st.write(f"beta (固定) = **{beta}**")
        with st.expander("高级参数 (a, c1, c2, m=1)"):
            a = st.number_input("a (>1)", value=1.5, min_value=1.01, step=0.1)
            c1 = st.number_input("c1 (> beta)", value=2.0, min_value=0.051, step=0.1)
            c2 = st.number_input("c2 (>0)", value=1.0, min_value=1e-6, step=0.1, format="%.6f")
            st.write("m = 1 (固定，一维)")
        st.markdown("---")
        tau = st.slider("下界保持平滑参数 tau", min_value=1.0, max_value=80.0, value=20.0, step=1.0)

    params = RadiusParams(beta=beta, m=1, a=float(a), c1=float(c1), c2=float(c2))

    try:
        eps = epsilon_n_theorem34(n=n, p=params)
        r_current = math.sqrt(max(gamma0_value(float(x_current), float(u_current)), 0.0))
        r_base = [math.sqrt(max(gamma0_value(ci, 0.0), 0.0)) for ci in centers]
        radii = [r_current + rb for rb in r_base]
        sol = solve_drpp_1d_exact(centers=centers, epsilon=eps, radii=radii)
        fig_lse, lam_lse, s_lse, j_lse = build_lse_plot(centers=centers, radii=radii, epsilon=eps)
        tau_sol = solve_lower_smooth_1d(
            centers=centers,
            radii=radii,
            epsilon=eps,
            tau=float(tau),
            initial_lambda=sol.lambda_star,
            initial_s=sol.s_star,
        )
        fig_tau = build_tau_smooth_lower_plot(centers=centers, radii=radii, sol=tau_sol, tau=float(tau))
    except Exception as exc:
        st.error(f"求解失败：{exc}")
        return

    c1_col, c2_col = st.columns([1, 2])
    with c1_col:
        st.subheader("数值结果（精确解）")
        st.write(f"- epsilon_N(beta): **{eps:.8f}**")
        st.write(f"- lambda*: **{sol.lambda_star:.8f}**")
        st.write(f"- objective J*: **{sol.objective_value:.8f}**")
        st.write(f"- G(lambda*, s*): **{sol.constraint_value:.8f}**")
        st.write(f"- success: **{sol.success}**")
        st.write(f"- iterations: **{sol.iterations}**")
        if (not sol.success) or (sol.constraint_value > 1.0 + 1e-6):
            st.warning("精确求解器未完全收敛或约束未严格满足，图像可能出现尖峰（已做数值防溢出裁剪）。")
        st.write(f"- r_current = sqrt(gamma0(z)): **{r_current:.8f}**")
        st.write("- R_i = r_current + sqrt(gamma0(zhat_i)):")
        for i, (rb, ri) in enumerate(zip(r_base, radii), start=1):
            st.write(f"  - i={i}: sqrt(gamma0(zhat_i))={rb:.8f},  R[{i}]={ri:.8f}")
        st.write("- s*:")
        for i, si in enumerate(sol.s_star, start=1):
            st.write(f"  - s[{i}] = {si:.8f}")

        st.markdown("---")
        st.subheader("LSE松弛闭式/KKT参数")
        st.write(f"- lambda_LSE: **{lam_lse:.8f}**")
        st.write(f"- J_LSE: **{j_lse:.8f}**")
        st.write("- s_LSE:")
        for i, si in enumerate(s_lse, start=1):
            st.write(f"  - s_LSE[{i}] = {si:.8f}")
        st.markdown("---")
        st.subheader("tau-平滑下界参数")
        st.write(f"- tau: **{tau:.2f}**")
        st.write(f"- lambda_tau: **{tau_sol.lambda_star:.8f}**")
        st.write(f"- J_tau: **{tau_sol.objective_value:.8f}**")
        st.write(f"- G_tau(lambda,s): **{tau_sol.constraint_value:.8f}**")
        st.write(f"- success: **{tau_sol.success}**")
        st.write(f"- iterations: **{tau_sol.iterations}**")
        st.write(f"- LSE gap scale log(N)/tau: **{math.log(n) / float(tau):.8f}**")
        st.write("- s_tau:")
        for i, si in enumerate(tau_sol.s_star, start=1):
            st.write(f"  - s_tau[{i}] = {si:.8f}")

    with c2_col:
        st.subheader("概率密度曲线（精确解）")
        fig_exact = build_exact_plot(sol=sol, centers=centers, radii=radii)
        st.pyplot(fig_exact, use_container_width=True)

        st.subheader("LSE松弛分布曲线（和式分布）")
        st.caption(r"$q_{\mathrm{LSE}}(x)=\sum_i \exp(s_i^{\mathrm{LSE}}-\lambda^{\mathrm{LSE}}[|x-c_i|-R_i]_+)$")
        st.pyplot(fig_lse, use_container_width=True)

        st.subheader("tau-平滑下界分布曲线")
        st.caption(
            r"$q_{\tau,+}(x)=\exp\!\left(\frac{1}{\tau}\log\sum_i e^{\tau(s_i-\lambda[|x-c_i|-R_i]_+)}\right)$"
        )
        st.pyplot(fig_tau, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "运行方式：`streamlit run theoremB_1d_visualizer.py`  \n"
        "依赖：`pip install streamlit numpy scipy matplotlib`"
    )


if __name__ == "__main__":
    main()

