from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.solvers.drpp_1d_exact_solver import solve_drpp_1d_exact

try:
    from scipy.optimize import minimize
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "This visualizer requires scipy. Please install it with: pip install scipy"
    ) from exc

# 运行指令：
# streamlit run theoremA_1d_visualizer.py


@dataclass(frozen=True)
class RadiusParams:
    beta: float = 0.05
    m: int = 1
    a: float = 1.5
    c1: float = 2.0
    c2: float = 1.0


@dataclass
class SmoothUpperApproxSolution:
    lambda_star: float
    s_star: List[float]
    objective_value: float
    constraint_value: float
    success: bool
    message: str
    iterations: int


def _safe_exp(log_value: float) -> float:
    if log_value > 700.0:
        return float("inf")
    if log_value < -745.0:
        return 0.0
    return math.exp(log_value)


def _logsumexp_1d(values: np.ndarray) -> float:
    vmax = float(np.max(values))
    return vmax + math.log(float(np.sum(np.exp(values - vmax))))


def upper_smooth_density_values(
    x: np.ndarray,
    centers: Sequence[float],
    lam: float,
    s: Sequence[float],
    tau: float,
) -> np.ndarray:
    if lam <= 0.0:
        raise ValueError("lambda must be positive.")
    if tau <= 0.0:
        raise ValueError("tau must be positive.")
    c = np.asarray(centers, dtype=float)
    s_arr = np.asarray(s, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    if s_arr.shape[0] != c.shape[0]:
        raise ValueError("s must have the same length as centers.")
    u = tau * (s_arr[:, None] - lam * np.abs(x_arr[None, :] - c[:, None]))
    umax = np.max(u, axis=0)
    lse = umax + np.log(np.sum(np.exp(u - umax), axis=0))
    m = (lse - math.log(len(c))) / tau
    return np.exp(np.clip(m, -745.0, 700.0))


def evaluate_upper_smooth_constraint(
    centers: Sequence[float],
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
    s_arr = np.asarray(s, dtype=float)
    if s_arr.shape[0] != c.shape[0]:
        raise ValueError("s must have the same length as centers.")

    cmin = float(np.min(c))
    cmax = float(np.max(c))
    n = len(c)
    log_n = math.log(n)

    left_vals = tau * (s_arr - lam * c)
    right_vals = tau * (s_arr + lam * c)
    log_k_left = (_logsumexp_1d(left_vals) - log_n) / tau
    log_k_right = (_logsumexp_1d(right_vals) - log_n) / tau

    left_tail = _safe_exp(log_k_left + lam * cmin - math.log(lam))
    right_tail = _safe_exp(log_k_right - lam * cmax - math.log(lam))

    interior = 0.0
    if cmax > cmin:
        m = max(int(interior_grid_points), 401)
        x_inner = np.linspace(cmin, cmax, m)
        y_inner = upper_smooth_density_values(x=x_inner, centers=c, lam=lam, s=s_arr, tau=tau)
        interior = float(np.trapz(y_inner, x_inner))

    return left_tail + interior + right_tail


def solve_upper_smooth_1d(
    centers: Sequence[float],
    epsilon: float,
    tau: float,
    initial_lambda: float,
    initial_s: Sequence[float],
    interior_grid_points: int = 1201,
    maxiter: int = 400,
    tol: float = 1e-8,
) -> SmoothUpperApproxSolution:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if tau <= 0.0:
        raise ValueError("tau must be positive.")
    if len(centers) == 0:
        raise ValueError("centers must be non-empty.")
    n = len(centers)
    if len(initial_s) != n:
        raise ValueError("initial_s must have the same length as centers.")

    lam0 = max(float(initial_lambda), 1e-6)
    x0 = np.array([lam0] + [float(v) for v in initial_s], dtype=float)
    c = [float(v) for v in centers]

    def objective(x: np.ndarray) -> float:
        lam = max(float(x[0]), 1e-8)
        s_vals = x[1:]
        return lam * epsilon - float(np.mean(s_vals))

    def objective_grad(_: np.ndarray) -> np.ndarray:
        return np.array([epsilon] + [-(1.0 / n)] * n, dtype=float)

    def constraint_fun(x: np.ndarray) -> float:
        lam = max(float(x[0]), 1e-8)
        s_vals = x[1:]
        g_val = evaluate_upper_smooth_constraint(
            centers=c,
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
    g_star = evaluate_upper_smooth_constraint(
        centers=c,
        lam=lam_star,
        s=s_star,
        tau=tau,
        interior_grid_points=interior_grid_points,
    )
    j_star = -lam_star * epsilon + sum(s_star) / n

    return SmoothUpperApproxSolution(
        lambda_star=lam_star,
        s_star=s_star,
        objective_value=j_star,
        constraint_value=g_star,
        success=bool(opt.success),
        message=str(opt.message),
        iterations=int(getattr(opt, "nit", -1)),
    )


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


def build_plot(sol, centers: List[float]) -> plt.Figure:
    lam = max(sol.lambda_star, 1e-8)
    cmin, cmax = min(centers), max(centers)
    pad = max(2.0, 6.0 / lam)
    x = np.linspace(cmin - pad, cmax + pad, 1000)
    y = np.array([sol.pdf(float(xx)) for xx in x])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, linewidth=2.0, label=r"$p^\star(x)$")
    ax.scatter(centers, np.zeros_like(centers), marker="x", s=80, label="samples")

    # Also show each component kernel: exp(s_i - lambda|x-c_i|)
    for i, (si, ci) in enumerate(zip(sol.s_star, centers), start=1):
        yi = np.exp(si - lam * np.abs(x - ci))
        ax.plot(x, yi, linestyle="--", alpha=0.45, linewidth=1.0, label=f"kernel {i}")

    ax.set_title("1D optimal predictive density from Theorem A / Theorem 1")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    return fig


def lse_relaxed_params_1d_tent(epsilon: float, n: int) -> tuple[float, float]:
    """
    定理1在 d=1 的加性 LSE 松弛闭式解：
      lambda_LSE = 1 / epsilon
      s_i^LSE = -log(2 N epsilon), i=1..N
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if n <= 0:
        raise ValueError("n must be positive.")
    lam = 1.0 / epsilon
    s = -math.log(2.0 * n * epsilon)
    return lam, s


def build_lse_relaxed_plot(centers: List[float], epsilon: float) -> plt.Figure:
    lam, s = lse_relaxed_params_1d_tent(epsilon=epsilon, n=len(centers))
    cmin, cmax = min(centers), max(centers)
    pad = max(2.0, 6.0 / lam)
    x = np.linspace(cmin - pad, cmax + pad, 1000)

    # q_LSE(x) = sum_i exp(s - lambda|x-c_i|)
    y = np.zeros_like(x)
    for ci in centers:
        y += np.exp(s - lam * np.abs(x - ci))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, linewidth=2.0, color="tab:orange", label=r"$q_{\mathrm{LSE}}(x)$")
    ax.scatter(centers, np.zeros_like(centers), marker="x", s=80, label="samples")

    # show each summand
    for i, ci in enumerate(centers, start=1):
        yi = np.exp(s - lam * np.abs(x - ci))
        ax.plot(x, yi, linestyle="--", alpha=0.45, linewidth=1.0, label=f"term {i}")

    ax.set_title(r"LSE-relaxed density: $q_{\mathrm{LSE}}(x)=\sum_i e^{s^{\mathrm{LSE}}-\lambda^{\mathrm{LSE}}|x-c_i|}$")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    return fig


def build_upper_smooth_plot(
    centers: List[float],
    sol: SmoothUpperApproxSolution,
    tau: float,
) -> plt.Figure:
    lam = max(sol.lambda_star, 1e-8)
    cmin, cmax = min(centers), max(centers)
    pad = max(2.0, 6.0 / lam)
    x = np.linspace(cmin - pad, cmax + pad, 1200)
    y = upper_smooth_density_values(x=x, centers=centers, lam=lam, s=sol.s_star, tau=tau)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, linewidth=2.0, color="tab:green", label=r"$q^{-}_{\tau}(x)$")
    ax.scatter(centers, np.zeros_like(centers), marker="x", s=80, label="samples")

    ax.set_title(r"Upper-preserving smooth approximation: $q^{-}_{\tau}(x)$")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    return fig


def main() -> None:
    st.set_page_config(page_title="Theorem A 1D Visualizer", layout="wide")
    st.title("定理A（一维）交互可视化：最优预测密度 $p^*(x)$")
    st.caption("设定：一维、名义映射为恒等映射、样本输入值=输出值、beta=0.05。")

    with st.sidebar:
        st.header("参数设置")
        n = st.slider("样本个数 N", min_value=1, max_value=10, value=3, step=1)
        slider_min = st.number_input("样本滑动条最小值", value=-5.0, step=0.5)
        slider_max = st.number_input("样本滑动条最大值", value=5.0, step=0.5)
        if slider_max <= slider_min:
            st.error("滑动条上界必须大于下界。")
            return

        samples: List[float] = []
        for i in range(n):
            v = st.slider(
                f"样本值 x[{i+1}] (= y[{i+1}])",
                min_value=float(slider_min),
                max_value=float(slider_max),
                value=float(-1.0 + i),
                step=0.01,
                key=f"sample_{i}",
            )
            samples.append(float(v))

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
        tau = st.slider("保上界平滑近似参数 tau", min_value=1.0, max_value=80.0, value=20.0, step=1.0)

    params = RadiusParams(beta=beta, m=1, a=float(a), c1=float(c1), c2=float(c2))

    try:
        eps = epsilon_n_theorem34(n=n, p=params)
        # In this app we use identity nominal mapping and x_in = x_out,
        # so anchor values are exactly the chosen sample values.
        sol = solve_drpp_1d_exact(centers=samples, epsilon=eps, radii=None)
    except Exception as exc:
        st.error(f"求解失败：{exc}")
        return
    try:
        smooth_sol = solve_upper_smooth_1d(
            centers=samples,
            epsilon=eps,
            tau=float(tau),
            initial_lambda=sol.lambda_star,
            initial_s=sol.s_star,
        )
    except Exception as exc:
        st.error(f"保上界平滑近似求解失败：{exc}")
        return

    c1_col, c2_col = st.columns([1, 2])
    with c1_col:
        st.subheader("数值结果")
        st.write(f"- epsilon_N(beta): **{eps:.8f}**")
        st.write(f"- lambda*: **{sol.lambda_star:.8f}**")
        st.write(f"- objective J*: **{sol.objective_value:.8f}**")
        st.write(f"- G(lambda*, s*): **{sol.constraint_value:.8f}**")
        st.write(f"- success: **{sol.success}**")
        st.write(f"- iterations: **{sol.iterations}**")
        st.write("- s*:")
        for i, si in enumerate(sol.s_star, start=1):
            st.write(f"  - s[{i}] = {si:.8f}")

        lam_lse, s_lse = lse_relaxed_params_1d_tent(epsilon=eps, n=n)
        j_lse = -lam_lse * eps + s_lse
        st.markdown("---")
        st.write("**LSE松弛闭式参数（d=1, tent）**")
        st.write(f"- lambda_LSE: **{lam_lse:.8f}**")
        st.write(f"- s_i^LSE (all i): **{s_lse:.8f}**")
        st.write(f"- J_LSE: **{j_lse:.8f}**")
        st.markdown("---")
        st.write("**保上界平滑近似（m^-_tau）**")
        st.write(f"- tau: **{tau:.2f}**")
        st.write(f"- lambda_tau: **{smooth_sol.lambda_star:.8f}**")
        st.write(f"- J_tau: **{smooth_sol.objective_value:.8f}**")
        st.write(f"- G_tau(lambda,s): **{smooth_sol.constraint_value:.8f}**")
        st.write(f"- success: **{smooth_sol.success}**")
        st.write(f"- iterations: **{smooth_sol.iterations}**")
        st.write(f"- theoretical gap bound log(N)/tau: **{math.log(n) / float(tau):.8f}**")
        st.write("- s_tau*:")
        for i, si in enumerate(smooth_sol.s_star, start=1):
            st.write(f"  - s_tau[{i}] = {si:.8f}")

    with c2_col:
        st.subheader("概率密度曲线")
        fig = build_plot(sol=sol, centers=samples)
        st.pyplot(fig, use_container_width=True)

        st.subheader("LSE松弛分布曲线（和式分布）")
        st.caption(r"$q_{\mathrm{LSE}}(x)=\sum_i \exp(s_i^{\mathrm{LSE}}-\lambda^{\mathrm{LSE}}|x-c_i|)$")
        fig_lse = build_lse_relaxed_plot(centers=samples, epsilon=eps)
        st.pyplot(fig_lse, use_container_width=True)

        st.subheader("保上界平滑近似分布曲线")
        st.caption(
            r"$q^{-}_{\tau}(x)=\exp\!\left(\frac{1}{\tau}\log\sum_i e^{\tau(s_i-\lambda|x-c_i|)}-\frac{\log N}{\tau}\right)$"
        )
        fig_upper = build_upper_smooth_plot(centers=samples, sol=smooth_sol, tau=float(tau))
        st.pyplot(fig_upper, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "运行方式：`streamlit run theoremA_1d_visualizer.py`  \n"
        "依赖：`pip install streamlit numpy scipy matplotlib`"
    )


if __name__ == "__main__":
    main()

