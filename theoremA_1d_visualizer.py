from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.solvers.drpp_1d_exact_solver import solve_drpp_1d_exact

# 运行指令：
# streamlit run theoremA_1d_visualizer.py


@dataclass(frozen=True)
class RadiusParams:
    beta: float = 0.05
    m: int = 1
    a: float = 1.5
    c1: float = 2.0
    c2: float = 1.0


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


def main() -> None:
    st.set_page_config(page_title="Theorem A 1D Visualizer", layout="wide")
    st.title("定理A（一维）交互可视化：最优预测密度 $p^*(x)$")
    st.caption("设定：一维、名义映射为恒等映射、样本输入值=输出值、beta=0.05。")

    with st.sidebar:
        st.header("参数设置")
        n = st.slider("样本个数 N", min_value=1, max_value=5, value=3, step=1)
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

    params = RadiusParams(beta=beta, m=1, a=float(a), c1=float(c1), c2=float(c2))

    try:
        eps = epsilon_n_theorem34(n=n, p=params)
        # In this app we use identity nominal mapping and x_in = x_out,
        # so anchor values are exactly the chosen sample values.
        sol = solve_drpp_1d_exact(centers=samples, epsilon=eps, radii=None)
    except Exception as exc:
        st.error(f"求解失败：{exc}")
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

    with c2_col:
        st.subheader("概率密度曲线")
        fig = build_plot(sol=sol, centers=samples)
        st.pyplot(fig, use_container_width=True)

        st.subheader("LSE松弛分布曲线（和式分布）")
        st.caption(r"$q_{\mathrm{LSE}}(x)=\sum_i \exp(s_i^{\mathrm{LSE}}-\lambda^{\mathrm{LSE}}|x-c_i|)$")
        fig_lse = build_lse_relaxed_plot(centers=samples, epsilon=eps)
        st.pyplot(fig_lse, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "运行方式：`streamlit run theoremA_1d_visualizer.py`  \n"
        "依赖：`pip install streamlit numpy scipy matplotlib`"
    )


if __name__ == "__main__":
    main()

