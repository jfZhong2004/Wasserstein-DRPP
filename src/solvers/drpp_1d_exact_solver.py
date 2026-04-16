from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import math

try:
    from scipy.optimize import minimize
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "This solver requires scipy. Please install it with: pip install scipy"
    ) from exc


EPS = 1e-12
INF = float("inf")


@dataclass(frozen=True)
class AffinePiece:
    """上包络线构造中使用的线性片段。"""

    kernel_idx: int
    slope: float
    intercept: float
    left: float
    right: float


@dataclass(frozen=True)
class EnvelopeSegment:
    """上包络线分段信息：在区间 [left, right] 内由 kernel_idx 支配。"""

    left: float
    right: float
    kernel_idx: int


@dataclass
class DRPP1DSolution:
    """
    一维精确解析算法的求解结果。

    该对象可直接被外部代码调用：
    1) 读取 lambda_star / s_star 作为最优参数；
    2) 调用 pdf(x) 或 log_pdf(x) 评估最优预测密度。
    """

    lambda_star: float
    s_star: List[float]
    centers: List[float]
    radii: List[float]
    epsilon: float
    model_type: str  # "tent" or "flat_top"
    objective_value: float
    constraint_value: float
    success: bool
    message: str
    iterations: int

    def log_pdf(self, x: float) -> float:
        """返回 log p*(x) = max_i [s_i - lambda * phi_i(x)]。"""
        best = -INF
        for c, r, s in zip(self.centers, self.radii, self.s_star):
            phi = _phi_1d(x=x, center=c, radius=r, model_type=self.model_type)
            val = s - self.lambda_star * phi
            if val > best:
                best = val
        return best

    def pdf(self, x: float) -> float:
        """返回最优预测密度 p*(x)。"""
        return math.exp(self.log_pdf(x))

    def to_dict(self) -> Dict[str, object]:
        """便于序列化/日志记录的字典格式。"""
        return {
            "lambda_star": self.lambda_star,
            "s_star": list(self.s_star),
            "centers": list(self.centers),
            "radii": list(self.radii),
            "epsilon": self.epsilon,
            "model_type": self.model_type,
            "objective_value": self.objective_value,
            "constraint_value": self.constraint_value,
            "success": self.success,
            "message": self.message,
            "iterations": self.iterations,
        }


def _phi_1d(x: float, center: float, radius: float, model_type: str) -> float:
    """
    一维核函数中的距离代价 phi_i(x)。

    - 尖顶模型（tent）:      phi = |x-c|
    - 平顶模型（flat_top）:  phi = max(0, |x-c|-R)
    """
    dist = abs(x - center)
    if model_type == "tent":
        return dist
    return max(0.0, dist - radius)


def _build_affine_pieces(
    centers: Sequence[float],
    radii: Sequence[float],
    s: Sequence[float],
    lam: float,
    model_type: str,
) -> List[AffinePiece]:
    """
    将每个 g_i(x) = s_i - lam * phi_i(x) 拆成有限个线性片段。

    这是“精确解析”的关键：在一维上 g_i 是分段线性，max 上包络也必然是分段线性。
    """
    pieces: List[AffinePiece] = []
    for idx, (c, r, si) in enumerate(zip(centers, radii, s)):
        if model_type == "tent":
            # 左支：x <= c, g = +lam*x + (s - lam*c)
            pieces.append(
                AffinePiece(
                    kernel_idx=idx,
                    slope=lam,
                    intercept=si - lam * c,
                    left=-INF,
                    right=c,
                )
            )
            # 右支：x >= c, g = -lam*x + (s + lam*c)
            pieces.append(
                AffinePiece(
                    kernel_idx=idx,
                    slope=-lam,
                    intercept=si + lam * c,
                    left=c,
                    right=INF,
                )
            )
        else:
            # 平顶模型：左斜面 + 平顶 + 右斜面
            left_kink = c - r
            right_kink = c + r
            # 左斜面：x <= c-R, g = +lam*x + (s - lam*(c-R))
            pieces.append(
                AffinePiece(
                    kernel_idx=idx,
                    slope=lam,
                    intercept=si - lam * left_kink,
                    left=-INF,
                    right=left_kink,
                )
            )
            # 平顶：c-R <= x <= c+R, g = s
            pieces.append(
                AffinePiece(
                    kernel_idx=idx,
                    slope=0.0,
                    intercept=si,
                    left=left_kink,
                    right=right_kink,
                )
            )
            # 右斜面：x >= c+R, g = -lam*x + (s + lam*(c+R))
            pieces.append(
                AffinePiece(
                    kernel_idx=idx,
                    slope=-lam,
                    intercept=si + lam * right_kink,
                    left=right_kink,
                    right=INF,
                )
            )
    return pieces


def _is_finite(x: float) -> bool:
    return not math.isinf(x)


def _unique_sorted(values: Sequence[float], tol: float = 1e-10) -> List[float]:
    """按容差去重并排序。"""
    if not values:
        return []
    arr = sorted(values)
    out = [arr[0]]
    for v in arr[1:]:
        if abs(v - out[-1]) > tol:
            out.append(v)
    return out


def _build_upper_envelope_segments(
    centers: Sequence[float],
    radii: Sequence[float],
    s: Sequence[float],
    lam: float,
    model_type: str,
) -> List[EnvelopeSegment]:
    """
    通过“候选断点集合”精确构造上包络线分段。

    断点来源：
    1) 所有核函数自身的拐点（kink）；
    2) 任意两条线性片段在公共定义域内的交点。

    在断点之间，各核都是线性的，且不存在新的交叉，因此支配核在区间内恒定。
    """
    pieces = _build_affine_pieces(centers=centers, radii=radii, s=s, lam=lam, model_type=model_type)

    candidates: List[float] = []
    for p in pieces:
        if _is_finite(p.left):
            candidates.append(p.left)
        if _is_finite(p.right):
            candidates.append(p.right)

    m = len(pieces)
    for i in range(m):
        p = pieces[i]
        for j in range(i + 1, m):
            q = pieces[j]
            # 同核的片段交点要么在自身拐点（已纳入），要么无意义，直接跳过。
            if p.kernel_idx == q.kernel_idx:
                continue
            if abs(p.slope - q.slope) <= 1e-14:
                continue
            left = max(p.left, q.left)
            right = min(p.right, q.right)
            if left >= right:
                continue
            x_cross = (q.intercept - p.intercept) / (p.slope - q.slope)
            if x_cross + 1e-12 < left or x_cross - 1e-12 > right:
                continue
            candidates.append(x_cross)

    points = _unique_sorted(candidates)

    intervals: List[Tuple[float, float]] = []
    if not points:
        intervals.append((-INF, INF))
    else:
        intervals.append((-INF, points[0]))
        for a, b in zip(points[:-1], points[1:]):
            if b - a > 1e-14:
                intervals.append((a, b))
        intervals.append((points[-1], INF))

    segments: List[EnvelopeSegment] = []
    for left, right in intervals:
        probe = _pick_probe_point(left=left, right=right)
        best_idx = _argmax_kernel_at_x(
            x=probe, centers=centers, radii=radii, s=s, lam=lam, model_type=model_type
        )
        if segments and segments[-1].kernel_idx == best_idx and abs(segments[-1].right - left) <= 1e-10:
            # 相邻同核分段合并，减少数值噪声。
            segments[-1] = EnvelopeSegment(left=segments[-1].left, right=right, kernel_idx=best_idx)
        else:
            segments.append(EnvelopeSegment(left=left, right=right, kernel_idx=best_idx))

    return segments


def _pick_probe_point(left: float, right: float) -> float:
    """为区间选择一个内部探测点，用于判断该区间由哪个核支配。"""
    if math.isinf(left) and math.isinf(right):
        return 0.0
    if math.isinf(left):
        return right - 1.0
    if math.isinf(right):
        return left + 1.0
    return 0.5 * (left + right)


def _argmax_kernel_at_x(
    x: float,
    centers: Sequence[float],
    radii: Sequence[float],
    s: Sequence[float],
    lam: float,
    model_type: str,
) -> int:
    """在给定 x 处返回使 g_i(x) 最大的核编号。"""
    best_idx = 0
    best_val = -INF
    for idx, (c, r, si) in enumerate(zip(centers, radii, s)):
        phi = _phi_1d(x=x, center=c, radius=r, model_type=model_type)
        val = si - lam * phi
        if val > best_val:
            best_val = val
            best_idx = idx
    return best_idx


def _primitive_weighted_tail(y: float, lam: float) -> float:
    """
    F(y) = (y/lam + 1/lam^2) * exp(-lam*y), y>=0。

    用于计算 ∫ y exp(-lam y) dy 的定积分。
    """
    if math.isinf(y):
        return 0.0
    return (y / lam + 1.0 / (lam * lam)) * math.exp(-lam * y)


def _exp_of_s(s: float) -> float:
    """安全计算 exp(s)，若溢出则返回 +inf。"""
    try:
        return math.exp(s)
    except OverflowError:
        return INF


def _integral_and_weight_tent(
    a: float,
    b: float,
    s: float,
    lam: float,
    c: float,
) -> Tuple[float, float]:
    """
    计算帐篷核在区间 [a,b] 的两个解析积分：
    1) I = ∫ exp(s - lam|x-c|) dx
    2) W = ∫ |x-c| exp(s - lam|x-c|) dx
    """
    exp_s = _exp_of_s(s)
    if math.isinf(exp_s):
        return INF, INF

    I = 0.0
    W = 0.0

    # 左侧部分 [a, min(b,c)]
    if a < c:
        u = a
        v = min(b, c)
        if u < v:
            eu = 0.0 if math.isinf(u) and u < 0 else math.exp(lam * (u - c))
            ev = math.exp(lam * (v - c))
            I += exp_s * (ev - eu) / lam
            W += exp_s * (
                _primitive_weighted_tail(c - v, lam) - _primitive_weighted_tail(c - u, lam)
            )

    # 右侧部分 [max(a,c), b]
    if b > c:
        u = max(a, c)
        v = b
        if u < v:
            eu = math.exp(-lam * (u - c))
            ev = 0.0 if math.isinf(v) and v > 0 else math.exp(-lam * (v - c))
            I += exp_s * (eu - ev) / lam
            W += exp_s * (
                _primitive_weighted_tail(u - c, lam) - _primitive_weighted_tail(v - c, lam)
            )

    return I, W


def _integral_and_weight_flat_top(
    a: float,
    b: float,
    s: float,
    lam: float,
    c: float,
    r: float,
) -> Tuple[float, float]:
    """
    计算平顶核在区间 [a,b] 的两个解析积分：
    1) I = ∫ exp(s - lam*max(0,|x-c|-r)) dx
    2) W = ∫ max(0,|x-c|-r) * exp(s - lam*max(0,|x-c|-r)) dx
    """
    exp_s = _exp_of_s(s)
    if math.isinf(exp_s):
        return INF, INF

    left_kink = c - r
    right_kink = c + r

    I = 0.0
    W = 0.0

    # 左指数尾
    left_u = a
    left_v = min(b, left_kink)
    if left_u < left_v:
        eu = 0.0 if math.isinf(left_u) and left_u < 0 else math.exp(lam * (left_u - left_kink))
        ev = math.exp(lam * (left_v - left_kink))
        I += exp_s * (ev - eu) / lam
        W += exp_s * (
            _primitive_weighted_tail(left_kink - left_v, lam)
            - _primitive_weighted_tail(left_kink - left_u, lam)
        )

    # 平顶段
    mid_u = max(a, left_kink)
    mid_v = min(b, right_kink)
    if mid_u < mid_v:
        I += exp_s * (mid_v - mid_u)
        # 平顶段 phi=0，因此 W 不增加

    # 右指数尾
    right_u = max(a, right_kink)
    right_v = b
    if right_u < right_v:
        eu = math.exp(-lam * (right_u - right_kink))
        ev = 0.0 if math.isinf(right_v) and right_v > 0 else math.exp(-lam * (right_v - right_kink))
        I += exp_s * (eu - ev) / lam
        W += exp_s * (
            _primitive_weighted_tail(right_u - right_kink, lam)
            - _primitive_weighted_tail(right_v - right_kink, lam)
        )

    return I, W


def _evaluate_constraint_and_gradients(
    lam: float,
    s: Sequence[float],
    centers: Sequence[float],
    radii: Sequence[float],
    model_type: str,
) -> Tuple[float, float, List[float], List[EnvelopeSegment]]:
    """
    解析计算：
    1) G(lam, s) = ∫ max_i exp(s_i - lam*phi_i(x)) dx
    2) dG/dlam
    3) dG/ds_i

    做法：先构造上包络线分段，再逐段闭式积分。
    """
    segments = _build_upper_envelope_segments(
        centers=centers, radii=radii, s=s, lam=lam, model_type=model_type
    )

    G = 0.0
    dG_dlam = 0.0
    dG_ds = [0.0 for _ in s]

    for seg in segments:
        i = seg.kernel_idx
        c = centers[i]
        r = radii[i]
        si = s[i]
        a = seg.left
        b = seg.right

        if model_type == "tent":
            I, W = _integral_and_weight_tent(a=a, b=b, s=si, lam=lam, c=c)
        else:
            I, W = _integral_and_weight_flat_top(a=a, b=b, s=si, lam=lam, c=c, r=r)

        G += I
        dG_ds[i] += I
        # dG/dlam = -∫phi * exp(...) dx
        dG_dlam -= W

    return G, dG_dlam, dG_ds, segments


def _solve_lambda_init_flat_top(epsilon: float, radii: Sequence[float]) -> float:
    """
    平顶模型的 LSE 初值 lambda。

    d=1 时：
      I_i(lambda) = 2R_i + 2/lambda
      -I_i'(lambda) = 2/lambda^2
    KKT 方程给出：
      epsilon = mean_i [ (-I_i') / I_i ] = mean_i [1 / (lambda*(1+lambda*R_i))]
    """

    def f(lam: float) -> float:
        return sum(1.0 / (lam * (1.0 + lam * r)) for r in radii) / len(radii)

    lo = 1e-10
    hi = max(1.0, 1.0 / max(epsilon, 1e-8))
    while f(hi) > epsilon:
        hi *= 2.0
        if hi > 1e8:
            break

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if f(mid) > epsilon:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_drpp_1d_exact(
    centers: Sequence[float],
    epsilon: float,
    radii: Optional[Sequence[float]] = None,
    initial_lambda: Optional[float] = None,
    initial_s: Optional[Sequence[float]] = None,
    maxiter: int = 500,
    tol: float = 1e-9,
) -> DRPP1DSolution:
    """
    一维 Wasserstein 单步 DRPP 的精确解析求解器。

    参数说明：
    - centers: 锚点 c_i（即文档中的 x_i^pred）
    - epsilon: Wasserstein 半径 epsilon > 0
    - radii:
        * None 或全 0 -> 尖顶模型（Theorem 1）
        * 存在正半径 -> 平顶模型（Theorem 2）
    - initial_lambda / initial_s: 可选自定义初值（用于外部 warm-start）
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if len(centers) == 0:
        raise ValueError("centers must be non-empty.")

    centers_list = [float(v) for v in centers]
    n = len(centers_list)

    if radii is None:
        radii_list = [0.0] * n
    else:
        if len(radii) != n:
            raise ValueError("radii must have the same length as centers.")
        radii_list = [float(v) for v in radii]
        if any(r < 0 for r in radii_list):
            raise ValueError("radii must be non-negative.")

    model_type = "flat_top" if any(r > 0 for r in radii_list) else "tent"

    # ---------- 初值构造（来自文档第 4 节的 LSE 解析解） ----------
    if initial_lambda is not None:
        lam0 = max(float(initial_lambda), 1e-8)
    else:
        if model_type == "tent":
            lam0 = 1.0 / epsilon  # d=1 的 LSE 闭式：lambda*=1/epsilon
        else:
            lam0 = _solve_lambda_init_flat_top(epsilon=epsilon, radii=radii_list)

    if initial_s is not None:
        if len(initial_s) != n:
            raise ValueError("initial_s must have the same length as centers.")
        s0 = [float(v) for v in initial_s]
    else:
        if model_type == "tent":
            # LSE 约束: N*exp(s)*(2/lambda)=1 -> s=log(lambda/(2N))
            s_scalar = math.log(lam0 / (2.0 * n))
            s0 = [s_scalar] * n
        else:
            # LSE 约束: exp(s_i)=1/(N*(2R_i+2/lambda))
            s0 = [-math.log(n * (2.0 * r + 2.0 / lam0)) for r in radii_list]

    x0 = [lam0] + s0

    def objective(x: Sequence[float]) -> float:
        lam = max(float(x[0]), 1e-10)
        s = x[1:]
        return lam * epsilon - sum(s) / n  # minimize -J

    def objective_grad(_: Sequence[float]) -> List[float]:
        return [epsilon] + [-(1.0 / n)] * n

    def constraint_fun(x: Sequence[float]) -> float:
        lam = max(float(x[0]), 1e-10)
        s = x[1:]
        G, _, _, _ = _evaluate_constraint_and_gradients(
            lam=lam, s=s, centers=centers_list, radii=radii_list, model_type=model_type
        )
        return 1.0 - G  # SLSQP 的 ineq 约束要求 >=0

    def constraint_jac(x: Sequence[float]) -> List[float]:
        lam = max(float(x[0]), 1e-10)
        s = x[1:]
        _, dG_dlam, dG_ds, _ = _evaluate_constraint_and_gradients(
            lam=lam, s=s, centers=centers_list, radii=radii_list, model_type=model_type
        )
        # c(x)=1-G(x) -> grad c = -grad G
        return [-dG_dlam] + [-v for v in dG_ds]

    bounds = [(1e-8, None)] + [(None, None)] * n
    constraints = [{"type": "ineq", "fun": constraint_fun, "jac": constraint_jac}]

    opt = minimize(
        fun=objective,
        x0=x0,
        method="SLSQP",
        jac=objective_grad,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": maxiter, "ftol": tol, "disp": False},
    )

    lam_star = max(float(opt.x[0]), 1e-10)
    s_star = [float(v) for v in opt.x[1:]]
    G_star, _, _, _ = _evaluate_constraint_and_gradients(
        lam=lam_star, s=s_star, centers=centers_list, radii=radii_list, model_type=model_type
    )
    J_star = -lam_star * epsilon + sum(s_star) / n

    return DRPP1DSolution(
        lambda_star=lam_star,
        s_star=s_star,
        centers=centers_list,
        radii=radii_list,
        epsilon=float(epsilon),
        model_type=model_type,
        objective_value=J_star,
        constraint_value=G_star,
        success=bool(opt.success),
        message=str(opt.message),
        iterations=int(getattr(opt, "nit", -1)),
    )


def _print_solution(title: str, sol: DRPP1DSolution) -> None:
    """示例输出（终端文本严格使用英文）。"""
    print(f"=== {title} ===")
    print(f"model_type: {sol.model_type}")
    print(f"success: {sol.success}")
    print(f"message: {sol.message}")
    print(f"iterations: {sol.iterations}")
    print(f"lambda*: {sol.lambda_star:.10f}")
    print(f"objective J*: {sol.objective_value:.10f}")
    print(f"G(lambda*, s*): {sol.constraint_value:.10f}")
    print("s*: [" + ", ".join(f"{v:.8f}" for v in sol.s_star) + "]")
    print("pdf samples:")
    for x in (-3.0, -1.0, 0.0, 1.0, 3.0):
        print(f"  x={x:+.2f}, p*(x)={sol.pdf(x):.10f}")
    print()


def main() -> None:
    """
    main 中给出两个可运行示例：
    1) 尖顶模型（R_i=0）
    2) 平顶模型（R_i>0）
    """
    # 尖顶示例（Theorem 1）
    centers_tent = [-2.0, -0.4, 1.1, 2.7]
    epsilon_tent = 0.9
    sol_tent = solve_drpp_1d_exact(centers=centers_tent, epsilon=epsilon_tent, radii=None)
    _print_solution("Sharp-Top (Tent) Example", sol_tent)

    # 平顶示例（Theorem 2）
    centers_flat = [-2.2, -0.7, 0.8, 2.4]
    radii_flat = [0.35, 0.55, 0.40, 0.70]
    epsilon_flat = 0.35
    sol_flat = solve_drpp_1d_exact(
        centers=centers_flat,
        epsilon=epsilon_flat,
        radii=radii_flat,
    )
    _print_solution("Flat-Top (Hinge-Laplace) Example", sol_flat)


if __name__ == "__main__":
    main()

