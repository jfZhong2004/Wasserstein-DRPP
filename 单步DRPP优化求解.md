# Wasserstein 单步 DRPP 凸优化问题的求解方案

本文档针对 `note0403.md` 中定理 1 与定理 2 的两个有限维凸优化问题，系统地给出可计算的求解方案。核心难点在于：约束中的积分 $\int_{\mathcal{X}} \max_i \exp(\cdots)\, dx$ 是一个**逐点最大值函数的连续积分**，无法直接输入标准凸优化求解器。我们从 KKT 结构分析出发，依次给出一维精确解析算法、LSE 松弛解析界、以及一般维度的数值优化算法。

---

## 1. 统一问题形式

定理 1 和定理 2 可统一写为如下形式：

$$
\max_{\lambda \geq 0,\; \mathbf{s} \in \mathbb{R}^N} \quad J(\lambda, \mathbf{s}) = -\lambda\varepsilon + \frac{1}{N}\sum_{i=1}^N s_i
$$
$$
\text{s.t.} \quad G(\lambda, \mathbf{s}) := \int_{\mathcal{X}} \max_{1 \leq i \leq N} \exp\!\Big(s_i - \lambda\,\varphi_i(x)\Big)\, dx \leq 1
$$

其中距离核函数 $\varphi_i(x)$ 在两个定理中分别为：

| | 定理 1（上界，$\gamma_0 \equiv 0$） | 定理 2（下界，保守近似） |
|---|---|---|
| $\varphi_i(x)$ | $\|x - \hat{x}_i^{pred}\|$ | $\max\!\big(0,\; \|x - \hat{x}_i^{pred}\| - R_i\big)$ |
| 核形状 | **尖顶指数核**（Laplace 核） | **平顶指数核**（Hinge-Laplace 核） |

其中 $R_i = \sqrt{\gamma_0(z)} + \sqrt{\gamma_0(\hat{z}_{k,i})}$ 是复合认知不确定性半径。注意当 $R_i = 0$ 时定理 2 退化为定理 1。

**约束中的计算难点：**
- 被积函数是 $N$ 个指数核的**逐点最大值**（非解析、非光滑）
- 积分定义在连续状态空间 $\mathcal{X}$ 上（可能无界）
- $G(\lambda, \mathbf{s})$ 关于 $(\lambda, \mathbf{s})$ 是凸的（已在 note0403.md 中证明），但没有闭式表达

---

## 2. KKT 最优性条件与结构性质

在设计算法之前，先分析最优解的结构，这将为算法设计提供关键指导。

### 2.1 约束活跃性

**命题 1.** *在最优解 $(\lambda^*, \mathbf{s}^*)$ 处，积分约束必然取等：$G(\lambda^*, \mathbf{s}^*) = 1$。*

**证明.** 若 $G < 1$，则可对所有 $s_i$ 同时增加 $\delta > 0$（令 $s_i' = s_i + \delta$），此时 $G' = e^\delta G$。取 $\delta = \log(1/G) > 0$，则 $G' = 1$，而目标函数增加 $\delta > 0$，与最优性矛盾。$\square$

### 2.2 等概率质量性质

定义**功率胞腔（Power Cell）** 为最优预测分布中第 $i$ 个核的支配区域：
$$
\Omega_i^* := \left\{x \in \mathcal{X} : s_i^* - \lambda^*\varphi_i(x) \geq s_j^* - \lambda^*\varphi_j(x),\; \forall j \neq i\right\}
$$

在 $\Omega_i^*$ 上，最优预测密度 $\hat{p}^*(x) = \exp\!\big(s_i^* - \lambda^*\varphi_i(x)\big)$。

**定理 3（等质量定理）.** *在最优解处，每个功率胞腔承载相等的概率质量：*
$$
\int_{\Omega_i^*} \hat{p}^*(x)\, dx = \frac{1}{N}, \quad \forall\, i = 1, \dots, N
$$

**证明.** 由于 $G = 1$ 在最优点活跃，引入 KKT 乘子 $\mu \geq 0$。最优性的一阶必要条件（对 $s_i$ 求导）为：
$$
\frac{\partial J}{\partial s_i} = \mu \cdot \frac{\partial G}{\partial s_i}
$$

左边：$\frac{\partial J}{\partial s_i} = \frac{1}{N}$。

右边：由于 $\max_j \exp(A_j) = \exp(\max_j A_j)$，在胞腔 $\Omega_i$ 内部，最大值由第 $i$ 项取到，故：
$$
\frac{\partial G}{\partial s_i} = \int_{\Omega_i} \exp\!\big(s_i - \lambda\varphi_i(x)\big)\, dx = \int_{\Omega_i} \hat{p}^*(x)\, dx
$$
（胞腔边界为零测集，不影响积分。）

因此 KKT 条件给出：
$$
\frac{1}{N} = \mu \int_{\Omega_i} \hat{p}^*(x)\, dx, \quad \forall\, i
$$

对所有 $i$ 求和并利用 $G = 1$：
$$
1 = \mu \sum_{i=1}^N \int_{\Omega_i} \hat{p}^*(x)\, dx = \mu \cdot G = \mu
$$

故 $\mu = 1$，代回得 $\int_{\Omega_i} \hat{p}^*(x)\, dx = \frac{1}{N}$。$\square$

**直觉解释：** 最优预测分布将概率质量**均匀分配**给每个经验样本所对应的功率胞腔。这体现了一种信息论意义上的"公平"——在最坏情况下，预测者不应偏向任何特定样本。

### 2.3 Wasserstein 半径的对偶解释

**命题 2.** *若最优解满足 $\lambda^* > 0$，则：*
$$
\varepsilon = \int_{\mathcal{X}} \hat{p}^*(x) \cdot \varphi_{i^*(x)}(x)\, dx = \mathbb{E}_{\hat{p}^*}\!\left[\varphi_{i^*(x)}(x)\right]
$$
*其中 $i^*(x) = \arg\max_i\!\big(s_i^* - \lambda^*\varphi_i(x)\big)$ 是点 $x$ 所在的功率胞腔标号。*

**证明.** 对 $\lambda$ 的 KKT 条件为 $-\varepsilon = \mu \cdot \frac{\partial G}{\partial \lambda}$。计算偏导：
$$
\frac{\partial G}{\partial \lambda} = -\int_{\mathcal{X}} \varphi_{i^*(x)}(x) \cdot \hat{p}^*(x)\, dx
$$
代入 $\mu = 1$ 即得。$\square$

**直觉解释：** Wasserstein 半径 $\varepsilon$ 恰等于最优预测分布下的**期望传输代价**。这与最优传输理论中的互补松弛条件一致：在最优解处，Wasserstein 球约束恰好用尽（不浪费也不超额）。

### 2.4 KKT 方程组

综合以上分析，最优解 $(\lambda^*, \mathbf{s}^*)$ 满足如下 $N+1$ 个非线性方程（$N+1$ 个未知量）：

$$
\boxed{
\begin{cases}
\displaystyle\int_{\Omega_i(\lambda, \mathbf{s})} \exp\!\big(s_i - \lambda\varphi_i(x)\big)\, dx = \frac{1}{N}, & i = 1, \dots, N \\[10pt]
\displaystyle\sum_{i=1}^N \int_{\Omega_i(\lambda, \mathbf{s})} \varphi_i(x) \cdot \exp\!\big(s_i - \lambda\varphi_i(x)\big)\, dx = \varepsilon &
\end{cases}
}
$$

其中 $\Omega_i(\lambda, \mathbf{s})$ 是由 $(\lambda, \mathbf{s})$ 决定的功率胞腔。这个方程组的结构是后续所有求解算法的基础。

---

## 3. 解法一：一维精确解析算法

当状态空间 $\mathcal{X} = \mathbb{R}$（$d = 1$）时，积分约束可以**精确解析计算**，从而将问题完全转化为标准非线性优化。

### 3.1 功率图的一维结构：上包络线

在一维情形下，每个核函数 $g_i(x) = s_i - \lambda\varphi_i(x)$ 具有简单的几何形状：

**定理 1（尖顶帐篷函数）：** $g_i(x) = s_i - \lambda|x - c_i|$ 是以 $c_i = \hat{x}_i^{pred}$ 为中心、峰值为 $s_i$、两侧斜率为 $\pm\lambda$ 的帐篷函数。

**定理 2（平顶梯形函数）：** $g_i(x) = s_i - \lambda\max(0, |x - c_i| - R_i)$ 是以 $c_i$ 为中心、在 $[c_i - R_i, c_i + R_i]$ 上取平顶值 $s_i$、两侧以斜率 $\lambda$ 衰减的梯形函数。

被积函数 $\max_i \exp(g_i(x)) = \exp(\max_i g_i(x))$，因此问题归结为计算这些函数的**上包络线（Upper Envelope）** $E(x) = \max_i g_i(x)$ 并对 $e^{E(x)}$ 积分。

### 3.2 上包络线的计算

#### 3.2.1 定理 1 的帐篷函数上包络线

每个帐篷函数由两条半直线组成：
- 左支：$g_i^L(x) = \lambda x + (s_i - \lambda c_i)$，斜率 $+\lambda$（$x \leq c_i$）
- 右支：$g_i^R(x) = -\lambda x + (s_i + \lambda c_i)$，斜率 $-\lambda$（$x \geq c_i$）

**关键观察：** 所有左支平行（斜率均为 $+\lambda$），所有右支平行（斜率均为 $-\lambda$）。因此：
- 在所有核的最左侧（$x \ll \min_i c_i$），**左截距最大**的核支配：$i_L^* = \arg\max_i (s_i - \lambda c_i)$
- 在所有核的最右侧（$x \gg \max_i c_i$），**右截距最大**的核支配：$i_R^* = \arg\max_i (s_i + \lambda c_i)$
- 在中间区域，核的支配权可能交替变换

**相邻支配核的交界点：** 设核 $i$ 和核 $j$（$c_i < c_j$）在中间区域存在交界，则交界点为：
$$
x_{ij}^* = \frac{c_i + c_j}{2} + \frac{s_i - s_j}{2\lambda}
$$
（当 $|s_i - s_j| \leq \lambda(c_j - c_i)$ 时该交界点落在 $[c_i, c_j]$ 之间。）

**算法（栈式扫描法求上包络线）：**

```
输入: 排序后的核参数 {(c_i, s_i)}_{i=1}^N，c_1 ≤ c_2 ≤ ... ≤ c_N；斜率 λ
输出: 有序支配核列表 active[] 及交界点列表 breaks[]

1. 初始化栈 S = [1]
2. for i = 2, ..., N:
3.     while |S| ≥ 2:
4.         j ← S.top()
5.         k ← S.second()
6.         x_kj ← (c_k + c_j)/2 + (s_k - s_j)/(2λ)   // k-j 交界
7.         x_ji ← (c_j + c_i)/2 + (s_j - s_i)/(2λ)   // j-i 交界
8.         if x_ji ≤ x_kj:   // 核 j 被 k 和 i 完全遮盖
9.             S.pop()
10.        else:
11.            break
12.    S.push(i)
13. 从栈 S 中提取 active[]，计算相邻核的交界点得 breaks[]
```

时间复杂度：排序 $O(N \log N)$，扫描 $O(N)$（每个核至多入栈和出栈各一次）。

#### 3.2.2 定理 2 的梯形函数上包络线

梯形函数由三段组成：
- 左斜面：$g_i(x) = \lambda x + (s_i - \lambda(c_i - R_i))$，$x \leq c_i - R_i$
- 平顶：$g_i(x) = s_i$，$c_i - R_i \leq x \leq c_i + R_i$
- 右斜面：$g_i(x) = -\lambda x + (s_i + \lambda(c_i + R_i))$，$x \geq c_i + R_i$

上包络线的计算方法与 3.2.1 类似，但需额外处理平顶区域的交叠。核心区别：当两个核的平顶区域重叠时（$c_i + R_i \geq c_j - R_j$），峰值较高者在重叠段直接支配；不重叠时，核 $i$ 的右斜面与核 $j$ 的左斜面的交界点为：
$$
x_{ij}^* = \frac{(c_i + R_i) + (c_j - R_j)}{2} + \frac{s_i - s_j}{2\lambda}
$$

算法框架与 3.2.1 的栈式扫描完全相同，仅需在交界点公式中将中心 $c_i$ 替换为有效边界 $c_i + R_i$（右侧）或 $c_i - R_i$（左侧）。

### 3.3 积分的闭式计算

上包络线将 $\mathbb{R}$ 分解为若干不重叠的区间 $\{I_m\}$，在每个区间上由固定的核 $i(m)$ 支配。积分分解为：
$$
G(\lambda, \mathbf{s}) = \sum_m \int_{I_m} \exp\!\big(s_{i(m)} - \lambda\varphi_{i(m)}(x)\big)\, dx
$$

每个分段积分均有闭式解。

#### 3.3.1 定理 1（帐篷核）的分段积分

$$
\int_a^b e^{s_i - \lambda|x - c_i|}\, dx =
\begin{cases}
\dfrac{e^{s_i}}{\lambda}\Big[e^{-\lambda(a - c_i)} - e^{-\lambda(b - c_i)}\Big], & a \geq c_i \\[8pt]
\dfrac{e^{s_i}}{\lambda}\Big[e^{\lambda(b - c_i)} - e^{\lambda(a - c_i)}\Big], & b \leq c_i \\[8pt]
\dfrac{e^{s_i}}{\lambda}\Big[2 - e^{\lambda(a - c_i)} - e^{-\lambda(b - c_i)}\Big], & a < c_i < b
\end{cases}
$$

半无界积分（用于最左和最右区间）：
$$
\int_{-\infty}^{b} e^{s_i - \lambda|x - c_i|}\, dx =
\begin{cases}
\dfrac{e^{s_i}}{\lambda} e^{\lambda(b - c_i)}, & b \leq c_i \\[6pt]
\dfrac{e^{s_i}}{\lambda}\Big(2 - e^{-\lambda(b - c_i)}\Big), & b > c_i
\end{cases}
$$
$$
\int_a^{+\infty} e^{s_i - \lambda|x - c_i|}\, dx =
\begin{cases}
\dfrac{e^{s_i}}{\lambda} e^{-\lambda(a - c_i)}, & a \geq c_i \\[6pt]
\dfrac{e^{s_i}}{\lambda}\Big(2 - e^{\lambda(a - c_i)}\Big), & a < c_i
\end{cases}
$$

全空间积分：$\int_{-\infty}^{+\infty} e^{s_i - \lambda|x - c_i|}\, dx = \frac{2e^{s_i}}{\lambda}$。

#### 3.3.2 定理 2（平顶核）的分段积分

平顶核 $e^{s_i - \lambda\max(0, |x-c_i|-R_i)}$ 在区间 $[a, b]$ 上的积分分解为**平顶段**和**指数尾**两部分：

- 平顶段贡献：$e^{s_i}$ 乘以区间 $[a,b]$ 与 $[c_i - R_i,\; c_i + R_i]$ 的交集长度
- 左指数尾（$x < c_i - R_i$）：等价于以 $c_i - R_i$ 为中心的帐篷核左半部分的积分
- 右指数尾（$x > c_i + R_i$）：等价于以 $c_i + R_i$ 为中心的帐篷核右半部分的积分

全空间积分：
$$
\int_{-\infty}^{+\infty} e^{s_i - \lambda\max(0, |x-c_i|-R_i)}\, dx = e^{s_i}\!\left(2R_i + \frac{2}{\lambda}\right)
$$

### 3.4 梯度的解析表达

在基于梯度的优化算法中，需要计算 $\nabla G$。由第 2 节的推导：

$$
\frac{\partial G}{\partial s_i} = \int_{\Omega_i} \hat{p}(x)\, dx \quad (\text{即第 } i \text{ 个核在其支配区间上的积分})
$$
$$
\frac{\partial G}{\partial \lambda} = -\sum_{i=1}^N \int_{\Omega_i} \varphi_i(x) \cdot \hat{p}(x)\, dx
$$

这些梯度可在计算 $G$ 的同一轮分段积分中**顺带计算**，无需额外代价。以定理 1 为例，$\lambda$ 方向的加权积分 $\int_a^b |x - c_i| \cdot e^{s_i - \lambda|x-c_i|}\, dx$ 可通过对 $\lambda$ 求导或分部积分获得闭式解：

$$
\int_a^b |x - c_i| \cdot e^{s_i - \lambda|x - c_i|}\, dx = -\frac{\partial}{\partial \lambda}\int_a^b e^{s_i - \lambda|x - c_i|}\, dx
$$

对 3.3.1 中的闭式结果关于 $\lambda$ 求导即可。

### 3.5 一维精确求解算法

**Algorithm 1: 一维 Wasserstein DRPP 精确求解**

```
输入: 锚点 {c_i}，不确定性半径 {R_i}（定理 1 时 R_i ≡ 0），
      Wasserstein 半径 ε，样本量 N
输出: 最优参数 (λ*, s*)

1. 初始化 (λ⁰, s⁰) ← LSE 松弛闭式解（见第 4 节）

2. repeat:
3.     // 步骤 A: 计算上包络线
4.     segments ← UpperEnvelope({c_i, s_i, R_i}, λ)

5.     // 步骤 B: 分段闭式积分，同步计算 G, ∂G/∂s, ∂G/∂λ
6.     G ← 0;  ∂G/∂s_i ← 0 (∀i);  ∂G/∂λ ← 0
7.     for each segment (interval [a,b], kernel i) in segments:
8.         val ← ClosedFormIntegral(a, b, s_i, λ, c_i, R_i)
9.         G += val
10.        ∂G/∂s_i += val
11.        ∂G/∂λ -= ClosedFormWeightedIntegral(a, b, s_i, λ, c_i, R_i)

12.    // 步骤 C: 用 L-BFGS 或 SLSQP 更新 (λ, s)
13.    传入目标函数 -J, 约束 G ≤ 1 及其梯度，执行优化单步

14. until 收敛（|ΔJ| < tol 且 |G - 1| < tol）

15. return (λ*, s*)
```

**复杂度：** 每次迭代 $O(N \log N)$（排序+扫描+积分）。总复杂度 $O(K \cdot N \log N)$，$K$ 为优化迭代数。

**实现建议：** 可用 `scipy.optimize.minimize(method='SLSQP')`，在目标函数和约束的回调中调用上包络线 + 闭式积分子程序。

---

## 4. 解法二：LSE 松弛解析界

通过 Log-Sum-Exp 松弛，可获得定理 1 和定理 2 最优值的**解析闭式下界**，同时为数值优化提供高质量初始点。

### 4.1 松弛原理

利用不等式 $\max_i a_i \leq \sum_i a_i$（对非负数成立）：
$$
G(\lambda, \mathbf{s}) = \int_{\mathcal{X}} \max_i \exp\!\big(s_i - \lambda\varphi_i(x)\big)\, dx \leq \sum_{i=1}^N \int_{\mathcal{X}} \exp\!\big(s_i - \lambda\varphi_i(x)\big)\, dx =: \bar{G}(\lambda, \mathbf{s})
$$

约束 $\bar{G} \leq 1$ 比 $G \leq 1$ 更严格（可行域更小），因此松弛问题的最优值 $\leq$ 原问题最优值。但 $\bar{G}$ 中的每个积分可以**解析计算**。

**命题（定理 2 的 LSE 仍是单步 DRPP 严格原问题下界）**  
记定理 2 的有限维问题最优值为
$$
V_{\mathrm{low}}
:=
\max_{\lambda\ge0,\mathbf s}\left\{-\lambda\varepsilon+\frac1N\sum_{i=1}^Ns_i\right\}
\quad\text{s.t.}\quad
G(\lambda,\mathbf s)\le1,
$$
其中
$$
G(\lambda,\mathbf s)=\int_{\mathcal X}\max_i \exp\!\big(s_i-\lambda\varphi_i(x)\big)\,dx,\qquad
\varphi_i(x)=\max\!\big(0,\|x-\hat x_i^{pred}\|-R_i\big).
$$
记其加性 LSE 松弛最优值为
$$
V_{\mathrm{LSE}}
:=
\max_{\lambda\ge0,\mathbf s}\left\{-\lambda\varepsilon+\frac1N\sum_{i=1}^Ns_i\right\}
\quad\text{s.t.}\quad
\bar G(\lambda,\mathbf s)\le1,
$$
$$
\bar G(\lambda,\mathbf s):=\int_{\mathcal X}\sum_{i=1}^N \exp\!\big(s_i-\lambda\varphi_i(x)\big)\,dx.
$$
则有
$$
V_{\mathrm{LSE}}\le V_{\mathrm{low}}\le V_{\mathrm{strict}},
$$
其中 $V_{\mathrm{strict}}$ 是单步严格耦合 DRPP 原问题最优值（见定理 2 结论）。

**证明：**
1. 对任意固定 $(\lambda,\mathbf s)$ 与任意 $x$，令
   $$
   u_i(x):=\exp\!\big(s_i-\lambda\varphi_i(x)\big)\ge0,
   $$
   则
   $$
   \max_i u_i(x)\le \sum_i u_i(x).
   $$
   对 $x$ 积分得
   $$
   G(\lambda,\mathbf s)\le \bar G(\lambda,\mathbf s).
   $$
2. 因此
   $$
   \{(\lambda,\mathbf s):\bar G\le1\}\subseteq \{(\lambda,\mathbf s):G\le1\}.
   $$
   两个问题目标函数相同且都是最大化，故
   $$
   V_{\mathrm{LSE}}\le V_{\mathrm{low}}.
   $$
3. 定理 2 已证明
   $$
   V_{\mathrm{low}}\le V_{\mathrm{strict}}.
   $$
   链式合并即得
   $$
   V_{\mathrm{LSE}}\le V_{\mathrm{low}}\le V_{\mathrm{strict}}.
   $$
命题证毕。$\square$

### 4.2 全空间单核积分

记 $\mathcal{X} = \mathbb{R}^d$。定义 $d$ 维全空间积分常数：

**定理 1（Laplace 核）：**
$$
\mathcal{I}_d^{\text{Lap}}(\lambda) := \int_{\mathbb{R}^d} e^{-\lambda\|x\|}\, dx = \frac{2\pi^{d/2}\,\Gamma(d)}{\Gamma(d/2)\,\lambda^d}
$$
（通过 $d$ 维球坐标变换：面积元 $S_{d-1} = \frac{2\pi^{d/2}}{\Gamma(d/2)}$，径向积分 $\int_0^\infty r^{d-1} e^{-\lambda r} dr = \frac{\Gamma(d)}{\lambda^d}$。）

特别地：$d=1$ 时 $\mathcal{I}_1 = 2/\lambda$；$d=2$ 时 $\mathcal{I}_2 = 2\pi/\lambda^2$；$d=3$ 时 $\mathcal{I}_3 = 8\pi/\lambda^3$。

**定理 2（Hinge-Laplace 核）：**
$$
\mathcal{I}_d^{\text{Hinge}}(\lambda, R) := \int_{\mathbb{R}^d} e^{-\lambda\max(0, \|x\|-R)}\, dx = V_d R^d + S_{d-1} \sum_{k=0}^{d-1} \binom{d-1}{k} \frac{k!\, R^{d-1-k}}{\lambda^{k+1}}
$$
其中 $V_d = \frac{\pi^{d/2}}{\Gamma(d/2+1)}$ 是 $d$ 维单位球体积，$S_{d-1} = \frac{2\pi^{d/2}}{\Gamma(d/2)}$ 是 $(d-1)$ 维单位球面面积。

（推导：将积分拆为球内（$\|x\| \leq R$，被积函数为 1）和球外（$\|x\| > R$，换元 $t = r - R$），球外部分对 $(t+R)^{d-1}$ 作二项式展开后逐项积分。）

特别地：$d=1$ 时 $\mathcal{I}_1^{\text{Hinge}} = 2R + 2/\lambda$。

### 4.3 松弛问题的闭式解

**定理 1 的 LSE 松弛：**

松弛约束为 $\bar{G} = \sum_{i=1}^N e^{s_i} \cdot \mathcal{I}_d^{\text{Lap}}(\lambda) \leq 1$。

由目标函数 $\frac{1}{N}\sum s_i$ 的**对称性**和约束 $\sum e^{s_i}$ 的**Schur 凸性**，最优解满足 $s_1 = s_2 = \cdots = s_N =: s$。

代入约束取等：$Ne^s \cdot \mathcal{I}_d^{\text{Lap}}(\lambda) = 1$，即：
$$
s = -\log N - \log \mathcal{I}_d^{\text{Lap}}(\lambda) = -\log N + d\log\lambda - \log C_d
$$
其中 $C_d = \frac{2\pi^{d/2}\Gamma(d)}{\Gamma(d/2)}$。

目标函数化为单变量函数：
$$
J(\lambda) = -\lambda\varepsilon + s = -\lambda\varepsilon - \log N + d\log\lambda - \log C_d
$$

对 $\lambda$ 求导令其为零：$-\varepsilon + d/\lambda = 0$，得：

$$
\boxed{\lambda^*_{\text{LSE}} = \frac{d}{\varepsilon}, \qquad s^*_{\text{LSE}} = -\log N + d\log\frac{d}{\varepsilon} - \log C_d}
$$
$$
\boxed{J^*_{\text{LSE}} = -d + d\log\frac{d}{\varepsilon} - \log N - \log C_d}
$$

**示例（$d=1$）：** $C_1 = 2$，故 $\lambda^* = 1/\varepsilon$，$s^* = -\log(2N\varepsilon)$，$J^* = -1 - \log(2N\varepsilon)$。

**定理 2 的 LSE 松弛：**

约束为 $\sum_{i=1}^N e^{s_i} \cdot \mathcal{I}_d^{\text{Hinge}}(\lambda, R_i) \leq 1$。由于 $R_i$ 各不相同，$s_i$ 一般不全相等。利用 KKT 条件可得：
$$
s_i^* = -\log\!\big(N \cdot \mathcal{I}_d^{\text{Hinge}}(\lambda^*, R_i)\big), \quad i = 1, \dots, N
$$
其中 $\lambda^*$ 满足一维方程：
$$
\varepsilon = \frac{1}{N}\sum_{i=1}^N \frac{-\frac{\partial}{\partial\lambda}\mathcal{I}_d^{\text{Hinge}}(\lambda, R_i)}{\mathcal{I}_d^{\text{Hinge}}(\lambda, R_i)}
$$

此方程可通过牛顿法或二分法在 $\lambda > 0$ 上快速求解。

### 4.4 松弛精度分析与参数化改进

LSE 松弛的精度取决于核函数之间的**重叠程度**：
- 当锚点 $\hat{x}_i^{pred}$ 彼此远离（间距 $\gg 1/\lambda$）时，各核几乎不重叠，$\sum_i \approx \max_i$，松弛近乎无损
- 当锚点密集聚集时，重叠严重，松弛较粗糙

#### 一维下“等价无穷小”结论（定理 1 问题）

下面给出一个可验证的充分条件，说明在一维中，当锚点逐渐分离时，LSE 加性松弛与原问题的最优值差异趋于 $0$。

> 注意：本结论比较的是**定理 1 对应上界子问题**（约束 $G\le 1$）与其 LSE 加性松弛（约束 $\bar G\le 1$），不是严格耦合原问题 $V_k^{strict}$。

设 $d=1$，$\mathcal X=\mathbb R$，$c_i:=\hat x_i^{pred}$，并定义
$$
f_i(x):=\exp\!\big(s_i-\lambda|x-c_i|\big),\qquad
G(\lambda,\mathbf s):=\int_{\mathbb R}\max_i f_i(x)\,dx,\qquad
\bar G(\lambda,\mathbf s):=\int_{\mathbb R}\sum_i f_i(x)\,dx.
$$
记原问题最优值为 $J^\star$，LSE 松弛最优值为 $\bar J^\star$，则总有 $\bar J^\star\le J^\star$。  
定义重叠误差
$$
\Delta_G(\lambda,\mathbf s):=\bar G(\lambda,\mathbf s)-G(\lambda,\mathbf s)\ge 0.
$$

对任意非负 $\{a_i\}$ 有点态不等式
$$
\sum_i a_i-\max_i a_i\le \sum_{i<j}\min(a_i,a_j),
$$
因此
$$
\Delta_G(\lambda,\mathbf s)\le \sum_{i<j}\int_{\mathbb R}\min\!\big(f_i(x),f_j(x)\big)\,dx.
$$

再用 $\min(u,v)\le \sqrt{uv}$ 得
$$
\int_{\mathbb R}\min(f_i,f_j)\,dx
\le
e^{(s_i+s_j)/2}\!\int_{\mathbb R}
e^{-\frac{\lambda}{2}(|x-c_i|+|x-c_j|)}dx.
$$
记 $\delta_{ij}:=|c_i-c_j|$，则一维恒等式
$$
|x-c_i|+|x-c_j|=\delta_{ij}+2\,\mathrm{dist}\!\left(x,[c_i,c_j]\right)
$$
给出
$$
\int_{\mathbb R}\min(f_i,f_j)\,dx
\le
e^{(s_i+s_j)/2}\,e^{-\lambda\delta_{ij}/2}\!\left(\delta_{ij}+\frac{2}{\lambda}\right).
$$

若 $(\lambda,\mathbf s)$ 满足原约束 $G\le 1$，则对每个 $i$：
$$
\int_{\mathbb R} f_i(x)\,dx=\frac{2e^{s_i}}{\lambda}\le \int_{\mathbb R}\max_j f_j(x)\,dx=G\le 1
\;\Longrightarrow\;
e^{s_i}\le \frac{\lambda}{2}.
$$
从而
$$
\Delta_G(\lambda,\mathbf s)
\le
\sum_{i<j}\left(\frac{\lambda\delta_{ij}}{2}+1\right)e^{-\lambda\delta_{ij}/2}.
\tag{4.4.1}
$$

现在取原问题最优解 $(\lambda^\star,\mathbf s^\star)$。令
$$
\eta^\star:=\log\!\big(1+\Delta_G(\lambda^\star,\mathbf s^\star)\big),\qquad
\mathbf s'=\mathbf s^\star-\eta^\star\mathbf 1.
$$
则
$$
\bar G(\lambda^\star,\mathbf s')
=e^{-\eta^\star}\bar G(\lambda^\star,\mathbf s^\star)
=\frac{G(\lambda^\star,\mathbf s^\star)+\Delta_G(\lambda^\star,\mathbf s^\star)}
{1+\Delta_G(\lambda^\star,\mathbf s^\star)}
\le 1,
$$
即 $(\lambda^\star,\mathbf s')$ 对松弛问题可行，且目标仅下降 $\eta^\star$。因此
$$
0\le J^\star-\bar J^\star\le \eta^\star
=\log\!\big(1+\Delta_G(\lambda^\star,\mathbf s^\star)\big).
\tag{4.4.2}
$$

若再假设：
1. 最小锚点间距 $\Delta:=\min_{i\ne j}\delta_{ij}\to\infty$（考虑一列问题）；  
2. 最优乘子有下界 $\lambda^\star\ge \lambda_0>0$；

则由 (4.4.1) 得 $\Delta_G(\lambda^\star,\mathbf s^\star)$ 指数衰减，进而
$$
0\le J^\star-\bar J^\star
\le C\,e^{-\lambda_0\Delta/2}=o(1)\qquad(\Delta\to\infty),
$$
其中 $C$ 与 $N$ 及锚点几何上界有关。  
这就给出了“一维下 LSE 松弛差异为等价无穷小”的一组充分条件。

**参数化松弛改进：** 利用 $\ell_p$-范数的极限性质：
$$
\max_i a_i = \lim_{p \to \infty} \Big(\sum_i a_i^p\Big)^{1/p}
$$
取有限 $p$ 值，定义：
$$
\bar{G}_p(\lambda, \mathbf{s}) = \int_{\mathcal{X}} \Big(\sum_{i=1}^N \exp\!\big(p(s_i - \lambda\varphi_i(x))\big)\Big)^{1/p}\, dx
$$

满足 $G \leq \bar{G}_p \leq G \cdot N^{1/p}$（即近似误差至多 $N^{1/p}$ 倍）。$p = 1$ 退化为本节的加性松弛；$p \to \infty$ 恢复原问题。实际中取 $p = 5 \sim 10$ 即可获得较好的近似精度。$\bar{G}_p$ 对 $(\lambda, \mathbf{s})$ 处处可微，便于梯度优化。

### 4.5 保持“上界性质”的平滑近似（定理 1）

上面的加性 LSE（$\max \le \sum$）会把定理 1 的可行域缩小，因此最优值变小；它一般不能保持“对严格原问题的上界”性质。  
如果希望近似后依然保持上界，可改用**下逼近 max** 的平滑函数：

$$
m_\tau^{-}(A_1,\dots,A_N)
:=
\frac{1}{\tau}\log\!\sum_{i=1}^N e^{\tau A_i}
-\frac{\log N}{\tau},\qquad \tau>0.
$$

由 LSE 基本不等式可得
$$
\max_i A_i-\frac{\log N}{\tau}\le m_\tau^{-}(A)\le \max_i A_i.
\tag{4.5.1}
$$

对定理 1，令
$$
A_i(x)=s_i-\lambda\|x-\hat x_i^{pred}\|,\qquad
G(\lambda,\mathbf s)=\int_{\mathcal X}\exp\!\big(\max_i A_i(x)\big)\,dx.
$$
定义“保上界平滑约束函数”
$$
G_\tau^{-}(\lambda,\mathbf s)
:=
\int_{\mathcal X}\exp\!\big(m_\tau^{-}(A_1(x),\dots,A_N(x))\big)\,dx.
$$
由 (4.5.1) 逐点指数化并积分，得到
$$
G_\tau^{-}(\lambda,\mathbf s)\le G(\lambda,\mathbf s).
\tag{4.5.2}
$$

据此构造近似问题
$$
V_\tau^{-}
:=
\max_{\lambda\ge0,\mathbf s}
\left\{-\lambda\varepsilon+\frac1N\sum_{i=1}^Ns_i\right\}
\quad
\text{s.t.}\quad
G_\tau^{-}(\lambda,\mathbf s)\le1.
$$

**命题（上界保持）**：设定理 1 精确值为 $\bar V$，严格原问题值为 $V_{\mathrm{strict}}$（满足 $V_{\mathrm{strict}}\le \bar V$）。则
$$
V_\tau^{-}\ge \bar V\ge V_{\mathrm{strict}}.
$$

**证明：**
1. 由 (4.5.2) 得：若 $G(\lambda,\mathbf s)\le1$，则必有 $G_\tau^{-}(\lambda,\mathbf s)\le1$。  
   因此可行域包含关系为
   $$
   \{G\le1\}\subseteq\{G_\tau^{-}\le1\}.
   $$
2. 两问题目标函数相同且均为最大化，故
   $$
   V_\tau^{-}\ge \bar V.
   $$
3. 由定理 1 对严格原问题的上界关系：$V_{\mathrm{strict}}\le \bar V$。  
   链式合并即得
   $$
   V_\tau^{-}\ge \bar V\ge V_{\mathrm{strict}}.
   $$
命题证毕。$\square$

另外，随着 $\tau\to\infty$，$m_\tau^{-}(A)\uparrow\max_i A_i$，因此 $G_\tau^{-}\uparrow G$，可行域收缩，$V_\tau^{-}$ 会从上方逼近 $\bar V$。

---

## 5. 解法三：一般维度数值优化算法

当状态空间维度 $d \geq 2$ 时，功率胞腔的几何结构复杂（高维功率图），闭式积分不再可用。我们给出基于**数值积分 + 梯度优化**的通用算法。

### 5.1 积分的数值计算

#### 方案 A：自适应数值积分（$d \leq 3$）

对于低维状态空间，使用**自适应高斯求积**。

被积函数 $h(x) = \max_i \exp(s_i - \lambda\varphi_i(x))$ 是连续的（虽然不光滑），且具有指数衰减。

**积分域截断.** 当 $\|x\|$ 足够大时 $h(x)$ 指数衰减至可忽略。将积分域截断为以锚点质心为中心、半径为 $R_{\text{cut}} = (\max_i s_i + M_{\text{tol}}) / \lambda$ 的球（取 $M_{\text{tol}} = 30$ 保证截断误差 $< e^{-30} \approx 10^{-13}$）。

实现工具：
- $d = 1$：`scipy.integrate.quad`
- $d = 2, 3$：`scipy.integrate.dblquad` / `tplquad`，或自适应 Gauss-Legendre 张量积求积

#### 方案 B：重要性采样 Monte Carlo（$d \geq 4$）

对于高维情形，使用 Monte Carlo 积分。以混合 Laplace 分布为**提议分布**：
$$
q(x) = \frac{1}{N}\sum_{i=1}^N \frac{\lambda^d}{C_d}\exp\!\big(-\lambda\|x - \hat{x}_i^{pred}\|\big)
$$

则：$G \approx \frac{1}{M}\sum_{m=1}^M \frac{h(x^{(m)})}{q(x^{(m)})}$，其中 $x^{(m)} \sim q$。

由于 $q$ 与被积函数形状匹配良好，方差较低，MC 估计高效。

#### 方案 C：LSE 平滑化（任意维度，推荐方案）

用温度参数 $\tau > 1$ 控制的 Log-Sum-Exp 替代不光滑的 $\max$：
$$
\max_i A_i \approx \text{LSE}_\tau(A_1, \dots, A_N) := \frac{1}{\tau}\log\sum_{i=1}^N e^{\tau A_i}
$$

满足 $\max_i A_i \leq \text{LSE}_\tau \leq \max_i A_i + \frac{\log N}{\tau}$。取 $\tau$ 足够大即可控制近似误差。

**优势：**
1. $\exp(\text{LSE}_\tau) = (\sum_i e^{\tau A_i})^{1/\tau}$ 对 $(\lambda, \mathbf{s})$ **处处可微**，梯度计算简单
2. 梯度具有 softmax 结构：$\frac{\partial}{\partial s_i}\exp(\text{LSE}_\tau) = \exp(\text{LSE}_\tau) \cdot \text{softmax}_i(\tau A_1, \dots, \tau A_N)$
3. 结合数值积分，可使用标准的 L-BFGS 或内点法优化

### 5.2 梯度的数值计算

无论使用哪种积分方案，梯度可从**同一批求积节点**中同步计算。

设在求积节点 $\{x^{(m)}\}_{m=1}^M$（带权重 $\{w_m\}$）上：
$$
G \approx \sum_m w_m \cdot h(x^{(m)})
$$

令 $i^*(x) = \arg\max_i(s_i - \lambda\varphi_i(x))$，则：
$$
\frac{\partial G}{\partial s_i} \approx \sum_{m:\, i^*(x^{(m)})=i} w_m \cdot h(x^{(m)})
$$
$$
\frac{\partial G}{\partial \lambda} \approx -\sum_m w_m \cdot \varphi_{i^*(x^{(m)})}(x^{(m)}) \cdot h(x^{(m)})
$$

若使用 LSE 平滑，则 $\arg\max$ 替换为 softmax 权重，梯度自动光滑。

### 5.3 优化算法

**Algorithm 2: 一般维度 Wasserstein DRPP 数值求解（对数障碍内点法）**

```
输入: 锚点 {c_i}，不确定性半径 {R_i}，维度 d，
      Wasserstein 半径 ε，样本量 N，初始 barrier 参数 μ₀
输出: 最优参数 (λ*, s*)

1. // 阶段 I: LSE 松弛闭式初始化
2. (λ, s) ← 第 4 节的解析解

3. // 阶段 II: 对数障碍内点法
4. μ ← μ₀
5. for outer = 1, 2, ...:
6.     // 求解 barrier 子问题:
7.     //   max  Φ_μ(λ, s) := J(λ, s) + μ · log(1 - G(λ, s))
8.     for inner = 1, 2, ...:
9.         // 数值积分计算 G 和 ∇G
10.        G_val, ∇G ← NumericalIntegration(λ, s)
11.
12.        // barrier 目标的梯度
13.        ∂Φ/∂s_i = 1/N - μ · (∂G/∂s_i) / (1 - G_val)
14.        ∂Φ/∂λ  = -ε  - μ · (∂G/∂λ)  / (1 - G_val)
15.
16.        // L-BFGS 更新 + 投影 λ ≥ δ > 0
17.        (λ, s) ← LBFGS_step(λ, s, ∇Φ)
18.        λ ← max(λ, δ)
19.
20.        if ‖∇Φ‖ < tol_inner: break
21.
22.    μ ← μ / 10
23.    if μ < tol_outer: break
24.
25. return (λ, s)
```

**收敛性保证：** 子问题为凸优化，对数障碍内点法在 $O(\sqrt{N+1} \log(1/\epsilon))$ 次外层迭代内收敛至 $\epsilon$-最优。

### 5.4 替代方案：直接求解 KKT 方程组

基于第 2.4 节的 KKT 方程组，可将优化问题转化为**非线性方程组求根**：

$$
F_i(\lambda, \mathbf{s}) := \int_{\Omega_i(\lambda, \mathbf{s})} \exp\!\big(s_i - \lambda\varphi_i(x)\big)\, dx - \frac{1}{N} = 0, \quad i = 1, \dots, N
$$
$$
F_{N+1}(\lambda, \mathbf{s}) := \sum_{i=1}^N \int_{\Omega_i} \varphi_i(x) \exp\!\big(s_i - \lambda\varphi_i(x)\big)\, dx - \varepsilon = 0
$$

使用**牛顿法**或**拟牛顿法**（Broyden 方法）求解 $F(\lambda, \mathbf{s}) = \mathbf{0}$。Jacobian 矩阵可从数值积分中计算。

**优势：** 直接利用等质量等结构性质，收敛速度快（牛顿法二次收敛）。

**劣势：** 需要良好初始点；功率胞腔在迭代中剧烈变化时 Jacobian 可能不连续。

---

## 6. 实现方案总结与推荐

| 方法 | 适用场景 | 精度 | 每次迭代复杂度 | 实现难度 |
|------|---------|------|--------------|---------|
| LSE 松弛闭式解（第 4 节） | 任意维度，快速初始化 / 粗估 | 近似（核重叠时偏差大） | $O(N)$ | 低 |
| 一维精确算法（第 3 节） | $d = 1$ | 精确（至机器精度） | $O(N\log N)$ | 中 |
| 数值积分 + 内点法（第 5 节） | $d \leq 3$（自适应求积）或任意 $d$（MC） | 可控精度 | $O(NM)$，$M$ 为求积点数 | 中高 |
| KKT 方程组牛顿法（第 5.4 节） | 任意维度 | 高（二阶收敛） | $O((N+1)^2 M)$ | 高 |

**推荐实施路线：**

1. **首先实现 LSE 松弛闭式解**（第 4 节）——代码量极少，提供解析基准值和优化初始点
2. **若 $d = 1$，实现精确算法**（第 3 节）——上包络线 + 闭式积分，无近似误差
3. **若 $d \geq 2$，实现数值优化**（第 5 节）——推荐 LSE 平滑（$\tau = 20$）+ 自适应求积 + L-BFGS

### 6.1 关于凸优化求解器的使用

虽然定理 1 和定理 2 的问题是凸优化，但由于积分约束的特殊性，**不能直接使用 CVXPY/MOSEK 等建模语言**（它们要求约束以 DCP 规则表示）。应使用以下工具：

- **`scipy.optimize.minimize`**：配合 `method='SLSQP'` 或 `method='trust-constr'`，传入自定义的目标函数、约束函数及其梯度
- **自定义内点法**：如 Algorithm 2 所述
- **`JAX` + 自动微分**：将积分约束用 LSE 平滑 + 数值求积实现，利用 JAX 自动获取精确梯度，配合 `jaxopt` 优化

### 6.2 收敛验证

无论采用哪种方法，应检验以下最优性条件作为 sanity check：

1. **约束活跃性：** $|G(\lambda^*, \mathbf{s}^*) - 1| < \text{tol}$
2. **等质量性质：** $\left|\int_{\Omega_i} \hat{p}^*(x)\,dx - \frac{1}{N}\right| < \text{tol}$，$\forall\, i$
3. **Wasserstein 对偶：** $\left|\mathbb{E}_{\hat{p}^*}[\varphi_{i^*(x)}(x)] - \varepsilon\right| < \text{tol}$（当 $\lambda^* > 0$）
4. **归一化：** $\int_{\mathcal{X}} \hat{p}^*(x)\,dx = 1$（由条件 1 自动保证）
