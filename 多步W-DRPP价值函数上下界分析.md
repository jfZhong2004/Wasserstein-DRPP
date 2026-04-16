# 多步 Wasserstein-DRPP 价值函数上下界分析

本文档系统分析基于 Wasserstein 模糊集的多步分布鲁棒概率预测（W-DRPP）框架中，鲁棒价值函数 $V_k^*(z)$ 的上界与下界。分析基于 `DRPP.md`（矩模糊集方法）、`定义与上下界证明.md`（Wasserstein 模糊集方法）中的理论框架，重点探索是否存在显式或可直接计算的界。

---

## 1. 问题回顾

### 1.1 多步 DRPP 的鲁棒价值函数

多步 DRPP 问题通过动态规划原理，归结为求解鲁棒 Bellman 方程。在 Wasserstein 严格耦合模糊集设定下，价值函数满足：

$$
V_k^*(z) = \max_{\hat{p}_k \in \mathcal{F}} \inf_{\substack{f_k:\; \|f_k(\cdot)-\bar{f}_k(\cdot)\|_2^2 \leq \gamma_0(\cdot)}} \inf_{P_w \in \mathbb{B}_\varepsilon(\hat{P}_{w,N}^{f_k})} \mathbb{E}_{P_w} \left[ \log \hat{p}_k(f_k(z) + w) + V_{k+1}^*(f_k(z) + w) \right]
$$

终端条件 $V_T^* \equiv 0$。

### 1.2 核心困难

这是一个**无穷维嵌套极小极大问题**：预测者在泛函空间 $\mathcal{F}$ 中选择预测密度 $\hat{p}_k$，大自然在函数空间中选择 $f_k$ 并在概率测度空间中选择 $P_w$。三者交织耦合，直接求解不可行。因此需要上下界来夹逼原始问题的最优值。

### 1.3 两种模糊集框架的对比

| 特征 | 矩模糊集（DRPP.pdf） | Wasserstein 模糊集（定义与上下界证明.md） |
|------|----------------------|------------------------------------------|
| 不确定性刻画 | 均值/协方差的锥约束 $\gamma_1, \gamma_2, \gamma_3$ | Wasserstein 球 $\mathbb{B}_\varepsilon(\hat{P}_{w,N})$ |
| 模型不确定性 | $\|f_k(z) - \bar{f}_k(z)\|_2^2 \leq \gamma_0(z)$ | 同左 |
| 数据驱动 | 依赖名义矩估计 | 直接利用经验分布 $\hat{P}_{w,N}$ |
| 最优预测分布族 | 高斯族（单步时） | 多峰指数核函数 |

---

## 2. 上界分析

### 2.1 上界策略：令 $\gamma_0(z) \equiv 0$

**核心思想：** 假设系统状态演化函数已知且精确，即 $f_k = \bar{f}_k$。此时大自然失去操控 $f_k$ 的自由度，模糊集缩小为

$$
\mathcal{I}_{k}^{W}(z)\big|_{\gamma_0=0} = \left\{ P_{\pmb{x}_{k+1}|z} : \pmb{x}_{k+1} = \bar{f}_k(z) + \pmb{w}_k,\; P_{\pmb{w}_k} \in \mathbb{B}_\varepsilon(\hat{P}_{w,N}^{\bar{f}_k}) \right\}
$$

由于 $\mathcal{I}_k^W|_{\gamma_0=0} \subset \mathcal{I}_k^{W,strict}$（大自然可选范围更小），因此：

$$
V_k^*\big|_{\gamma_0=0}(z) \geq V_k^{*,strict}(z)
$$

即 $\gamma_0 = 0$ 下的价值函数构成原始问题的**上界**。

### 2.2 单步上界的凸优化形式（定理 1）

在 $\gamma_0 \equiv 0$ 设定下，单步 DRPP 等价于如下有限维凸优化：

$$
\max_{\lambda \geq 0,\; \mathbf{s} \in \mathbb{R}^N} \quad -\lambda\varepsilon + \frac{1}{N}\sum_{i=1}^N s_i
$$
$$
\text{s.t.} \quad \int_{\mathcal{X}} \max_{1 \leq i \leq N} \exp\!\left(s_i - \lambda\|x - \hat{x}_i^{pred}\|\right) dx \leq 1
$$

其中预测锚点 $\hat{x}_i^{pred} = \bar{f}_k(z) + \hat{w}_{k,i}$，$\hat{w}_{k,i} = \hat{x}_{k+1,i} - \bar{f}_k(\hat{z}_{k,i})$。

最优预测分布为多峰指数核：
$$
\hat{p}_k^*(x) = \max_{1 \leq i \leq N} \exp\!\left(s_i^* - \lambda^*\|x - \hat{x}_i^{pred}\|\right)
$$

### 2.3 多步上界的递推构造

**关键挑战：** 与 DRPP.pdf 中矩模糊集下的 Noise-DRPP 不同（其单步最优值 $-\frac{1}{2}[d_x\log(2\pi) + d_x + \log\det(\gamma_2\bar{\Sigma}_k)]$ 不依赖于 $z$，可以直接递推求和），Wasserstein 模糊集下定理 1 的单步最优值**一般依赖于 $z$**（因为预测锚点 $\hat{x}_i^{pred}$ 和经验噪声样本 $\hat{w}_{k,i}$ 都依赖于当前状态）。

#### 方法 A：直接贪心递推（忽略未来价值函数）

如果在每步 Bellman 递推中忽略 $V_{k+1}^*$ 的贡献（即近似地用贪心单步策略替代全局最优策略），则多步上界可以逐步独立求解：

$$
\bar{V}_k^{upper}(z) \approx \sum_{t=k}^{T-1} \text{OPT}_{\text{Thm1}}(z_t)
$$

其中 $\text{OPT}_{\text{Thm1}}(z_t)$ 是时刻 $t$ 定理 1 凸优化问题的最优值。

**严格性分析：** 这并非严格的上界递推，因为每一步的最优预测策略可能不是全局最优的。但它提供了一个**实用的性能估计**。

#### 方法 B：保守上界——与 DRPP.pdf 方法的类比

参考 DRPP.pdf 中 Theorem 4 的上界构造思路，其核心是找到一个**不依赖 $z$ 的单步界**从而可以直接求和。在 Wasserstein 设定下，可以考虑：

**命题（Wasserstein 多步上界的保守估计）.** 如果对定理 1 的凸优化最优值取关于所有可能 $z$ 的上确界：

$$
\bar{U}_k := \sup_{z \in \mathcal{Z}} \text{OPT}_{\text{Thm1}}(z)
$$

则多步价值函数上界为：

$$
V_k^*(z) \leq \sum_{t=k}^{T-1} \bar{U}_t
$$

**困难：** 计算 $\bar{U}_k$ 本身可能不平凡，因为定理 1 的最优值隐式依赖 $z$（通过预测锚点）。实际操作中可能需要对 $z$ 做网格搜索或利用特殊结构（如线性系统）给出解析上界。

#### 方法 C（严格版）：直接基于 Theorem 3.2（Kantorovich-Rubinstein）

设
$$
\hat P_{N,z}^{pred}:=\frac1N\sum_{i=1}^N\delta_{\hat x_i^{pred}},\qquad
\Phi_k(z):=\sup_{\hat p_k\in\mathcal F}\inf_{P\in\mathbb B_\varepsilon(\hat P_{N,z}^{pred})}\mathbb E_P[\log \hat p_k(X)].
$$

**直接引用 `Wassersteein.md` Theorem 3.2（KR 对偶）**：对任意分布 $Q_1,Q_2$，
$$
W_1(Q_1,Q_2)=\sup_{f\in\text{Lip}_1}\left(\mathbb E_{Q_1}[f]-\mathbb E_{Q_2}[f]\right).
$$
其标准推论（KR-Lip）可直接由缩放得到：若 $g$ 为 $L_g$-Lipschitz，令 $\tilde g:=g/L_g$，则 $\tilde g\in\text{Lip}_1$。因此
$$
\mathbb E_{Q_1}[g]-\mathbb E_{Q_2}[g]
=
L_g\!\left(\mathbb E_{Q_1}[\tilde g]-\mathbb E_{Q_2}[\tilde g]\right)
\le
L_g\,W_1(Q_1,Q_2).
$$
对 $-g$ 重复同样推导可得
$$
\mathbb E_{Q_2}[g]-\mathbb E_{Q_1}[g]\le L_g\,W_1(Q_1,Q_2),
$$
合并即
$$
\big|\mathbb E_{Q_1}[g]-\mathbb E_{Q_2}[g]\big|\le L_g\,W_1(Q_1,Q_2). \tag{KR-Lip}
$$

**用 Theorem 3.2 证明方法 C：**

1. 固定任意预测密度 $\hat p$，令 $g(x)=\log\hat p(x)$，并假设 $g$ 是 $L_{\hat p}$-Lipschitz。  
2. 对任意 $P\in\mathbb B_\varepsilon(\hat P_{N,z}^{pred})$，由 (KR-Lip) 得
$$
\mathbb E_P[g]-\mathbb E_{\hat P_{N,z}^{pred}}[g]\ge -L_{\hat p}W_1(P,\hat P_{N,z}^{pred})\ge -L_{\hat p}\varepsilon.
$$
即
$$
\inf_{P\in\mathbb B_\varepsilon(\hat P_{N,z}^{pred})}\mathbb E_P[\log\hat p]
\ge
\frac1N\sum_{i=1}^N\log\hat p(\hat x_i^{pred})-L_{\hat p}\varepsilon. \tag{C1}
$$
3. 又因为 $\hat P_{N,z}^{pred}\in\mathbb B_\varepsilon(\hat P_{N,z}^{pred})$，必有
$$
\inf_{P\in\mathbb B_\varepsilon(\hat P_{N,z}^{pred})}\mathbb E_P[\log\hat p]
\le
\frac1N\sum_{i=1}^N\log\hat p(\hat x_i^{pred}). \tag{C2}
$$

由 (C1)-(C2) 对任意 $\hat p$ 成立。再定义
$$
J_k^{nom}(z):=\sup_{\hat p_k\in\mathcal F}\frac1N\sum_{i=1}^N\log\hat p_k(\hat x_i^{pred}),
\qquad
\bar L_k:=\sup_{\hat p_k\in\mathcal F}L_{\hat p_k}<\infty,
$$
即可得到
$$
J_k^{nom}(z)-\bar L_k\varepsilon
\le
\Phi_k(z)
\le
J_k^{nom}(z). \tag{C3}
$$
若再令 $\bar J_k:=\sup_{z\in\mathcal Z}J_k^{nom}(z)$，则得到不依赖 $z$ 的理论上界
$$
\Phi_k(z)\le \bar J_k.
$$

补充说明：Theorem 6.3 属于“凸损失”情形下的更强结果；方法 C 的严格证明本身只需要 Theorem 3.2。

**(iv) 从单步界到完整多步价值函数上界**

定义 $\gamma_0\equiv0$ 上界问题的 Bellman 递推：
$$
V_T^{up}\equiv 0,\qquad
V_k^{up}(z):=\sup_{\hat p_k\in\mathcal F}\inf_{P\in\mathbb B_\varepsilon(\hat P_{N,z}^{pred})}
\mathbb E_P\!\left[\log \hat p_k(X)+V_{k+1}^{up}(X)\right].
$$
对任意 $z$，有
$$
\begin{aligned}
V_k^{up}(z)
&\le
\sup_{\hat p_k\in\mathcal F}
\mathbb E_{\hat P_{N,z}^{pred}}\!\left[\log \hat p_k(X)+V_{k+1}^{up}(X)\right] \\
&\le
\sup_{\hat p_k\in\mathcal F}\mathbb E_{\hat P_{N,z}^{pred}}[\log \hat p_k(X)]
\;+\;
\sup_{x\in\mathcal X}V_{k+1}^{up}(x) \\
&=
J_k^{nom}(z)+M_{k+1},
\end{aligned}
$$
其中
$$
M_{k}:=\sup_{z\in\mathcal Z}V_k^{up}(z),\qquad M_T=0.
$$
因此
$$
M_k\le \bar J_k+M_{k+1}
\;\Longrightarrow\;
M_k\le \sum_{t=k}^{T-1}\bar J_t.
$$
从而对任意 $z$，
$$
V_k^{up}(z)\le \sum_{t=k}^{T-1}\bar J_t. \tag{C4}
$$
又由于 $\mathcal I_k^W|_{\gamma_0=0}\subset\mathcal I_k^{W,strict}$（上界问题的大自然更弱），有
$$
V_k^{strict}(z)\le V_k^{up}(z)\le \sum_{t=k}^{T-1}\bar J_t. \tag{C5}
$$

这就是方法 C 给出的“完整多步价值函数理论上界”。其可计算性取决于每个 $\bar J_t=\sup_{z}J_t^{nom}(z)$ 是否可有效评估（通常仍需数值上确界）。

**(v) 附加结构假设下的显式无 $z$ 上界**

若进一步假设对每个时刻 $t$ 存在已知常数 $M_t<\infty$，使得
$$
\sup_{\hat p_t\in\mathcal F}\ \sup_{x\in\mathcal X}\hat p_t(x)\le M_t, \tag{A-M}
$$
则对任意 $z$，
$$
J_t^{nom}(z)
=
\sup_{\hat p_t\in\mathcal F}\frac1N\sum_{i=1}^N\log \hat p_t(\hat x_{t,i}^{pred})
\le
\sup_{\hat p_t\in\mathcal F}\log\!\Big(\sup_x \hat p_t(x)\Big)
\le \log M_t.
$$
故
$$
\bar J_t\le \log M_t.
$$
代入 (C5) 得到显式、与 $z$ 无关的完整多步上界：
$$
V_k^{strict}(z)\le \sum_{t=k}^{T-1}\log M_t. \tag{C6}
$$
特别地，若 $M_t\equiv M$（与时刻无关），则
$$
V_k^{strict}(z)\le (T-k)\log M.
$$

该上界是显式可算的，但通常比 (C5) 更保守。

### 2.4 上界小结

| 上界方法 | 是否显式 | 是否依赖 $z$ | 紧致程度 |
|----------|---------|-------------|---------|
| 定理 1 凸优化（单步） | 数值求解 | 是 | 紧（精确单步界） |
| $\sup_z$ 保守化 + 求和 | 需数值 | 否 | 中等 |
| KR-Lipschitz 理论界（凸性附加时可对接 Thm 6.3） | 显式 | 否 | 松 |
| KR-Lipschitz + 密度上界假设 (A-M) | **显式闭式** | 否 | 更松 |

---

## 3. 下界分析

### 3.1 定义与上下界证明.md 中的下界（定理 2）

**策略：** 保守近似——允许大自然对每个样本 $i$ 独立选择偏移 $\Delta\nu_i$，而非共用全局 $\Delta\nu$。这给予大自然更多自由度，使结果更悲观，构成**下界**。

单步下界的凸优化形式：

$$
\max_{\lambda \geq 0,\; \mathbf{s} \in \mathbb{R}^N} \quad -\lambda\varepsilon + \frac{1}{N}\sum_{i=1}^N s_i
$$
$$
\text{s.t.} \quad \int_{\mathcal{X}} \max_{1 \leq i \leq N} \exp\!\left(s_i - \lambda\max\!\left(0,\; \|x - \hat{x}_i^{pred}\| - R_i\right)\right) dx \leq 1
$$

其中复合认知不确定性半径 $R_i = \sqrt{\gamma_0(z)} + \sqrt{\gamma_0(\hat{z}_{k,i})}$。

最优预测分布为**平顶指数核**：
$$
\hat{p}_k^*(x) = \max_{1 \leq i \leq N} \exp\!\left(s_i^* - \lambda^*\max\!\left(0,\; \|x - \hat{x}_i^{pred}\| - R_i\right)\right)
$$

**上下界关系：** $V^*_{\text{Thm2}} \leq V^*_{strict} \leq V^*_{\text{Thm1}}$

### 3.2 多步下界的递推困难

与上界面临类似困难：定理 2 的单步最优值依赖于 $z$（通过 $R_i$ 中的 $\gamma_0(z)$ 和预测锚点）。

### 3.3 是否存在显式或可直接计算的多步下界？

参考 DRPP.pdf 中 Theorem 6 的方法，我们分析在 Wasserstein 设定下构造显式多步下界的可行性。

#### 方法 1：DRPP.pdf 下界方法的移植（Eig-DRPP 类比）

**DRPP.pdf 的 Eig-DRPP 下界策略回顾：**
- 通过限制预测分布的协方差矩阵的特征向量与名义协方差一致（$\hat{Q}_k = Q_k$），将原始 NP-hard 的极小极大问题转化为可解的凸优化
- 利用 $\gamma_0(z)$ 的全局上界 $\Gamma_0$（使得 $\gamma_0(z) \leq \Gamma_0, \forall z$），消除单步值对 $z$ 的依赖
- 由此获得不依赖 $z$ 的显式下界，可通过 Bellman 递推直接求和

**在 Wasserstein 设定下的对应策略：**

如果存在全局上界 $\Gamma_0 \geq \gamma_0(z), \forall z \in \mathcal{Z}$，以及关于历史数据点的上界 $\Gamma_{0,i} \geq \gamma_0(\hat{z}_{k,i}), \forall i$，可以定义：

$$
\bar{R} := \sqrt{\Gamma_0} + \max_{1 \leq i \leq N}\sqrt{\Gamma_{0,i}}
$$

**命题（Wasserstein 保守多步下界）.** *假设存在 $\Gamma_0 \geq \gamma_0(z), \forall z \in \mathcal{Z}$，且对所有时刻 $t$ 和所有 $z$，将定理 2 中的 $R_i$ 统一替换为 $\bar{R}_t := \sqrt{\Gamma_0} + \max_i \sqrt{\gamma_0(\hat{z}_{t,i})}$，记对应的凸优化最优值为 $L_t(\bar{R}_t)$。则多步下界为：*

$$
V_k^*(z) \geq \sum_{t=k}^{T-1} L_t(\bar{R}_t)
$$

**证明思路：**
1. 由于 $\bar{R}_t \geq R_i$ 对所有 $i$ 成立，替换后大自然的"免费搬运"半径更大，约束更宽松，最优值更小（更悲观）
2. 更悲观的单步值不依赖 $z$（仅依赖数据和全局参数），因此可以通过 Bellman 递推直接累加
3. 每一步的 $L_t(\bar{R}_t)$ 通过求解一个标准的有限维凸优化问题得到

**可计算性：** 每个 $L_t(\bar{R}_t)$ 是定理 2 类型的凸优化问题，仅需数值求解 $N+1$ 维变量 $(\lambda, \mathbf{s})$，使用内点法可高效求解。

#### 方法 2：一维情况下的解析下界探索

在一维（$d_x = 1$）情况下，可以尝试给出更明确的解析表达。

**一维简化：** 当 $\mathcal{X} = \mathbb{R}$ 时，定理 2 的约束积分可以显式化。对于单个指数核 $\exp(s_i - \lambda \max(0, |x - \hat{x}_i^{pred}| - R_i))$，在一维下其积分为：

$$
\int_{-\infty}^{+\infty} \exp(s_i - \lambda \max(0, |x - \hat{x}_i^{pred}| - R_i)) dx = e^{s_i}\left(2R_i + \frac{2}{\lambda}\right)
$$

当 $N = 1$ 时，约束简化为 $e^{s_1}(2R_1 + 2/\lambda) \leq 1$，最优解可以闭式求解：

$$
s_1^* = -\log\left(2R_1 + \frac{2}{\lambda^*}\right)
$$

目标函数变为 $-\lambda\varepsilon - \log(2R_1 + 2/\lambda)$，对 $\lambda$ 求导得最优条件：

$$
-\varepsilon + \frac{2}{\lambda^2(2R_1 + 2/\lambda)} = 0
$$

即 $\varepsilon\lambda^2(2R_1 + 2/\lambda) = 2$，可以化为关于 $\lambda$ 的二次方程。

当 $N > 1$ 时，由于 $\max$ 运算使得各指数核的支配区域（Voronoi-like partition）交织，闭式积分变得困难。但在一维等距分布的锚点情况下，仍可能通过分段积分给出半解析表达。

#### 方法 3：与矩模糊集 Eig-DRPP 下界的直接对比

DRPP.pdf Theorem 6 给出的矩模糊集下界为：

$$
V_k^*(z) \geq \sum_{t=k}^{T-1} -\frac{1}{2}\left\{ d_x\log(2\pi) + \sum_{i=1}^{d_x}\left[\log(\hat{\lambda}_{i,t}^*) + \gamma_2\frac{\lambda_{i,t}}{\hat{\lambda}_{i,t}^*}\right] + \frac{2\sqrt{\Gamma_0\gamma_1\lambda_{j_t^*,t}} + \Gamma_0}{\hat{\lambda}_{j_t^*,t}^*}\right\}
$$

其中 $\hat{\lambda}_{i,t}^*, j_t^*$ 通过求解凸优化 $\mathbf{P}_3$ 得到。

这个下界是**完全显式可计算的**——每个时刻只需求解 $d_x$ 个标量凸优化问题（对每个候选 $j_k$ 各一个），取最小值即可。

**可否为 Wasserstein 设定构造类似的显式下界？**

关键区别在于：
- 矩模糊集下，最优预测分布被证明为高斯分布，参数可显式求解
- Wasserstein 模糊集下，最优预测分布是多峰指数核，参数需数值求解

因此，**在 Wasserstein 设定下完全不依赖数值优化的显式闭式下界目前不可得**，但可以通过以下方式逼近：

**方案 3a：高斯次优策略下界.** 不使用定理 2 的最优多峰指数核分布，而是限制预测分布为高斯族 $\hat{p}_k \sim \mathcal{N}(\hat{\mu}_k, \hat{\Sigma}_k)$。由于高斯族是 $\mathcal{F}$ 的子集，限制后的最优值 $\leq$ 原始最优值，构成下界。此时问题变为：

$$
\max_{\hat{\mu}_k, \hat{\Sigma}_k} \inf_{P_w \in \mathbb{B}_\varepsilon(\hat{P}_{w,N})} \mathbb{E}_{P_w}\left[\log \mathcal{N}(\bar{f}_k(z) + w; \hat{\mu}_k, \hat{\Sigma}_k)\right]
$$

利用 Wasserstein 对偶和高斯对数密度的二次结构，内层可以化为关于 $(\hat{\mu}_k, \hat{\Sigma}_k)$ 的半定规划，可能给出半显式解。

**方案 3b：经验分布直接评估下界.** 对于任何固定的预测策略 $\hat{p}_k$，其在 Wasserstein 最坏情况下的性能为：

$$
\inf_{P_w \in \mathbb{B}_\varepsilon(\hat{P}_{w,N})} \mathbb{E}_{P_w}[\log \hat{p}_k] = \sup_{\lambda \geq 0}\left\{-\lambda\varepsilon + \frac{1}{N}\sum_{i=1}^N \inf_x (\log \hat{p}_k(x) + \lambda\|x - \hat{x}_i^{pred}\|)\right\}
$$

如果选择 Noise-DRPP 的高斯策略 $\hat{p}_k \sim \mathcal{N}(\bar{f}_k(z) + \bar{\mu}_k, \gamma_2\bar{\Sigma}_k)$（即 DRPP.pdf 的上界预测策略）作为 Wasserstein 问题的次优策略，则上式给出一个可以解析或数值评估的下界。

---

## 4. 综合分析与界的层次结构

### 4.1 完整的界的层次

将矩模糊集和 Wasserstein 模糊集的结果综合，多步价值函数的界可以排列为：

$$
\underbrace{V_{\text{Thm2}}^{*,W}(\bar{R})}_{\text{W-保守下界}} \;\leq\; \underbrace{V_{\text{Thm2}}^{*,W}}_{\text{W-下界(定理2)}} \;\leq\; \underbrace{V_{strict}^{*,W}}_{\text{W-原始问题}} \;\leq\; \underbrace{V_{\text{Thm1}}^{*,W}}_{\text{W-上界(定理1)}}
$$

其中上标 $W$ 表示 Wasserstein 模糊集。

注意：矩模糊集和 Wasserstein 模糊集下的界**不能直接比较大小**，因为它们对应的原始问题（模糊集定义）不同。两者的比较需要在实验中通过统一的真实系统来评估。

### 4.2 实际可计算性总结

| 界 | 形式 | 可计算性 | 依赖于 $z$？ | 可否直接递推？ |
|----|------|---------|-------------|--------------|
| W-上界（定理 1） | 凸优化 | 数值求解（内点法） | 是 | 需逐状态求解 |
| W-下界（定理 2） | 凸优化 | 数值求解（内点法） | 是 | 需逐状态求解 |
| W-保守多步下界 | 凸优化 + $\Gamma_0$ | 数值求解 | 否 | **可以直接求和** |
| 矩-上界（Noise-DRPP） | 显式闭式 | 直接计算 | 否 | **可以直接求和** |
| 矩-下界（Eig-DRPP） | 标量凸优化 | 高效求解 | 否（用 $\Gamma_0$） | **可以直接求和** |

### 4.3 Optimality Gap（最优性间隙）

类比 DRPP.pdf Theorem 6 的最优性间隙分析，W-DRPP 的最优性间隙为：

$$
\text{Gap}_k(z) = V_{\text{Thm1}}^{*,W}(z) - V_{\text{Thm2}}^{*,W}(z)
$$

在单步情况下，差异来源于：
- 定理 1 中距离惩罚为 $\lambda\|x - \hat{x}_i^{pred}\|$
- 定理 2 中距离惩罚为 $\lambda\max(0, \|x - \hat{x}_i^{pred}\| - R_i)$

二者之差反映了模型不确定性（$\gamma_0$）带来的额外保守性。当 $\gamma_0 \to 0$ 时，$R_i \to 0$，间隙消失。

多步间隙的保守上界可通过将每步间隙求和得到：

$$
\text{Gap}_{0:T-1} \leq \sum_{t=0}^{T-1} [\bar{U}_t - L_t(\bar{R}_t)]
$$

---

## 5. 可行的进一步探索方案

### 5.1 短期可行方向

1. **一维数值验证：** 在一维 $d_x = 1$ 的简单系统上，通过网格离散化精确求解定理 1 和定理 2 的凸优化问题，数值比较上下界的紧致程度，验证 $\text{Gap}$ 随 $\varepsilon$ 和 $\gamma_0$ 的变化规律。

2. **高斯次优下界的解析推导：** 将预测分布限制为高斯族后，利用 Wasserstein 对偶和高斯分布的二次对数密度结构，推导半解析的下界表达式，建立与矩模糊集 Eig-DRPP 下界的显式对比。

3. **保守多步下界的数值实现：** 实现方法 1 中的保守多步下界（用 $\bar{R}_t$ 替换 $R_i$），在实验中验证其相对于贪心递推的保守性程度。

### 5.2 中期研究方向

4. **近似动态规划（ADP）方法：** 对价值函数 $V_{k+1}^*$ 进行参数化近似（如分段线性或二次近似），代入 Bellman 方程中与定理 1/2 联合求解，获得更紧的多步界。

5. **信息松弛对偶界（Information Relaxation Duality）：** 利用 Brown, Smith & Sun (2010) 的信息松弛方法，构造多步问题的对偶上界。允许预测者获取未来信息但施加惩罚，得到的界通常比贪心递推更紧。

6. **Wasserstein 半径 $\varepsilon \to 0$ 的渐近分析：** 利用 `Wasserstein.pdf` Theorem 3.6 的渐近一致性结论，分析当样本量 $N \to \infty$（$\varepsilon \to 0$）时上下界的收敛速率和渐近间隙。

### 5.3 总结

**关于显式下界的结论：**

- 在 Wasserstein 模糊集设定下，**完全显式的闭式多步下界目前不可得**，主要原因是最优预测分布为非参数的多峰指数核，不像矩模糊集下可以得到高斯闭式解。
- 但通过 $\Gamma_0$ 保守化（方法 1）可以得到**不依赖 $z$ 的可递推下界**，且每步只需求解一个标准的 $N+1$ 维凸优化问题。
- 通过高斯次优策略（方案 3a）可能得到介于显式和数值之间的**半解析下界**，这是最有希望产生简洁结果的方向。
- 矩模糊集下 DRPP.pdf 的 Eig-DRPP 下界方法论（特征向量限制 + $\Gamma_0$ 统一化）提供了构造 Wasserstein 下界的重要参考范式。
