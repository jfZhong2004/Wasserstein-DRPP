# DRPP 定义及单步求解

## 0. 目标

本文基于 `新定义的构造方法与严格证明.md` 的模糊集定义，结合 `Wasserstein.md` 的求解结论，给出 Wasserstein-based DRPP（WDRPP）定义、单步问题与可解化结果，并补全严谨性所需的关键假设。

---

## 1. 原始模糊集与“球化”化简

在时刻 $k$、给定 $z\in\mathcal Z$ 时，定义
$$
\mathcal I_{k,N}^{W,\mathrm{raw}}(z;\beta_1,\beta_2;\bar f_k,\mathcal D_N)
:=
\left\{
(\tau_{g(z)})_\#Q:
\|g-\hat f_k\|_{\infty,2}\le \gamma_{k,N}(\beta_1),
W_1\!\left(Q,\hat P_{\omega,k,N}^{\hat f_k}\right)\le \varepsilon_{k,N}(\beta_2)
\right\}.
$$

其中
$$
\hat P_{\omega,k,N}^{\hat f_k}:=\frac1N\sum_{i=1}^N\delta_{\hat\omega_{k,i}},
\qquad
\hat\omega_{k,i}:=x_{k+1,i}-\hat f_k(z_{k,i}).
$$

定义中心经验预测测度
$$
\hat P_{x^+|z,k,N}^{\hat f_k}
:=
\frac1N\sum_{i=1}^N \delta_{\hat f_k(z)+\hat\omega_{k,i}},
\qquad
\rho_{k,N}:=\gamma_{k,N}(\beta_1)+\varepsilon_{k,N}(\beta_2).
$$

并定义球化集合
$$
\bar{\mathcal I}_{k,N}^{W}(z)
:=
\left\{
P\in\mathcal P_1(\mathbb R^{d_x}):
W_1\!\left(P,\hat P_{x^+|z,k,N}^{\hat f_k}\right)\le \rho_{k,N}
\right\}.
$$

### 命题 1（外包络包含）
对任意 $z$，
$$
\mathcal I_{k,N}^{W,\mathrm{raw}}(z;\beta_1,\beta_2;\bar f_k,\mathcal D_N)
\subseteq
\bar{\mathcal I}_{k,N}^{W}(z).
$$

**证明要点：**  
任取 $P=(\tau_{g(z)})_\#Q\in\mathcal I_{k,N}^{W,\mathrm{raw}}$，由三角不等式
$$
\begin{aligned}
W_1\!\left(P,\hat P_{x^+|z,k,N}^{\hat f_k}\right)
&\le
W_1\!\left((\tau_{g(z)})_\#Q,(\tau_{\hat f_k(z)})_\#Q\right)\\
&\quad+
W_1\!\left((\tau_{\hat f_k(z)})_\#Q,(\tau_{\hat f_k(z)})_\#\hat P_{\omega,k,N}^{\hat f_k}\right).
\end{aligned}
$$
第一项等于 $\|g(z)-\hat f_k(z)\|_2\le\gamma_{k,N}$，第二项由平移不变性等于
$W_1(Q,\hat P_{\omega,k,N}^{\hat f_k})\le\varepsilon_{k,N}$，故总和不超过 $\rho_{k,N}$。

### 覆盖概率说明
若 `新定义的构造方法与严格证明.md` 中对应高概率事件成立：
$$
\|f_k-\hat f_k\|_{\infty,2}\le\gamma_{k,N},\qquad
W_1(P_{\omega,k},\hat P_{\omega,k,N}^{\hat f_k})\le\varepsilon_{k,N}
$$
且其概率至少 $1-\beta$（例如 $\beta_1+\beta_2=\beta$），则对所有 $z$ 有
$$
P^\star_{x^+|z,k}\in\bar{\mathcal I}_{k,N}^{W}(z)
$$
同样以概率至少 $1-\beta$ 成立。  
因此球化化简保留（不弱于）同级别覆盖保证，但会更保守。

---

## 2. WDRPP（多步）定义

### 假设 B（Bellman 递推所需）
1. **矩形性（rectangularity）**：对每个时刻/状态，系统可从该时刻对应集合独立选取条件核；  
2. **可测性与可积性**：$\hat p_k(\cdot\mid z)$ 可测且在相关分布下 $\log\hat p_k$ 可积；  
3. **策略可测性**：控制策略 $\pi$ 与预测策略族 $\mathfrak F$ 满足标准可测选择条件。

给定时域 $k=0,\dots,T-1$，预测器策略
$$
\mathcal F=\{\hat p_0,\dots,\hat p_{T-1}\},\qquad \hat p_k(\cdot\mid z_k)\in\mathfrak F_k.
$$
评分函数取
$$
\mathcal L(\hat p_k,x):=\log \hat p_k(x\mid z_k).
$$

性能指标
$$
J_k^{\mathcal F,\mathcal P}(z)
:=
\mathbb E_{\mathcal P_{k:T-1}}
\!\left[
\sum_{t=k}^{T-1}\log \hat p_t(x_{t+1}\mid z_t)\ \middle|\ z_k=z
\right].
$$

系统核集合（球化版本）：
$$
\mathfrak P^{W}
:=
\left\{
\mathcal P:
P_{x_{t+1}\mid z_t=z}\in \bar{\mathcal I}_{t,N}^{W}(z),\ \forall t,\forall z
\right\}.
$$

WDRPP：
$$
(\mathbf P_0^W):
\quad
\sup_{\mathcal F\in\mathfrak F}\ \inf_{\mathcal P\in\mathfrak P^W}
J_0^{\mathcal F,\mathcal P}(z_0).
$$

在假设 B 下，其 Bellman 递推可写为
$$
V_T^W(z)=0,
$$
$$
V_k^W(z)=
\sup_{\hat p_k\in\mathfrak F_k}
\inf_{Q\in \bar{\mathcal I}_{k,N}^{W}(z)}
\int_{\mathcal X}
\left[
\log \hat p_k(x\mid z)+V_{k+1}^W(x,\pi_{k+1}(x))
\right]Q(dx).
$$

---

## 3. 单步 WDRPP：原始与化简

原始单步问题：
$$
(\mathbf P_{1,k}^{W,\mathrm{raw}}(z)):
\sup_{\hat p_k\in\mathfrak F_k}
\inf_{Q\in\mathcal I_{k,N}^{W,\mathrm{raw}}(z)}
\mathbb E_Q[\log \hat p_k(X\mid z)].
$$

球化单步问题：
$$
(\mathbf P_{1,k}^{W,\mathrm{ball}}(z)):
\sup_{\hat p_k\in\mathfrak F_k}
\inf_{Q\in\bar{\mathcal I}_{k,N}^{W}(z)}
\mathbb E_Q[\log \hat p_k(X\mid z)].
$$

由命题 1（内层可行域变大）可知
$$
V_{1,k}^{W,\mathrm{ball}}(z)\le V_{1,k}^{W,\mathrm{raw}}(z),
$$
即球化问题给出的是原始问题的保守值（下界）。

---

## 4. 球化单步问题的等价解

记
$$
\hat x_{k,i}^{\mathrm{pred}}(z):=\hat f_k(z)+\hat\omega_{k,i},
\qquad
\hat P_{x^+|z,k,N}^{\hat f_k}=\frac1N\sum_{i=1}^N\delta_{\hat x_{k,i}^{\mathrm{pred}}(z)}.
$$

### 假设 C（Theorem 6.3 精确适用）
1. $\mathcal X=\mathbb R^{d_x}$；  
2. Wasserstein 距离使用范数 $\|\cdot\|$（对偶为 $\|\cdot\|_*$）；  
3. 对每个候选 $\hat p_k\in\mathfrak F_k$，函数
   $$
   g_k(x):=-\log \hat p_k(x\mid z)
   $$
   为 proper、convex、lower-semicontinuous；  
4. 有限 steepness：
   $$
   \kappa_k(\hat p_k):=
   \sup\{\|\theta\|_*: g_k^*(\theta)<\infty\}<\infty,
   $$
   且 $\hat p_k>0$（在相关支撑上）以保证对数项有定义。

在假设 C 下，由 `Wasserstein.md` (Theorem 6.3, $\Xi=\mathbb R^{d_x}$)：
$$
\sup_{Q\in\bar{\mathcal I}_{k,N}^{W}(z)}\mathbb E_Q[g_k(X)]
=
\frac1N\sum_{i=1}^N g_k(\hat x_{k,i}^{\mathrm{pred}}(z))
+\rho_{k,N}\,\kappa_k(\hat p_k).
$$
等价地，
$$
\inf_{Q\in\bar{\mathcal I}_{k,N}^{W}(z)}\mathbb E_Q[\log \hat p_k(X\mid z)]
=
\frac1N\sum_{i=1}^N \log \hat p_k(\hat x_{k,i}^{\mathrm{pred}}(z)\mid z)
-\rho_{k,N}\,\kappa_k(\hat p_k).
$$

因此
$$
(\mathbf P_{1,k}^{W,\mathrm{ball}}(z))\equiv
\sup_{\hat p_k\in\mathfrak F_k}
\left\{
\frac1N\sum_{i=1}^N \log \hat p_k(\hat x_{k,i}^{\mathrm{pred}}(z)\mid z)
-\rho_{k,N}\,\kappa_k(\hat p_k)
\right\}.
$$

这就是“经验对数似然 + Wasserstein 正则”的严格等价形式（针对球化问题）。

---

## 5. 闭式特例：固定尺度 Laplace 族

取
$$
\hat p_{k,m}(x\mid z)=\frac1{(2b)^{d_x}}
\exp\!\left(-\frac{\|x-m\|_1}{b}\right),\qquad b>0\ \text{固定}.
$$

并令 Wasserstein 距离使用 $\|\cdot\|_1$（因此对偶范数为 $\|\cdot\|_\infty$）。则
$$
g_{k,m}(x)=d_x\log(2b)+\frac{\|x-m\|_1}{b},
\qquad
\kappa_k(\hat p_{k,m})=\frac1b.
$$

单步目标化为
$$
\max_{m\in\mathbb R^{d_x}}
\left[
-d_x\log(2b)
-\frac1{bN}\sum_{i=1}^N\|\hat x_{k,i}^{\mathrm{pred}}(z)-m\|_1
-\frac{\rho_{k,N}}b
\right],
$$
等价于
$$
m_k^*(z)\in
\arg\min_{m\in\mathbb R^{d_x}}
\frac1N\sum_{i=1}^N\|\hat x_{k,i}^{\mathrm{pred}}(z)-m\|_1.
$$

因此最优位置参数是 **$L_1$-median（坐标中位点）**；在 $d_x=1$ 时即样本中位数。  
对应最优预测器
$$
\hat p_k^*(x\mid z)=\hat p_{k,m_k^*(z)}(x\mid z),
$$
最优单步值
$$
V_{1,k}^{W,\mathrm{ball},*}(z)=
-d_x\log(2b)
-\frac1b
\left(
\min_m\frac1N\sum_{i=1}^N\|\hat x_{k,i}^{\mathrm{pred}}(z)-m\|_1
+\rho_{k,N}
\right).
$$

---

## 6. 结论性备注

1. 本文给出的“等价解”严格对应球化问题 $(\mathbf P_{1,k}^{W,\mathrm{ball}})$；对原始问题是保守近似。  
2. 若去掉 $\mathcal X=\mathbb R^{d_x}$ 等条件，Theorem 6.3 一般只给上界，等价性会退化为不等式界。  
3. 将第 4 节单步目标作为 stage reward 可嵌入第 2 节 Bellman 递推，得到多步保守求解框架。

