# 单步 WDRPP（复合"函数球 x 分布球"）上下界严格推导（修正版）

## 修正说明

本文档修正了原文档中的以下严重逻辑错误：

**原文错误：假设 (H3) 自相矛盾。** 原文要求 "$-\ell_p = \log p$ 是 proper、凸、下半连续函数"，即 $\log p$ 在 $\mathbb{R}^{d_x}$ 上为 proper 凸函数。然而：

1. **不存在满足此条件的概率密度。** 设 $h := \log p$ 是 $\mathbb{R}^{d_x}$ 上的 proper 凸函数。由凸分析基本定理，$h$ 在任意点 $x_0$ 存在次梯度 $g \in \partial h(x_0)$，满足 $h(x) \ge h(x_0) + g^\top(x - x_0)$。因此 $p(x) = e^{h(x)} \ge e^{h(x_0) + g^\top(x-x_0)}$。但右端在 $g$ 方向上指数增长，$\int_{\mathbb{R}^{d_x}} e^{h(x_0)+g^\top(x-x_0)}\,dx = +\infty$，故 $\int p = +\infty$，与 $p$ 为概率密度矛盾。这意味着原文的预测类 $\mathcal{P}$ 为空集，所有涉及 $\sup_{p \in \mathcal{P}}$ 的结论均为空真（vacuously true）。

2. **消去 $p$ 的充分性论证失效。** 原文定理 1 和定理 3 中构造 $p = g_{\lambda,s,m} + (1-\int g)\rho$。但 $\log g_{\lambda,s,m} = \max_i(s_i - \lambda\|x-c_i\|)$ 是有限个 concave 函数的逐点 max，为 concave 而非 convex。因此构造的 $p$ 不在原文的受限 $\mathcal{P}$ 中。

**修正方案：** 将 (H3) 中的条件替换为标准 Wasserstein DRO 对偶所需的正确条件：$\ell_p = -\log p$ **上半连续**（等价于 $p$ 下半连续）。这是 Mohajerin Esfahani & Kuhn (2018) Theorem 2.1 或 Blanchet & Murthy (2019) 等标准文献中的标准假设。在此修正下：
- $\mathcal{P}$ 包含所有连续正概率密度（如高斯、Laplace 等），非空；
- 消去 $p$ 的充分性论证成立（构造的密度满足正则性条件）；
- **所有定理的结论不变，证明逻辑链完整。**

---

## 0. 目标

在当前模糊集
$$
\mathcal A_k
:=
\left\{
(g,Q):
\|g-\hat f_k\|_{\infty,2}\le\gamma,
W_1\!\left(Q,\hat P_{\omega,k,N}^{\hat f_k}\right)\le\varepsilon
\right\}
$$
下，研究单步值函数
$$
V_k^{\mathrm{raw}}(z)
:=
\sup_{p\in\mathcal P}
\inf_{(g,Q)\in\mathcal A_k}
\mathbb E_{X\sim (\tau_{g(z)})_\#Q}\!\left[\log p(X)\right].
$$

要求：不使用单球化 $\rho=\gamma+\varepsilon$ 近似，直接在"函数球 x 分布球"结构上给出可计算且严格可证的上下界。

---

## 1. 假设与记号

### 1.1 记号
固定时刻 $k$ 与当前 $z\in\mathcal Z$，定义
$$
\hat P_{\omega,k,N}^{\hat f_k}:=\frac1N\sum_{i=1}^N\delta_{\hat\omega_{k,i}},
\qquad
a_i:=\hat f_k(z)+\hat\omega_{k,i},\quad i=1,\dots,N.
$$

### 1.2 假设

**(H1)** 状态空间为 $\mathcal X=\mathbb R^{d_x}$，且 $W_1$ 使用欧氏范数 $\|\cdot\|_2$。

**(H2)** 预测类
$$
\mathcal P=\left\{p:\mathbb R^{d_x}\to(0,\infty)\ \middle|\ \int p(x)\,dx=1,\ -\log p\ \text{上半连续}\right\}.
$$

> **注（与原文的区别）：** 原文 (H3) 要求 $\log p$ 为 proper 凸 lsc 函数。如修正说明所论，该条件与 $p$ 为概率密度不相容（$\mathcal{P}$ 为空集）。此处改为标准 Wasserstein DRO 对偶条件：$\ell_p := -\log p$ 上半连续（等价于 $p$ 下半连续且严格正）。此条件涵盖所有连续正密度函数，非空且自然。

**(H3)** 对任意 $p\in\mathcal P$，记 $\ell_p(x):=-\log p(x)$。显式要求：
1. $\mathcal X=\mathbb R^{d_x}$ 为闭凸集；
2. $\ell_p$ 为上半连续函数（usc），且 $\ell_p$ 不恒等于 $-\infty$。

> **关于 Wasserstein DRO 对偶的适用性：** 条件 (H3) 的目的是保证后文定理 A（Wasserstein 对偶表示）中的强对偶成立。标准结果（如 Mohajerin Esfahani & Kuhn, 2018, Theorem 2.1；Blanchet & Murthy, 2019）仅需损失函数 $\ell$ 上半连续，而不需要 $\ell$ 的凸性或 $-\ell$ 的凸性。原文引用 "Theorem 4.2" 的更强条件（凸性/凹性）是为了特定的有限维凸规划可处理性结果，但本文的推导（引入 $s_i$ 变量、消去 $p$）并不依赖该特殊结构，只需基本对偶等式即可。

---

## 2. 需要调用的外部定理（来自 Wasserstein.md）

### 定理 A（Wasserstein 球上的对偶表示）
设经验分布 $\hat P_N=\frac1N\sum_{i=1}^N\delta_{\hat\xi_i}$，半径 $\varepsilon\ge0$，损失函数 $\ell:\mathbb{R}^{d_x}\to\overline{\mathbb{R}}$ 上半连续且不恒为 $-\infty$。则
$$
\sup_{Q\in\mathbb B_\varepsilon(\hat P_N)}\mathbb E_Q[\ell(X)]
=
\inf_{\lambda\ge0}
\left\{
\lambda\varepsilon+\frac1N\sum_{i=1}^N
\sup_{x\in\mathbb R^{d_x}}\big(\ell(x)-\lambda\|x-\hat\xi_i\|_2\big)
\right\}.
$$

> 本文仅把它作为外部已知结论使用；后续所有不等式方向与规约步骤均在本文完整证明。
> **参考文献：** Mohajerin Esfahani & Kuhn (2018), Theorem 2.1; Blanchet & Murthy (2019); Gao & Kleywegt (2022), Theorem 1。
> **适用条件验证：** 在本文中，$\ell = \ell_p = -\log p$，由 (H3) 保证其上半连续；参考分布为有限支撑经验分布；$W_1$ 球非空（含参考分布本身）。故定理 A 适用。

---

## 3. 原问题的等价重写

先把 $(g,Q)$ 结构中的函数不确定性压缩为当前点位移变量。

### 引理 1（函数球到点位移球）
在固定 $z$ 下，
$$
\{g(z):\|g-\hat f_k\|_{\infty,2}\le\gamma\}
=
\{\hat f_k(z)+m:\|m\|_2\le\gamma\}.
$$

**证明：**
1. 若 $\|g-\hat f_k\|_{\infty,2}\le\gamma$，则 $\|g(z)-\hat f_k(z)\|_2\le\gamma$，令 $m=g(z)-\hat f_k(z)$ 即得一侧包含。
2. 反向：任意 $\|m\|_2\le\gamma$，构造 $g_m(\zeta)=\hat f_k(\zeta)+m$，则
$\|g_m-\hat f_k\|_{\infty,2}=\|m\|_2\le\gamma$，且 $g_m(z)=\hat f_k(z)+m$。故另一侧包含。证毕。

由引理 1，
$$
V_k^{\mathrm{raw}}(z)=
\sup_{p\in\mathcal P}
\inf_{\|m\|_2\le\gamma}
\inf_{Q\in\mathbb B_\varepsilon(\hat P_{\omega,k,N}^{\hat f_k})}
\mathbb E_{W\sim Q}\!\left[\log p\!\big(\hat f_k(z)+m+W\big)\right].
\tag{P-raw}
$$

记
$$
\hat P_{x,m}:=\frac1N\sum_{i=1}^N\delta_{a_i+m}.
$$
由于平移推前保持 Wasserstein 距离（$W_1((\tau_c)_\#P,(\tau_c)_\#Q)=W_1(P,Q)$），$(\mathrm{P\!-\!raw})$ 等价为
$$
V_k^{\mathrm{raw}}(z)=
\sup_{p\in\mathcal P}
\inf_{\|m\|_2\le\gamma}
\inf_{P\in\mathbb B_\varepsilon(\hat P_{x,m})}
\mathbb E_P[\log p(X)].
\tag{P0}
$$

**平移不变性的验证：** 令 $c := \hat f_k(z)+m$。映射 $Q \mapsto P := (\tau_c)_\# Q$ 在 $\mathbb{B}_\varepsilon(\hat P_{\omega,k,N}^{\hat f_k})$ 与 $\mathbb{B}_\varepsilon(\hat P_{x,m})$ 之间建立双射，因为：
- $(\tau_c)_\# \hat P_{\omega,k,N}^{\hat f_k} = \frac{1}{N}\sum_i \delta_{\hat\omega_{k,i}+c} = \frac{1}{N}\sum_i \delta_{a_i+m} = \hat P_{x,m}$；
- $W_1(P, \hat P_{x,m}) = W_1((\tau_c)_\#Q, (\tau_c)_\#\hat P_\omega) = W_1(Q, \hat P_\omega) \le \varepsilon$。

且 $\mathbb{E}_{W\sim Q}[\log p(c+W)] = \mathbb{E}_{X\sim P}[\log p(X)]$。故 (P-raw) 与 (P0) 等价。

---

## 4. 固定 $m$ 的精确有限维规约

定义
$$
\Phi(p,m):=
\inf_{P\in\mathbb B_\varepsilon(\hat P_{x,m})}\mathbb E_P[\log p(X)],
\qquad
U(m):=\sup_{p\in\mathcal P}\Phi(p,m).
$$

### 定理 1（固定 $m$ 的精确规约）
对任意固定 $m$，有
$$
U(m)=
\sup_{\lambda\ge0,\ s\in\mathbb R^N}
\left\{
-\lambda\varepsilon+\frac1N\sum_{i=1}^N s_i
\right\}
$$
$$
\text{s.t.}\quad
\int_{\mathbb R^{d_x}}
\max_{1\le i\le N}\exp\!\big(s_i-\lambda\|x-a_i-m\|_2\big)\,dx
\le1.
\tag{U-m}
$$

**证明：**

**第一步（对偶化内层 Wasserstein inf）。** 对固定 $p,m$，令 $\ell_p=-\log p$。由 (H3)，$\ell_p$ 上半连续。由定理 A：
$$
\sup_{P\in\mathbb B_\varepsilon(\hat P_{x,m})}\mathbb E_P[\ell_p(X)]
=
\inf_{\lambda\ge0}
\left\{
\lambda\varepsilon+\frac1N\sum_{i=1}^N
\sup_x\big(\ell_p(x)-\lambda\|x-a_i-m\|_2\big)
\right\}.
$$
两边乘以 $-1$（$\inf$ 变 $\sup$，$\sup$ 变 $\inf$），得到
$$
\Phi(p,m)=
\sup_{\lambda\ge0}
\left\{
-\lambda\varepsilon+\frac1N\sum_{i=1}^N
\inf_x\big(\log p(x)+\lambda\|x-a_i-m\|_2\big)
\right\}.
\tag{1}
$$

**第二步（引入 $s_i$ 变量）。** 由 $\sup_p \sup_\lambda = \sup_{p,\lambda}$，
$$
U(m)=
\sup_{p,\lambda}
\left\{
-\lambda\varepsilon+\frac1N\sum_i
\inf_x\big(\log p(x)+\lambda\|x-a_i-m\|_2\big)
\right\}.
$$
引入辅助变量 $s_i$：
$$
s_i\le \inf_x\big(\log p(x)+\lambda\|x-a_i-m\|_2\big)
\iff
\log p(x)\ge s_i-\lambda\|x-a_i-m\|_2,\ \forall x.
$$
对所有 $i$ 同时成立等价于
$$
p(x)\ge g_{\lambda,s,m}(x):=
\max_i\exp\!\big(s_i-\lambda\|x-a_i-m\|_2\big),\ \forall x.
\tag{2}
$$

由于目标函数 $-\lambda\varepsilon + \frac{1}{N}\sum_i s_i$ 关于 $s_i$ 递增，在最优处 $s_i$ 取到约束上界，故引入 $s_i$ 不改变最优值：
$$
U(m)=
\sup_{p,\lambda,s}
\left\{-\lambda\varepsilon+\frac1N\sum_i s_i\right\}
\ \text{s.t.}\ p\ge g_{\lambda,s,m},\ p\in\mathcal P.
\tag{3}
$$

**第三步（消去 $p$）。** 约束"$\exists\, p\in\mathcal P:\ p\ge g_{\lambda,s,m}$"等价于"$\int g_{\lambda,s,m}\le 1$"。证明如下：

- **必要性：** 若 $p\ge g_{\lambda,s,m}$ 且 $\int p=1$，则 $\int g_{\lambda,s,m}\le\int p=1$。

- **充分性：** 若 $\int g_{\lambda,s,m}\le 1$，构造
$$
p(x)=g_{\lambda,s,m}(x)+\Big(1-\int g_{\lambda,s,m}\Big)\rho(x),
$$
其中 $\rho$ 为任意固定的连续正概率密度（例如标准高斯密度 $\rho(x)=(2\pi)^{-d_x/2}e^{-\|x\|^2/2}$）。则：
  - $p(x) \ge g_{\lambda,s,m}(x)$（因第二项非负）；
  - $\int p = \int g + (1-\int g)\int\rho = \int g + 1 - \int g = 1$；
  - $p(x) > 0$（因 $\rho(x) > 0$ 且系数 $\ge 0$，而 $g \ge 0$）；
  - $p$ 连续：$g_{\lambda,s,m}$ 为有限个连续函数的逐点 max，故连续；$\rho$ 连续；故 $p$ 连续。从而 $-\log p$ 连续（特别地上半连续），满足 (H3)。

> **关键修正点：** 原文在此步的充分性论证中未验证构造的 $p$ 是否满足假设条件。在原文的错误假设 (H3)（$\log p$ 凸）下，构造的 $p$ 无法满足条件。在修正后的 (H3)（$\ell_p$ 上半连续）下，上述验证表明构造的 $p \in \mathcal{P}$，充分性成立。

故 (3) 与 $(\mathrm{U\!-\!m})$ 等价。证毕。

---

## 5. 原始值的严格上界

### 定理 2（上界）
$$
V_k^{\mathrm{raw}}(z)\le \inf_{\|m\|_2\le\gamma}U(m)=:V_k^{\mathrm{up}}(z).
\tag{4}
$$
特别地，对任意可行 $m_0$，
$$
V_k^{\mathrm{raw}}(z)\le U(m_0),
\quad\text{尤其 }V_k^{\mathrm{raw}}(z)\le U(0).
\tag{5}
$$

**证明：**
由 (P0) 与 $U(m)=\sup_p\Phi(p,m)$，
$$
V_k^{\mathrm{raw}}(z)
=
\sup_p\inf_{\|m\|\le\gamma}\Phi(p,m).
$$
应用 minimax 不等式（对任意函数 $F$，$\sup_x\inf_y F(x,y)\le \inf_y\sup_x F(x,y)$），得
$$
\sup_p\inf_{\|m\|\le\gamma}\Phi(p,m)\le \inf_{\|m\|\le\gamma}\sup_p\Phi(p,m) = \inf_{\|m\|\le\gamma}U(m).
$$
这给出 (4)。再由 $\inf_{\|m\|\le\gamma}U(m)\le U(m_0)$ 得 (5)。证毕。

> **注：** minimax 不等式 $\sup\inf \le \inf\sup$ 对任意函数恒成立，无需凸凹性或紧致性条件。

---

## 6. 原始值的严格下界

为给出可计算下界，先证明一个几何引理。

### 引理 2（到欧氏球的距离）
对任意 $u\in\mathbb R^{d_x}$、$\gamma\ge0$，
$$
\inf_{\|m\|_2\le\gamma}\|u-m\|_2
=
\big[\|u\|_2-\gamma\big]_+.
\tag{6}
$$

**证明：**

下界：由三角不等式 $\|u\|_2\le \|u-m\|_2+\|m\|_2$，故
$\|u-m\|_2\ge \|u\|_2-\|m\|_2\ge\|u\|_2-\gamma$，且显然 $\|u-m\|_2\ge0$，于是
$\|u-m\|_2\ge[\|u\|_2-\gamma]_+$。

上界可达：
1. 若 $\|u\|_2\le\gamma$，取 $m=u$（满足 $\|m\|_2=\|u\|_2\le\gamma$），值为 0。
2. 若 $\|u\|_2>\gamma$，取 $m=\gamma\,u/\|u\|_2$（满足 $\|m\|_2=\gamma$），则
$\|u-m\|_2=\|u\|_2(1-\gamma/\|u\|_2)=\|u\|_2-\gamma$。

两种情形合并即得 (6)。证毕。

---

设
$$
F(p,m,\lambda):=
-\lambda\varepsilon+\frac1N\sum_{i=1}^N
\inf_x\big(\log p(x)+\lambda\|x-a_i-m\|_2\big).
$$
由 (1) 与 (P0)：
$$
V_k^{\mathrm{raw}}(z)=
\sup_{p\in\mathcal P}
\inf_{\|m\|_2\le\gamma}
\sup_{\lambda\ge0}F(p,m,\lambda).
\tag{7}
$$

### 定理 3（严格下界）
定义
$$
V_k^{\mathrm{low}}(z):=
\sup_{\lambda\ge0,\ s\in\mathbb R^N}
\left\{
-\lambda\varepsilon+\frac1N\sum_{i=1}^N s_i
\right\}
$$
$$
\text{s.t.}\quad
\int_{\mathbb R^{d_x}}
\max_{1\le i\le N}
\exp\!\Big(s_i-\lambda[\|x-a_i\|_2-\gamma]_+\Big)\,dx
\le1.
\tag{L}
$$
则
$$
V_k^{\mathrm{low}}(z)\le V_k^{\mathrm{raw}}(z).
\tag{8}
$$

**证明：**

**第一步（minimax 弱对偶）。** 对每个固定的 $p\in\mathcal P$，由 minimax 不等式：
$$
\inf_{\|m\|_2\le\gamma}\sup_{\lambda\ge0}F(p,m,\lambda)
\ge
\sup_{\lambda\ge0}\inf_{\|m\|_2\le\gamma}F(p,m,\lambda).
$$
代回 (7)，取外层 $\sup_p$：
$$
V_k^{\mathrm{raw}}(z)\ge
\sup_p\sup_{\lambda\ge0}
\inf_{\|m\|_2\le\gamma}
\left\{
-\lambda\varepsilon+\frac1N\sum_i\phi_i(m)
\right\},
$$
其中
$$
\phi_i(m):=\inf_x(\log p(x)+\lambda\|x-a_i-m\|_2).
$$

> **弱对偶方向验证：** 对固定 $p$，函数 $m\mapsto \sup_\lambda F(p,m,\lambda)$ 的 $\inf_m$ 不小于 $\sup_\lambda$ 的 $\inf_m F$。这是因为 $\inf_m \sup_\lambda \ge \sup_\lambda \inf_m$ 对任意函数恒成立。故此步放松了原问题，给出下界。

**第二步（共享 $m$ 放松为独立 $m_i$）。** 设
$$
\mathcal S:=\{(m_1,\dots,m_N):m_1=\cdots=m_N,\ \|m_i\|_2\le\gamma\},
\quad
\mathcal T:=\{(m_1,\dots,m_N):\|m_i\|_2\le\gamma,\forall i\}.
$$
有 $\mathcal S\subseteq\mathcal T$（共享约束严格强于独立约束），因此
$$
\inf_{(m_i)\in\mathcal S}\frac1N\sum_i\phi_i(m_i)
\ge
\inf_{(m_i)\in\mathcal T}\frac1N\sum_i\phi_i(m_i)
=
\frac1N\sum_i\inf_{\|m_i\|_2\le\gamma}\phi_i(m_i).
$$

> **不等式方向验证：** 在更大集合 $\mathcal{T}$ 上取 inf 只可能更小（或相等），故 $\inf_{\mathcal{S}} \ge \inf_{\mathcal{T}}$。等号右端可分离是因为约束和求和项在独立 $m_i$ 下可逐项优化。

故
$$
V_k^{\mathrm{raw}}(z)\ge
\sup_p\sup_{\lambda\ge0}
\left\{
-\lambda\varepsilon+\frac1N\sum_i
\inf_{\|m_i\|_2\le\gamma}\inf_x
\big(\log p(x)+\lambda\|x-a_i-m_i\|_2\big)
\right\}.
\tag{9}
$$

**第三步（计算内层距离）。** 由 $\inf_{m_i}\inf_x=\inf_x\inf_{m_i}$（两个无约束/紧约束 inf 可交换顺序）。对固定 $x$ 和 $\lambda\ge 0$：
$$
\inf_{\|m_i\|_2\le\gamma}\big(\log p(x)+\lambda\|x-a_i-m_i\|_2\big)
=
\log p(x)+\lambda\inf_{\|m_i\|_2\le\gamma}\|x-a_i-m_i\|_2.
$$
由引理 2（令 $u = x - a_i$）：
$$
\inf_{\|m_i\|_2\le\gamma}\|x-a_i-m_i\|_2
=
[\|x-a_i\|_2-\gamma]_+.
$$
于是 (9) 变为
$$
V_k^{\mathrm{raw}}(z)\ge
\sup_{p,\lambda\ge0}
\left\{
-\lambda\varepsilon+\frac1N\sum_i
\inf_x\big(\log p(x)+\lambda r_i(x)\big)
\right\},
\tag{10}
$$
其中 $r_i(x):=[\|x-a_i\|_2-\gamma]_+$。

**第四步（显式消去 $p$）。** 对 (10) 的右侧引入变量 $s_i$，得到等价形式
$$
\sup_{p,\lambda,s}
\left\{
-\lambda\varepsilon+\frac1N\sum_i s_i
\right\}
$$
$$
\text{s.t.}\quad
s_i\le\inf_x(\log p(x)+\lambda r_i(x)),\ \forall i.
$$
该约束等价于
$$
p(x)\ge \max_i\exp(s_i-\lambda r_i(x)) =: g_{\lambda,s}(x),\quad \forall x,
$$
其中 $r_i(x)=[\|x-a_i\|_2-\gamma]_+$。

与定理 1 中同样的可行性判据：

- **必要性：** 若存在 $p\ge g_{\lambda,s}$ 且 $\int p=1$，则 $\int g_{\lambda,s}\le1$。
- **充分性：** 若 $\int g_{\lambda,s}\le1$，取
$$
p(x)=g_{\lambda,s}(x)+\big(1-\int g_{\lambda,s}\big)\rho(x),
$$
其中 $\rho$ 为连续正概率密度。则 $p > 0$，$p$ 连续，$\int p = 1$，$p\ge g_{\lambda,s}$。特别地 $-\log p$ 连续故上半连续，满足 (H3)，从而 $p\in\mathcal P$。

> **充分性中 $g_{\lambda,s}$ 的正则性验证：** $r_i(x) = [\|x-a_i\|_2 - \gamma]_+$ 为连续函数（$[\cdot]_+$ 和 $\|\cdot\|_2$ 的复合）。$\exp(s_i - \lambda r_i(x))$ 为连续函数。有限个连续函数的逐点 max 仍连续。故 $g_{\lambda,s}$ 连续。
> **$g_{\lambda,s}$ 的可积性：** 当 $\lambda > 0$ 时，$\|x\| \to \infty$ 时 $r_i(x) \to \infty$，故每个 $\exp(s_i - \lambda r_i(x)) \to 0$，且衰减速度为 $O(e^{-\lambda\|x\|})$。这在 $\mathbb{R}^{d_x}$ 上可积。当 $\lambda = 0$ 时，$g = \max_i e^{s_i}$ 为常数，$\int g = +\infty$（在 $\mathbb{R}^{d_x}$ 上），因此约束 $\int g \le 1$ 不被满足。故最优解必有 $\lambda > 0$。

因此消去 $p$ 后恰得到程序 (L)。

综上，(L) 的最优值等于 (10) 右端的值（两者相等，因消去 $p$ 是等价变换），而 (10) 右端不超过 $V_k^{\mathrm{raw}}(z)$（因前三步均为放松）。故 $V_k^{\mathrm{low}}(z) \le V_k^{\mathrm{raw}}(z)$，即 (8) 成立。证毕。

---

## 7. 最终严格界链

由定理 2 与定理 3：
$$
V_k^{\mathrm{low}}(z)
\le
V_k^{\mathrm{raw}}(z)
\le
V_k^{\mathrm{up}}(z)
:=
\inf_{\|m\|_2\le\gamma}U(m)
\le
U(0).
$$

这给出了不经单球化近似时，单步 WDRPP 原始值的严格可计算上下包络。

### 关于上下界间隙的分析

下界 $V_k^{\mathrm{low}}$ 与上界 $V_k^{\mathrm{up}}$ 之间的间隙来自两处放松：

1. **minimax 弱对偶间隙**（定理 3 第一步）：$\inf_m \sup_\lambda F \ge \sup_\lambda \inf_m F$。如果 $F(p,m,\lambda)$ 关于 $(m,\lambda)$ 满足 Sion minimax 定理的条件（$m$ 的可行域紧凸，$F$ 关于 $m$ 凸、关于 $\lambda$ 凹），则此间隙为零。**但本文的 $F$ 关于 $m$ 不一定是凸的**（$\inf_x(\log p(x) + \lambda\|x - a_i - m\|)$ 关于 $m$ 的凸性取决于 $p$），故一般存在间隙。

2. **共享到独立 $m$ 的放松间隙**（定理 3 第二步）：$\inf_{\text{shared }m}\sum\phi_i(m) \ge \sum\inf_{m_i}\phi_i(m_i)$。当不同 $\phi_i$ 在不同 $m$ 处取到最小值时，此间隙非零。

**特殊情形 $\gamma = 0$（无函数不确定性）：** 此时 $m = 0$ 被强制，两处放松均退化为等式：
- 第一步：$\inf_m$ 只有 $m=0$ 一个可行点，故 $\inf_m\sup_\lambda = \sup_\lambda\inf_m$。
- 第二步：共享与独立相同（只有 $m_i=0$）。

因此 $\gamma=0$ 时 $V_k^{\mathrm{low}} = V_k^{\mathrm{raw}} = V_k^{\mathrm{up}} = U(0)$，上下界精确闭合。

---

## 8. 推论：最优值可达时的最优分布恢复

### 推论 4（上界最优值处可恢复最优预测分布；并存在对应内层最坏分布）
设
$$
m^{\mathrm{up}}\in\arg\min_{\|m\|_2\le\gamma}U(m),
$$
且程序 $(\mathrm{U\!-\!}m^{\mathrm{up}})$ 的最优解存在，记为 $(\lambda^{\mathrm{up}},s^{\mathrm{up}})$。定义
$$
g^{\mathrm{up}}(x):=
\max_{1\le i\le N}\exp\!\big(s_i^{\mathrm{up}}-\lambda^{\mathrm{up}}\|x-a_i-m^{\mathrm{up}}\|_2\big).
$$
则
$$
\int_{\mathbb R^{d_x}}g^{\mathrm{up}}(x)\,dx=1,\qquad
p^{\mathrm{up},*}(x):=g^{\mathrm{up}}(x)\in\mathcal P,
$$
并且
$$
V_k^{\mathrm{up}}(z)=U(m^{\mathrm{up}})=\Phi\!\big(p^{\mathrm{up},*},m^{\mathrm{up}}\big).
$$
进一步，存在
$$
P^{\mathrm{up},*}\in\mathbb B_\varepsilon(\hat P_{x,m^{\mathrm{up}}})
$$
使
$$
\Phi\!\big(p^{\mathrm{up},*},m^{\mathrm{up}}\big)
=
\mathbb E_{P^{\mathrm{up},*}}\!\left[\log p^{\mathrm{up},*}(X)\right].
$$

**证明：**

1. **积分约束绑定。** 若 $\int g^{\mathrm{up}}<1$，取
$$
0<\delta<-\log\!\int g^{\mathrm{up}}
$$
并令 $s_i'=s_i^{\mathrm{up}}+\delta$。则
$$
\int \max_i e^{s_i'-\lambda^{\mathrm{up}}\|x-a_i-m^{\mathrm{up}}\|_2}
=
e^\delta\int g^{\mathrm{up}}\le1
$$
仍可行，而目标值增加 $\delta$，与 $(\lambda^{\mathrm{up}}, s^{\mathrm{up}})$ 的最优性矛盾。故 $\int g^{\mathrm{up}}=1$。

2. **$p^{\mathrm{up},*} \in \mathcal{P}$ 的验证。** 由 $\int g^{\mathrm{up}} = 1$ 且 $g^{\mathrm{up}} > 0$（指数函数恒正），$g^{\mathrm{up}}$ 为概率密度。又 $g^{\mathrm{up}}$ 连续（有限个连续函数的逐点 max），故 $-\log g^{\mathrm{up}}$ 连续，特别地上半连续，满足 (H3)。因此 $p^{\mathrm{up},*} = g^{\mathrm{up}} \in \mathcal{P}$。

3. **$p^{\mathrm{up},*}$ 达到 $U(m^{\mathrm{up}})$。** 对任意 $i,x$，
$$
\log p^{\mathrm{up},*}(x)
=\log\max_j\exp(s_j^{\mathrm{up}}-\lambda^{\mathrm{up}}\|x-a_j-m^{\mathrm{up}}\|_2)
\ge
s_i^{\mathrm{up}}-\lambda^{\mathrm{up}}\|x-a_i-m^{\mathrm{up}}\|_2,
$$
从而
$$
s_i^{\mathrm{up}}
\le
\inf_x\!\Big(\log p^{\mathrm{up},*}(x)+\lambda^{\mathrm{up}}\|x-a_i-m^{\mathrm{up}}\|_2\Big).
$$
故 $(p^{\mathrm{up},*},\lambda^{\mathrm{up}},s^{\mathrm{up}})$ 可行于定理 1 证明中的程序 (3)，其目标值等于 $(\mathrm{U\!-\!}m^{\mathrm{up}})$ 最优值 $U(m^{\mathrm{up}})$。由程序等价性，得
$$
U(m^{\mathrm{up}})=\Phi(p^{\mathrm{up},*},m^{\mathrm{up}}).
$$

4. **内层最坏分布可达。** 记 $\ell^{\mathrm{up}}(x):=-\log p^{\mathrm{up},*}(x)$。由 $p^{\mathrm{up},*} \in \mathcal{P}$ 知 $\ell^{\mathrm{up}}$ 上半连续。又 $\ell^{\mathrm{up}}(x) = -\log(\max_i e^{s_i - \lambda\|x-c_i\|})$ 在 $\|x\|\to\infty$ 时线性增长（增长率为 $\lambda^{\mathrm{up}}$），故 $\ell^{\mathrm{up}}(x) - \lambda^{\mathrm{up}}\|x-c_i\|$ 有界，确保 Wasserstein 对偶的最优值有限。由标准 Wasserstein DRO 中最坏分布的存在性结果（$W_1$ 球在弱收敛拓扑下紧，usc 函数的期望在弱收敛下取到上确界），存在 $P^{\mathrm{up},*} \in \mathbb{B}_\varepsilon(\hat P_{x,m^{\mathrm{up}}})$ 使
$$
\Phi(p^{\mathrm{up},*},m^{\mathrm{up}}) = \mathbb{E}_{P^{\mathrm{up},*}}[\log p^{\mathrm{up},*}(X)].
$$
证毕。

### 推论 5（下界最优值处仅能恢复松弛问题的最优预测分布）
设程序 (L) 的最优解存在，记为 $(\lambda^{\mathrm{low}},s^{\mathrm{low}})$。定义
$$
g^{\mathrm{low}}(x):=
\max_{1\le i\le N}\exp\!\Big(s_i^{\mathrm{low}}-\lambda^{\mathrm{low}}[\|x-a_i\|_2-\gamma]_+\Big).
$$
则
$$
\int_{\mathbb R^{d_x}}g^{\mathrm{low}}(x)\,dx=1,\qquad
p^{\mathrm{low},*}(x):=g^{\mathrm{low}}(x)\in\mathcal P,
$$
且 $p^{\mathrm{low},*}$ 是定理 3 第四步中等价松弛问题的最优预测分布。

但一般不能仅由 $(\lambda^{\mathrm{low}},s^{\mathrm{low}})$ 推出原始问题 $V_k^{\mathrm{raw}}(z)$ 的最优对抗分布（也不能保证恢复其最优共享位移 $m^*$）。

**证明：**

1. 与推论 4 同理，若 $\int g^{\mathrm{low}}<1$ 可统一抬升 $s_i$ 提高目标，故最优时必有 $\int g^{\mathrm{low}}=1$。$g^{\mathrm{low}}$ 连续且严格正，故 $p^{\mathrm{low},*} = g^{\mathrm{low}} \in \mathcal{P}$。

2. 定理 3 给出 $V_k^{\mathrm{low}}(z)\le V_k^{\mathrm{raw}}(z)$。下界是通过两步放松（minimax 交换 + 共享到独立 $m$）得到的。若出现严格不等式
$$
V_k^{\mathrm{low}}(z)<V_k^{\mathrm{raw}}(z),
$$
则 (L) 的最优解对应的是松弛问题的最优点，不对应原始问题的最优解。故在未额外证明间隙闭合之前，不能从下界最优解推出原始问题的最优对抗分布。证毕。

---

## 9. 计算建议

1. 解一次下界凸程序 (L)，得到 $V_k^{\mathrm{low}}(z)$。
2. 对若干 $m\in\{ \|m\|_2\le\gamma\}$ 解 (U-m)，取最小值得到上界近似。
3. 报告区间 $[V_k^{\mathrm{low}}(z),\,V_k^{\mathrm{up}}(z)]$ 作为原始单步值证书。
4. 当 $\gamma$ 较小时，间隙预期较小；$\gamma=0$ 时间隙精确闭合（见第 7 节分析）。

---

## 附录：修正版逻辑自检

以下逐项验证修正后的证明链完整性。

### A.1 假设一致性检查

| 假设 | 内容 | 自洽性 |
|------|------|--------|
| (H1) | $W_1$ 用欧氏范数 | 与定理 A 一致 |
| (H2) | $\mathcal{P}$ 为连续正密度 | 非空（如高斯、Laplace） |
| (H3) | $\ell_p = -\log p$ usc | 对 $\mathcal{P}$ 中所有 $p$ 成立（连续函数 usc） |

**(H2) 与 (H3) 的相容性：** (H2) 中 $p > 0$ 连续，故 $-\log p$ 连续（特别地 usc）。$\mathcal{P}$ 非空。

### A.2 定理 A 适用性检查

| 条件 | 验证 |
|------|------|
| 参考分布有限支撑 | $\hat P_{x,m} = \frac{1}{N}\sum\delta_{a_i+m}$，$N$ 有限 |
| $\ell_p$ 上半连续 | 由 (H3) |
| $W_1$ 球非空 | $\hat P_{x,m} \in \mathbb{B}_\varepsilon(\hat P_{x,m})$ |

### A.3 各定理的不等式方向检查

| 步骤 | 不等式 | 方向 | 理由 |
|------|--------|------|------|
| 定理 1 | 等式 | $=$ | Wasserstein 强对偶 + 等价消去 $p$ |
| 定理 2 | $\sup\inf \le \inf\sup$ | 上界 | minimax 不等式 |
| 定理 3 步骤 1 | $\inf_m\sup_\lambda \ge \sup_\lambda\inf_m$ | 下界 | minimax 不等式 |
| 定理 3 步骤 2 | $\inf_{\mathcal{S}} \ge \inf_{\mathcal{T}}$ | 下界 | $\mathcal{S}\subseteq\mathcal{T}$ |
| 定理 3 步骤 3 | $=$ | 等式 | Lemma 2 精确计算 |
| 定理 3 步骤 4 | $=$ | 等式 | 等价消去 $p$ |

### A.4 消去 $p$ 的充分性检查

| 项目 | 原文 | 修正版 |
|------|------|--------|
| 构造 | $p = g + (1-\int g)\rho$ | 相同 |
| $p > 0$ | 成立（$\rho > 0$） | 成立 |
| $\int p = 1$ | 成立 | 成立 |
| $\log p$ 凸？ | **不成立**（原文要求） | 不要求 |
| $-\log p$ usc？ | 未检查 | 成立（$p$ 连续正） |
| $p \in \mathcal{P}$？ | **不成立** | **成立** |

### A.5 $\gamma=0$ 退化一致性

$\gamma=0$ 时：
- 上界：$V_k^{\mathrm{up}} = U(0)$（$m=0$ 唯一可行）
- 下界：$[\|x-a_i\|-0]_+ = \|x-a_i\|$，(L) 与 (U-0) 完全相同
- 结论：$V_k^{\mathrm{low}} = V_k^{\mathrm{up}} = U(0)$，精确闭合。
