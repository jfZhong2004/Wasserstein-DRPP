# 单步 DRPP 上下界严格证明（修正版）

本文给出 `定义与上下界证明.md` 中定理 1、定理 2 的**严格修正版本**。  
核心原则：凡是用到 `Wassersteein.md` Theorem 4.2 的地方，都先把可用条件写清楚；若条件不满足，只给可保证成立的不等式版本，不做伪证。

---

## 0. 先说明原版本的关键风险

原文把如下对偶恒等式直接当作一般结论使用：
$$
\sup_{Q\in\mathbb B_\varepsilon(\hat P)}\mathbb E_Q[\ell]
=
\inf_{\lambda\ge0}\left\{\lambda\varepsilon+\frac1N\sum_{i=1}^N\sup_x\big(\ell(x)-\lambda\|x-\hat x_i\|\big)\right\}.
$$
在 `Wassersteein.md` 中，这个“等号版”依赖 Theorem 4.2 的前提（Assumption 4.1）。  
若前提未验证，则只有 Corollary 4.3 给出的单边界（保守不等式）可直接使用。

---

## 1. 统一记号与新增假设

### 1.1 单步问题（$\gamma_0\equiv0$ 的上界子问题）

固定时刻 $k$ 与当前状态控制对 $z$，定义名义预测锚点
$$
a_i:=\hat x_i^{pred}=\bar f_k(z)+\hat w_{k,i},\quad i=1,\dots,N,
$$
经验分布
$$
\hat P_{k,z}:=\frac1N\sum_{i=1}^N\delta_{a_i}.
$$
定义单步鲁棒值
$$
\bar V_k(z):=\sup_{p\in\mathcal P}\inf_{Q\in\mathbb B_\varepsilon(\hat P_{k,z})}\mathbb E_Q[\log p(X)].
$$

### 1.2 假设（严格版）

**A1（空间与密度）**：$\mathcal X\subseteq\mathbb R^{d_x}$ 为 Borel 集；存在参考密度 $\rho$（$\rho\ge0,\int\rho=1$）。  

**A2（可行预测类）**：$\mathcal P$ 包含所有满足 $p\ge0,\int_{\mathcal X}p=1$ 且 $\log p$ 可测、目标值有限的密度。  

**A3'（按 Assumption 4.1 前推的可验证假设）**：对任意 $p\in\mathcal P$，定义损失
$$
\ell_p(x):=-\log p(x),\quad x\in\mathcal X.
$$
并假设 $\mathcal X$ 凸且闭，且 $\ell_p$ 在 $\mathcal X$ 上可表示为
$$
\ell_p(x)=\max_{1\le j\le J_p}\ell_{p,j}(x),
$$
其中每个 $-\ell_{p,j}$ 都是 proper、convex、lower semicontinuous，且 $\ell_{p,j}$ 在 $\mathcal X$ 上不恒为 $-\infty$。  
即：对每个固定 $p$，`Wassersteein.md` 的 Assumption 4.1 可用于 $\ell_p$。

**A4（模型误差约束）**：$\|f_k(\xi)-\bar f_k(\xi)\|_2\le\sqrt{\gamma_0(\xi)}$，用于定理 2。

**Remark 1（旧版 A3 作为推论）**：在 A3' 下，由 `Wassersteein.md` Theorem 4.2（等价凸规约）可推出：对任意 $p\in\mathcal P$ 与任意锚点集合 $\{a_i\}_{i=1}^N$，
$$
\inf_{Q\in\mathbb B_\varepsilon(\hat P)}\mathbb E_Q[\log p]
=
\sup_{\lambda\ge0}\left\{-\lambda\varepsilon+\frac1N\sum_{i=1}^N\inf_{x\in\mathcal X}\big(\log p(x)+\lambda\|x-a_i\|\big)\right\}. \tag{D}
$$
若 A3' 无法验证，则只能使用 Corollary 4.3 的单边界版本，不能使用 (D) 的等号形式。

---

## 2. 修正定理 1（单步上界子问题的严格等价）

### 定理 1'

在 A1、A2、A3' 下，
$$
\bar V_k(z)
=
\max_{\lambda\ge0,\;s\in\mathbb R^N}
\left\{-\lambda\varepsilon+\frac1N\sum_{i=1}^N s_i\right\}
$$
$$
\text{s.t.}\quad
\int_{\mathcal X}\max_{1\le i\le N}\exp\!\big(s_i-\lambda\|x-a_i\|\big)\,dx\le1. \tag{P1}
$$
并且存在最优密度可写成
$$
p^*(x)=\max_{1\le i\le N}\exp\!\big(s_i^*-\lambda^*\|x-a_i\|\big).
$$

#### 证明

对固定 $p$，由 Remark 1 的 (D) 得
$$
\inf_{Q\in\mathbb B_\varepsilon(\hat P_{k,z})}\mathbb E_Q[\log p]
=
\sup_{\lambda\ge0}\left\{-\lambda\varepsilon+\frac1N\sum_{i=1}^N \inf_x\big(\log p(x)+\lambda\|x-a_i\|\big)\right\}. \tag{1}
$$
代回外层 $\sup_p$，并引入变量 $s_i$：
$$
s_i\le \inf_x\big(\log p(x)+\lambda\|x-a_i\|\big)
\iff
\log p(x)\ge s_i-\lambda\|x-a_i\|,\ \forall x.
$$
合并 $i$ 得
$$
p(x)\ge g_{\lambda,s}(x):=\max_i\exp\!\big(s_i-\lambda\|x-a_i\|\big). \tag{2}
$$
因此问题等价为
$$
\sup_{p,\lambda,s}\left\{-\lambda\varepsilon+\frac1N\sum_i s_i\right\}
\ \text{s.t.}\ p\ge g_{\lambda,s},\ \int p=1. \tag{3}
$$

对给定 $(\lambda,s)$，(3) 可行当且仅当
$$
\int_{\mathcal X}g_{\lambda,s}(x)\,dx\le1. \tag{4}
$$
必要性显然（$p\ge g$ 且 $\int p=1$）。充分性由构造
$$
p(x)=g_{\lambda,s}(x)+\Big(1-\int g_{\lambda,s}\Big)\rho(x)
$$
得到（A1）。

于是 (3) 等价为 (P1)。再证最优点处约束必绑定：若严格小于 1，则取任意
$$
0<\delta<-\log\!\int g_{\lambda,s},
$$
令 $s_i' = s_i+\delta$，则 $\int g_{\lambda,s'}=e^\delta\int g_{\lambda,s}\le1$，而目标增大 $\delta$，矛盾。故最优时 $\int g_{\lambda^*,s^*}=1$，可取 $p^*=g_{\lambda^*,s^*}$。证毕。$\square$

**推论 1（对严格耦合原问题的上界）**  
定义严格耦合单步值 $V_k^{strict}(z)$（见第 3 节）。由于 $\gamma_0\equiv0$ 时对抗者可行集是严格耦合集的子集，故在同一预测类 $\mathcal P$ 下有
$$
V_k^{strict}(z)\le \bar V_k(z).
$$

---

## 3. 修正定理 2（严格耦合问题的可计算下界）

我们只讨论**单步**，避免把 $V_{k+1}^*$ 的正则性混入当前证明。

定义严格耦合单步值
$$
V_k^{strict}(z):=
\sup_{p\in\mathcal P}
\inf_{\substack{f_k:\ \|f_k(\cdot)-\bar f_k(\cdot)\|_2\le\sqrt{\gamma_0(\cdot)}}}
\inf_{Q\in\mathbb B_\varepsilon(\hat P_{w,N}^{f_k})}
\mathbb E_Q[\log p(X)].
$$

名义锚点
$$
a_i:=\bar f_k(z)+\hat x_{k+1,i}-\bar f_k(\hat z_{k,i}),
$$
半径
$$
R_i:=\sqrt{\gamma_0(z)}+\sqrt{\gamma_0(\hat z_{k,i})}.
$$

### 定理 2'（严格下界）

在 A1、A2、A3'、A4 下，令
$$
V_k^{low}(z):=
\max_{\lambda\ge0,\;s\in\mathbb R^N}
\left\{-\lambda\varepsilon+\frac1N\sum_{i=1}^N s_i\right\}
$$
$$
\text{s.t.}\quad
\int_{\mathcal X}\max_i
\exp\!\Big(s_i-\lambda\,[\|x-a_i\|-R_i]_+\Big)\,dx\le1, \tag{P2}
$$
其中 $[t]_+:=\max(0,t)$。则
$$
V_k^{low}(z)\le V_k^{strict}(z). \tag{5}
$$
并且存在 (P2) 的最优密度表示
$$
p_{low}^*(x)=\max_i\exp\!\Big(s_i^*-\lambda^*[\|x-a_i\|-R_i]_+\Big).
$$

#### 证明

对固定 $p$ 与固定 $f_k$，由 Remark 1 的 (D)（锚点换为严格锚点）：
$$
\inf_{Q\in\mathbb B_\varepsilon(\hat P_{w,N}^{f_k})}\mathbb E_Q[\log p]
=
\sup_{\lambda\ge0}
\left\{-\lambda\varepsilon+\frac1N\sum_i\inf_x\big(\log p(x)+\lambda\|x-a_i-\Delta\nu+\Delta f_i\|\big)\right\}, \tag{6}
$$
其中 $\Delta\nu=f_k(z)-\bar f_k(z)$，$\Delta f_i=f_k(\hat z_{k,i})-\bar f_k(\hat z_{k,i})$。

**(a) 放松共享偏移（保守）**：把共享 $\Delta\nu$ 放松为独立 $\Delta\nu_i$，对抗者可行域变大，内层值不增，故得到不大于原值的量。  

**(b) 弱对偶（$\inf\sup\ge\sup\inf$）**：
$$
\inf_{\{\Delta\nu_i,\Delta f_i\}}\sup_{\lambda\ge0}G(\lambda,\Delta)
\ge
\sup_{\lambda\ge0}\inf_{\{\Delta\nu_i,\Delta f_i\}}G(\lambda,\Delta). \tag{7}
$$

令 $\delta_i:=\Delta\nu_i-\Delta f_i$，由 A4 得 $\delta_i\in\mathbb B_{R_i}$（Minkowski 和）。于是对每个 $i$、固定 $x$：
$$
\inf_{\delta_i\in\mathbb B_{R_i}}\|x-a_i-\delta_i\|
=[\|x-a_i\|-R_i]_+. \tag{8}
$$
故对每个固定 $\lambda$，
$$
\inf_{\delta_i\in\mathbb B_{R_i}}\inf_x\big(\log p(x)+\lambda\|x-a_i-\delta_i\|\big)
=
\inf_x\big(\log p(x)+\lambda[\|x-a_i\|-R_i]_+\big). \tag{9}
$$

由 (6)–(9) 得到对固定 $p$ 的可计算下界
$$
\underline\Phi(p):=
\sup_{\lambda\ge0}
\left\{-\lambda\varepsilon+\frac1N\sum_i\inf_x\big(\log p(x)+\lambda[\|x-a_i\|-R_i]_+\big)\right\}
\le
\text{严格内层值}. \tag{10}
$$
再对 $p$ 取上确界：
$$
\sup_{p\in\mathcal P}\underline\Phi(p)\le V_k^{strict}(z). \tag{11}
$$

接下来把左边化成有限维优化：与定理 1' 完全同型，只需把距离 $\|x-a_i\|$ 换成 $[\|x-a_i\|-R_i]_+$。同样可得等价程序即 (P2)，且最优处积分约束绑定，从而可取
$$
p_{low}^*(x)=\max_i\exp\!\Big(s_i^*-\lambda^*[\|x-a_i\|-R_i]_+\Big).
$$
于是 (11) 即 (5)。证毕。$\square$

---

## 4. 与原文结论关系（是否成立）

1. **若 A3' 可被严格验证**：  
   - 定理 1' 的“等价 + 指数核最优结构”成立；  
   - 定理 2' 的“可计算保守下界”成立。  

2. **若 A3' 不能验证**（这是常见风险）：  
   - 原文“定理 1 精确等价”不成立（不能直接声称）；  
   - 原文“定理 2 的等式化对偶步骤”也不能直接声称；  
   - 只能退回 `Wassersteein.md` Corollary 4.3 的保守不等式框架。

在 A3' 成立时，单步界链严格成立：
$$
V_k^{low}(z)\le V_k^{strict}(z)\le \bar V_k(z).
$$

这就是当前能给出的“无伪证”版本：成立就写清条件，不成立就明确降级为保守界。

