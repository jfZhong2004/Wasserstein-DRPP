# 基于 Wasserstein 模糊集的分布鲁棒概率预测：单步上下界严格证明

## 0. 文档目标

本文完整给出单步 DRPP（Distributionally Robust Probabilistic Prediction）在 Wasserstein 模糊集下的建模、半径选取、以及两个核心结果的严格证明：

1. **定理1（上界问题的精确有限维化）**：在 $\gamma_0\equiv0$（名义模型精确）时，单步 DRPP 与一个有限维凸优化问题严格等价。  
2. **定理2（严格耦合问题的可计算下界）**：在模型误差有界的严格耦合设定下，经过保守放松得到可计算凸优化下界，并严格证明其下界性质。

文中会显式写出 Wasserstein 球半径 $\varepsilon_N(\beta)$ 的来源（基于测度集中定理）。

---

## 1. 系统、数据与单步对象

考虑离散时间随机系统
$$
x_{k+1}=f_k(z_k)+w_k,\qquad z_k=(x_k,u_k).
$$

- $f_k$：真实（未知）动力学映射；
- $\bar f_k$：名义映射；
- $w_k$：噪声；
- $\gamma_0(\cdot)$：模型误差上界函数。

给定历史数据集（样本量 $N$）
$$
\mathcal D_N:=\{(\hat x_{k+1,i},\hat z_{k,i})\}_{i=1}^N.
$$

本文固定一个时刻 $k$ 和当前状态控制对 $z$，研究单步预测分布优化。

---

## 2. Wasserstein 半径与模糊集完整定义

### 2.1 半径：由置信度与样本量决定

先给出 `Wasserstein.md`（对应文献中 Theorem 3.4）使用的轻尾假设：

**轻尾假设（Assumption 3.3）**：存在 $a>1$ 使得
$$
A:=\mathbb E^{\mathbb P}\!\left[\exp\!\big(\|\xi\|^a\big)\right]<\infty.
$$

在该假设下，测度集中结果（Theorem 3.4）给出：存在仅依赖于 $a,A,m$ 的常数 $c_1,c_2>0$，使得
$$
\mathbb P^N\!\left\{d_W\!\left(\mathbb P,\widehat{\mathbb P}_N\right)\ge \varepsilon\right\}
\le
\begin{cases}
c_1\exp\!\left(-c_2N\varepsilon^{\max\{m,2\}}\right), & \varepsilon\le1,\\[1mm]
c_1\exp\!\left(-c_2N\varepsilon^{a}\right), & \varepsilon>1.
\end{cases}
$$

令右侧等于 $\beta\in(0,1)$，得到置信半径
$$
\varepsilon_N(\beta)=
\begin{cases}
\left(\dfrac{\log(c_1\beta^{-1})}{c_2N}\right)^{\!1/\max\{m,2\}},
& N\ge \dfrac{\log(c_1\beta^{-1})}{c_2},\\[3mm]
\left(\dfrac{\log(c_1\beta^{-1})}{c_2N}\right)^{\!1/a},
& N< \dfrac{\log(c_1\beta^{-1})}{c_2}.
\end{cases}
$$

因此本文中的 Wasserstein 球半径明确依赖于 **样本量 $N=|\mathcal D_N|$** 和 **置信度 $\beta$**。

---

### 2.2 严格耦合模糊集（完整参数化）

固定 $z$，给定参数
$$
(\mathcal D_N,\beta,\bar f_k,\gamma_0),
$$
定义可行模型集合
$$
\mathfrak F_k:=\Big\{f:\|f(\xi)-\bar f_k(\xi)\|_2\le \sqrt{\gamma_0(\xi)},\ \forall \xi\Big\}.
$$

对任意 $f\in\mathfrak F_k$，定义经验噪声分布
$$
\hat P_{w,N}^{f}:=\frac1N\sum_{i=1}^N\delta_{\hat x_{k+1,i}-f(\hat z_{k,i})}.
$$

定义由该噪声生成的下一步状态条件分布集合
$$
\mathcal I_k^{W,\mathrm{strict}}\!\left(z;\mathcal D_N,\beta,\bar f_k,\gamma_0\right)
:=
\left\{
P_{x_{k+1}|z}:
\begin{array}{l}
\exists f\in\mathfrak F_k,\ \exists P_w\in\mathbb B_{\varepsilon_N(\beta)}(\hat P_{w,N}^{f}),\\
x_{k+1}=f(z)+w,\ w\sim P_w
\end{array}
\right\}.
$$

等价地（把噪声球平移到状态球），对
$$
a_i^{\mathrm{strict}}(f,z):=f(z)+\hat x_{k+1,i}-f(\hat z_{k,i}),\qquad
\hat P_{x,N}^{f,z}:=\frac1N\sum_{i=1}^N\delta_{a_i^{\mathrm{strict}}(f,z)},
$$
有
$$
\mathcal I_k^{W,\mathrm{strict}}(z;\cdot)
=
\left\{
Q:\ \exists f\in\mathfrak F_k,\ Q\in\mathbb B_{\varepsilon_N(\beta)}(\hat P_{x,N}^{f,z})
\right\}.
$$

---

### 2.3 单步值函数

定义严格耦合单步值
$$
V_k^{\mathrm{strict}}(z):=
\sup_{p\in\mathcal P}\ \inf_{f\in\mathfrak F_k}\ \inf_{Q\in\mathbb B_{\varepsilon_N(\beta)}(\hat P_{x,N}^{f,z})}
\mathbb E_Q[\log p(X)].
$$

当 $\gamma_0\equiv0$ 时，$f=\bar f_k$ 固定，名义锚点
$$
a_i:=\bar f_k(z)+\hat x_{k+1,i}-\bar f_k(\hat z_{k,i}),
\qquad
\hat P_{k,z}:=\frac1N\sum_{i=1}^N\delta_{a_i},
$$
并得到上界子问题
$$
\bar V_k(z):=
\sup_{p\in\mathcal P}\ \inf_{Q\in\mathbb B_{\varepsilon_N(\beta)}(\hat P_{k,z})}\mathbb E_Q[\log p(X)].
$$

---

## 3. 证明所需假设与基础定理

### 3.1 假设

**A1（空间）**：$\mathcal X\subset\mathbb R^{d_x}$ 非空、紧、Borel。  

**A2（锚点可行性）**：名义锚点与严格锚点均在 $\mathcal X$ 内。  

**A3（预测密度类）**：$\mathcal P$ 中每个 $p$ 满足
$$
p:\mathcal X\to(0,\infty)\ \text{可测},\quad
\int_{\mathcal X}p(x)\,dx=1,\quad
\log p\ \text{下半连续}.
$$

**A4（参考密度）**：存在连续函数 $\rho:\mathcal X\to(0,\infty)$，$\int_{\mathcal X}\rho=1$。  

**A5（模型误差）**：
$$
\|f(\xi)-\bar f_k(\xi)\|_2\le\sqrt{\gamma_0(\xi)},\quad \forall\xi.
$$

---

### 3.2 定理A（固定损失的 Wasserstein 强对偶）

这是 Gao–Kleywegt (2022) 强对偶结果在本文设定下的直接特例（$p=1$、经验分布中心）。

设 $\hat P=\frac1N\sum_{i=1}^N\delta_{a_i}$，$\phi:\mathcal X\to\mathbb R$ 上半连续，则
$$
\sup_{Q\in\mathbb B_\varepsilon(\hat P)}\mathbb E_Q[\phi(X)]
=
\inf_{\lambda\ge0}
\left\{
\lambda\varepsilon+\frac1N\sum_{i=1}^N\sup_{x\in\mathcal X}\big(\phi(x)-\lambda\|x-a_i\|_2\big)
\right\}.
$$

令 $\phi=-\log p$（由 A3，$\phi$ 上半连续），可得推论
$$
\inf_{Q\in\mathbb B_\varepsilon(\hat P)}\mathbb E_Q[\log p(X)]
=
\sup_{\lambda\ge0}
\left\{
-\lambda\varepsilon+\frac1N\sum_{i=1}^N\inf_{x\in\mathcal X}\big(\log p(x)+\lambda\|x-a_i\|_2\big)
\right\}.
$$

---

## 4. 定理1：$\gamma_0\equiv0$ 时的精确有限维化

### 4.1 定理陈述

在 A1–A4 下，
$$
\bar V_k(z)=
\sup_{\lambda\ge0,\ s\in\mathbb R^N}
\left\{
-\lambda\varepsilon_N(\beta)+\frac1N\sum_{i=1}^Ns_i
\right\}
$$
$$
\text{s.t.}\quad
\int_{\mathcal X}\max_{1\le i\le N}\exp\!\big(s_i-\lambda\|x-a_i\|_2\big)\,dx\le1.
\tag{P1}
$$

且若 $(\lambda^\star,s^\star)$ 是 $P1$ 的最优解，可取最优预测密度
$$
p^\star(x)=\max_{1\le i\le N}\exp\!\big(s_i^\star-\lambda^\star\|x-a_i\|_2\big).
$$

---

### 4.2 详细证明

记 $\varepsilon:=\varepsilon_N(\beta)$ 以简化记号。

#### 第一步：对固定 $p$ 做内层对偶化

固定任意 $p\in\mathcal P$。由定理A推论：
$$
\inf_{Q\in\mathbb B_{\varepsilon}(\hat P_{k,z})}\mathbb E_Q[\log p]
=
\sup_{\lambda\ge0}
\left\{
-\lambda\varepsilon+\frac1N\sum_{i=1}^N
\inf_{x\in\mathcal X}\big(\log p(x)+\lambda\|x-a_i\|_2\big)
\right\}.
\tag{1}
$$

代回外层 $\sup_p$：
$$
\bar V_k(z)=
\sup_{p\in\mathcal P,\ \lambda\ge0}
\left\{
-\lambda\varepsilon+\frac1N\sum_{i=1}^N
\inf_{x\in\mathcal X}\big(\log p(x)+\lambda\|x-a_i\|_2\big)
\right\}.
\tag{2}
$$

#### 第二步：引入辅助变量，把 $\inf$ 约束线性化

对每个 $i$ 引入 $s_i\in\mathbb R$，并施加
$$
s_i\le \inf_{x\in\mathcal X}\big(\log p(x)+\lambda\|x-a_i\|_2\big).
\tag{3}
$$

(3) 等价于
$$
\log p(x)\ge s_i-\lambda\|x-a_i\|_2,\quad \forall x\in\mathcal X,\ \forall i.
\tag{4}
$$

把所有 $i$ 合并：
$$
p(x)\ge g_{\lambda,s}(x):=
\max_{1\le i\le N}\exp\!\big(s_i-\lambda\|x-a_i\|_2\big),\quad \forall x.
\tag{5}
$$

于是 (2) 等价改写为
$$
\sup_{p,\lambda,s}
\left\{-\lambda\varepsilon+\frac1N\sum_{i=1}^Ns_i\right\}
\quad
\text{s.t.}\quad
p\in\mathcal P,\ p\ge g_{\lambda,s}.
\tag{6}
$$

#### 第三步：消去函数变量 $p$（关键等价）

我们证明：
$$
\exists\,p\in\mathcal P\ \text{s.t.}\ p\ge g_{\lambda,s}
\iff
\int_{\mathcal X}g_{\lambda,s}(x)\,dx\le1.
\tag{7}
$$

**必要性**：若存在 $p\in\mathcal P$ 且 $p\ge g_{\lambda,s}$，则
$$
1=\int_{\mathcal X}p(x)\,dx\ge \int_{\mathcal X}g_{\lambda,s}(x)\,dx.
$$
故右侧成立。

**充分性**：若 $\int g_{\lambda,s}\le1$，构造
$$
\tilde p(x):=g_{\lambda,s}(x)+\Big(1-\int_{\mathcal X}g_{\lambda,s}(u)\,du\Big)\rho(x).
\tag{8}
$$
则
1. $\tilde p(x)\ge g_{\lambda,s}(x)$；
2. $\int\tilde p=1$；
3. 由 A4，$\rho>0$ 连续；由 (5) 可知 $g_{\lambda,s}>0$ 且连续，所以 $\tilde p>0$ 连续，从而 $\log\tilde p$ 连续（特别是下半连续）。

故 $\tilde p\in\mathcal P$，充分性成立，(7) 证毕。

因此 (6) 完全等价于 $P1$。

#### 第四步：最优点积分约束必然绑定

设 $(\lambda,s)$ 可行且
$$
\int g_{\lambda,s}(x)\,dx<1.
\tag{9}
$$
取任意
$$
0<\delta<-\log\!\left(\int g_{\lambda,s}\right),
$$
令 $s_i' = s_i+\delta$，则
$$
g_{\lambda,s'}(x)=e^\delta g_{\lambda,s}(x),\qquad
\int g_{\lambda,s'}=e^\delta\int g_{\lambda,s}\le1.
$$
所以 $(\lambda,s')$ 仍可行，而目标值增加
$$
\frac1N\sum_i s_i' -\frac1N\sum_i s_i=\delta>0,
$$
与最优性矛盾。故最优时必有 $\int g_{\lambda^\star,s^\star}=1$。

于是
$$
p^\star(x):=g_{\lambda^\star,s^\star}(x)
$$
本身就是合法密度且达到同一目标值。

#### 第五步：凸性说明

对固定 $x$ 与 $i$，函数
$$
(\lambda,s_i)\mapsto \exp\!\big(s_i-\lambda\|x-a_i\|_2\big)
$$
是凸函数（指数复合仿射）。有限个凸函数取最大仍凸，积分保持凸，因此约束函数
$$
(\lambda,s)\mapsto
\int_{\mathcal X}\max_i\exp\!\big(s_i-\lambda\|x-a_i\|_2\big)\,dx
$$
凸。目标是仿射（等价于最小化其相反数），所以 $P1$ 是凸优化问题。  
定理1得证。$\square$

---

## 5. 定理2：严格耦合单步问题的可计算下界

### 5.1 定理陈述

定义名义锚点与半径
$$
a_i:=\bar f_k(z)+\hat x_{k+1,i}-\bar f_k(\hat z_{k,i}),\qquad
R_i:=\sqrt{\gamma_0(z)}+\sqrt{\gamma_0(\hat z_{k,i})}.
$$

再定义
$$
V_k^{\mathrm{low}}(z):=
\sup_{\lambda\ge0,\ s\in\mathbb R^N}
\left\{
-\lambda\varepsilon_N(\beta)+\frac1N\sum_{i=1}^Ns_i
\right\}
$$
$$
\text{s.t.}\quad
\int_{\mathcal X}\max_{1\le i\le N}
\exp\!\Big(s_i-\lambda[\|x-a_i\|_2-R_i]_+\Big)\,dx\le1.
\tag{P2}
$$
其中 $[t]_+:=\max\{t,0\}$。则在 A1–A5 下
$$
V_k^{\mathrm{low}}(z)\le V_k^{\mathrm{strict}}(z).
$$

若 $P2$ 最优解存在，且记为 $(\lambda^\star,s^\star)$，则可取
$$
p_{\mathrm{low}}^\star(x)=
\max_i\exp\!\Big(s_i^\star-\lambda^\star[\|x-a_i\|_2-R_i]_+\Big).
$$

---

### 5.2 详细证明

记 $\varepsilon:=\varepsilon_N(\beta)$。固定任意 $p\in\mathcal P$，定义
$$
J_p^{\mathrm{strict}}(z):=
\inf_{f\in\mathfrak F_k}\ \inf_{Q\in\mathbb B_\varepsilon(\hat P_{x,N}^{f,z})}
\mathbb E_Q[\log p(X)].
$$

我们将构造可计算函数 $\underline\Phi(p)$，证明
$$
\underline\Phi(p)\le J_p^{\mathrm{strict}}(z).
\tag{10}
$$

#### 第一步：固定 $f$ 时应用定理A

给定 $f\in\mathfrak F_k$，定义
$$
\Delta\nu:=f(z)-\bar f_k(z),\qquad
\Delta f_i:=f(\hat z_{k,i})-\bar f_k(\hat z_{k,i}),
$$
则
$$
a_i^{\mathrm{strict}}(f,z)=a_i+\Delta\nu-\Delta f_i.
$$

对固定 $f$，由定理A推论：
$$
\inf_{Q\in\mathbb B_\varepsilon(\hat P_{x,N}^{f,z})}\mathbb E_Q[\log p]
=
\sup_{\lambda\ge0}
\left\{
-\lambda\varepsilon+\frac1N\sum_{i=1}^N
\inf_x\big(\log p(x)+\lambda\|x-a_i-\Delta\nu+\Delta f_i\|_2\big)
\right\}.
\tag{11}
$$

于是
$$
J_p^{\mathrm{strict}}(z)=
\inf_{(\Delta\nu,\Delta f_i)\in\mathcal S_{\mathrm{shared}}}
\sup_{\lambda\ge0}G(\lambda,\Delta\nu,\Delta f),
\tag{12}
$$
其中
$$
\mathcal S_{\mathrm{shared}}:=
\left\{
(\Delta\nu,\Delta f_i):
\|\Delta\nu\|_2\le\sqrt{\gamma_0(z)},\ 
\|\Delta f_i\|_2\le\sqrt{\gamma_0(\hat z_{k,i})}
\right\}.
$$

#### 第二步：放松共享变量（得到保守下界）

引入更大的可行域
$$
\mathcal S_{\mathrm{relax}}:=
\left\{
(\Delta\nu_i,\Delta f_i):
\|\Delta\nu_i\|_2\le\sqrt{\gamma_0(z)},\ 
\|\Delta f_i\|_2\le\sqrt{\gamma_0(\hat z_{k,i})}
\right\}.
$$

由于共享情形可嵌入独立情形（取 $\Delta\nu_i\equiv\Delta\nu$），有
$$
\mathcal S_{\mathrm{shared}}\subseteq \mathcal S_{\mathrm{relax}}.
$$
因此（最小化区域变大，值不增）：
$$
\inf_{\mathcal S_{\mathrm{shared}}}\sup_\lambda G
\ge
\inf_{\mathcal S_{\mathrm{relax}}}\sup_\lambda \widetilde G.
\tag{13}
$$

#### 第三步：弱对偶交换次序

对任意函数 $\widetilde G$ 均有
$$
\inf_{\mathcal S_{\mathrm{relax}}}\sup_{\lambda\ge0}\widetilde G
\ge
\sup_{\lambda\ge0}\inf_{\mathcal S_{\mathrm{relax}}}\widetilde G.
\tag{14}
$$

把 (13)(14) 合并，得
$$
J_p^{\mathrm{strict}}(z)\ge
\sup_{\lambda\ge0}\inf_{\mathcal S_{\mathrm{relax}}}\widetilde G(\lambda,\cdot).
\tag{15}
$$

#### 第四步：显式计算内层最短距离

对每个 $i$ 定义
$$
\delta_i:=\Delta\nu_i-\Delta f_i.
$$
由 Minkowski 和，
$$
\delta_i\in \mathbb B_{R_i},\quad
R_i=\sqrt{\gamma_0(z)}+\sqrt{\gamma_0(\hat z_{k,i})}.
$$

于是对任意 $x$：
$$
\inf_{\delta_i\in\mathbb B_{R_i}}\|x-a_i-\delta_i\|_2
=\operatorname{dist}(x-a_i,\mathbb B_{R_i})
=[\|x-a_i\|_2-R_i]_+.
\tag{16}
$$

因此
$$
\inf_{\delta_i\in\mathbb B_{R_i}}\inf_x
\big(\log p(x)+\lambda\|x-a_i-\delta_i\|_2\big)
=
\inf_x\big(\log p(x)+\lambda[\|x-a_i\|_2-R_i]_+\big).
\tag{17}
$$

将其代回 (15)，得到
$$
\underline\Phi(p):=
\sup_{\lambda\ge0}
\left\{
-\lambda\varepsilon+\frac1N\sum_{i=1}^N
\inf_x\big(\log p(x)+\lambda[\|x-a_i\|_2-R_i]_+\big)
\right\}
\le
J_p^{\mathrm{strict}}(z).
\tag{18}
$$

对 $p$ 取上确界：
$$
\sup_{p\in\mathcal P}\underline\Phi(p)\le
\sup_{p\in\mathcal P}J_p^{\mathrm{strict}}(z)=V_k^{\mathrm{strict}}(z).
\tag{19}
$$

#### 第五步：把左边精确化为 $P2$

定义
$$
h_i(x):=[\|x-a_i\|_2-R_i]_+.
$$
与定理1同样的变量替换（把 $\|x-a_i\|_2$ 换成 $h_i(x)$）得到：
$$
\sup_{p\in\mathcal P}\underline\Phi(p)
=
\sup_{\lambda\ge0,s\in\mathbb R^N}
\left\{-\lambda\varepsilon+\frac1N\sum_i s_i\right\}
$$
$$
\text{s.t.}\quad
\int_{\mathcal X}\max_i\exp\!\big(s_i-\lambda h_i(x)\big)\,dx\le1,
$$
即正是 $P2$，因此
$$
\sup_{p\in\mathcal P}\underline\Phi(p)=V_k^{\mathrm{low}}(z).
\tag{20}
$$

由 (19)(20) 立即得到
$$
V_k^{\mathrm{low}}(z)\le V_k^{\mathrm{strict}}(z).
$$

若 $P2$ 最优解存在，积分约束同理在最优点绑定，于是可取
$$
p_{\mathrm{low}}^\star(x)=\max_i\exp\!\big(s_i^\star-\lambda^\star h_i(x)\big).
$$
定理2证毕。$\square$

---

## 6. 两个结论的关系

在同一组假设下，单步严格问题满足
$$
V_k^{\mathrm{low}}(z)\le V_k^{\mathrm{strict}}(z).
$$

当 $\gamma_0\equiv0$ 时，严格问题退化到名义上界子问题，得到
$$
V_k^{\mathrm{strict}}(z)=\bar V_k(z),
$$
此时定理1给出精确有限维形式。

---

## 7. 参考文献

1. **R. Gao, A. J. Kleywegt** (2022). *Distributionally Robust Stochastic Optimization with Wasserstein Distance*.  
2. **P. M. Esfahani, D. Kuhn** (2018). *Data-Driven Distributionally Robust Optimization Using the Wasserstein Metric*.  
3. **N. Fournier, A. Guillin** (2015). *On the Rate of Convergence in Wasserstein Distance of the Empirical Measure*.  

