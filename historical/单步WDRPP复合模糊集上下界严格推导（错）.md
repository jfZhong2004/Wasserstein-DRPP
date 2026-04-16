# 单步 WDRPP（复合“函数球 × 分布球”）上下界严格推导

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

要求：不使用单球化 $\rho=\gamma+\varepsilon$ 近似，直接在“函数球 × 分布球”结构上给出可计算且严格可证的上下界。

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
\mathcal P=\left\{p:\mathbb R^{d_x}\to[0,\infty)\ \middle|\ \int p(x)\,dx=1\right\},
$$
并且 $\log p$ 可测。  

**(H3)** 对任意 $p\in\mathcal P$，记 $\ell_p(x):=-\log p(x)$。并显式要求（对应 Theorem 4.2 的 $K=1,\ \ell_1=\ell_p$ 情形）：  
1) $\mathcal X=\mathbb R^{d_x}$ 为闭凸集；  
2) $-\ell_p=\log p$ 是 proper、凸、下半连续函数；  
3) $\ell_p$ 在 $\mathcal X$ 上不恒等于 $-\infty$。  

---

## 2. 需要调用的外部定理（来自 Wasserstein.md）

### 定理 A（Wasserstein 球上的对偶表示）
设经验分布 $\hat P_N=\frac1N\sum_{i=1}^N\delta_{\hat\xi_i}$，半径 $\varepsilon\ge0$，损失函数 $\ell$ 满足上面的显式凸性条件（同型于 (H3)）。则
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
由于平移推前保持 Wasserstein 距离，$(\mathrm{P\!-\!raw})$ 等价为
$$
V_k^{\mathrm{raw}}(z)=
\sup_{p\in\mathcal P}
\inf_{\|m\|_2\le\gamma}
\inf_{P\in\mathbb B_\varepsilon(\hat P_{x,m})}
\mathbb E_P[\log p(X)].
\tag{P0}
$$

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

对固定 $p,m$，令 $\ell_p=-\log p$，由定理 A：
$$
\sup_{P\in\mathbb B_\varepsilon(\hat P_{x,m})}\mathbb E_P[\ell_p(X)]
=
\inf_{\lambda\ge0}
\left\{
\lambda\varepsilon+\frac1N\sum_{i=1}^N
\sup_x\big(\ell_p(x)-\lambda\|x-a_i-m\|_2\big)
\right\}.
$$
两边乘以 $-1$，得到
$$
\Phi(p,m)=
\sup_{\lambda\ge0}
\left\{
-\lambda\varepsilon+\frac1N\sum_{i=1}^N
\inf_x\big(\log p(x)+\lambda\|x-a_i-m\|_2\big)
\right\}.
\tag{1}
$$

故
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
等价于
$$
p(x)\ge g_{\lambda,s,m}(x):=
\max_i\exp\!\big(s_i-\lambda\|x-a_i-m\|_2\big),\ \forall x.
\tag{2}
$$

于是
$$
U(m)=
\sup_{p,\lambda,s}
\left\{-\lambda\varepsilon+\frac1N\sum_i s_i\right\}
\ \text{s.t.}\ p\ge g_{\lambda,s,m},\ \int p=1.
\tag{3}
$$

接下来消去 $p$：

- 必要性：若存在 $p$ 满足 (3)，则 $\int g_{\lambda,s,m}\le\int p=1$。  
- 充分性：若 $\int g_{\lambda,s,m}\le1$，取
$$
p(x)=g_{\lambda,s,m}(x)+\Big(1-\int g_{\lambda,s,m}\Big)\rho(x),
$$
其中 $\rho$ 为任意固定密度（例如标准高斯密度）。则 $p\ge g_{\lambda,s,m}$ 且 $\int p=1$。

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
由 (P0) 与 $U(m)=\sup_p\inf_{P\in\mathbb B_\varepsilon(\hat P_{x,m})}\mathbb E_P[\log p]$，
$$
V_k^{\mathrm{raw}}(z)
=
\sup_p\inf_{\|m\|\le\gamma}F(p,m),
\quad
F(p,m):=\inf_{P\in\mathbb B_\varepsilon(\hat P_{x,m})}\mathbb E_P[\log p].
$$
应用一般不等式 $\sup_p\inf_m F(p,m)\le \inf_m\sup_p F(p,m)$，得 (4)。  
再由 $\inf_{\|m\|\le\gamma}U(m)\le U(m_0)$ 得 (5)。证毕。

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
$\|u-m\|_2\ge \|u\|_2-\gamma$，且显然 $\|u-m\|_2\ge0$，于是
$\|u-m\|_2\ge[\|u\|_2-\gamma]_+$。

上界可达：  
1. 若 $\|u\|_2\le\gamma$，取 $m=u$，值为 0。  
2. 若 $\|u\|_2>\gamma$，取 $m=\gamma\,u/\|u\|_2$，则
$\|u-m\|_2=\|u\|_2-\gamma$。
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

第一步（弱对偶）：
$$
\inf_{\|m\|_2\le\gamma}\sup_{\lambda\ge0}F(p,m,\lambda)
\ge
\sup_{\lambda\ge0}\inf_{\|m\|_2\le\gamma}F(p,m,\lambda).
$$
代回 (7)：
$$
V_k^{\mathrm{raw}}(z)\ge
\sup_p\sup_{\lambda\ge0}
\inf_{\|m\|_2\le\gamma}
\left\{
-\lambda\varepsilon+\frac1N\sum_i\phi_i(m)
\right\},
$$
其中
$
\phi_i(m):=\inf_x(\log p(x)+\lambda\|x-a_i-m\|_2).
$

第二步（共享 $m$ 放松为独立 $m_i$）：
设
$$
\mathcal S:=\{(m_1,\dots,m_N):m_1=\cdots=m_N,\ \|m_i\|_2\le\gamma\},
\quad
\mathcal T:=\{(m_1,\dots,m_N):\|m_i\|_2\le\gamma,\forall i\}.
$$
有 $\mathcal S\subseteq\mathcal T$，因此
$$
\inf_{(m_i)\in\mathcal S}\frac1N\sum_i\phi_i(m_i)
\ge
\inf_{(m_i)\in\mathcal T}\frac1N\sum_i\phi_i(m_i)
=
\frac1N\sum_i\inf_{\|m_i\|_2\le\gamma}\phi_i(m_i).
$$
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

第三步（计算内层距离）：
由 $\inf_{m_i}\inf_x=\inf_x\inf_{m_i}$ 及引理 2，
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

第四步（显式消去 $p$）：
对 (10) 的右侧引入变量 $s_i$，得到等价形式
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
s_i\le\inf_x(\log p(x)+\lambda r_i(x))
\iff
p(x)\ge \max_i\exp(s_i-\lambda r_i(x)),
$$
其中 $r_i(x)=[\|x-a_i\|_2-\gamma]_+$。令
$$
g_{\lambda,s}(x):=\max_i\exp(s_i-\lambda r_i(x)).
$$
于是约束变为 $p\ge g_{\lambda,s}$ 且 $\int p=1$。与定理 1 中同样的可行性判据：

- 若存在 $p\ge g_{\lambda,s}$ 且 $\int p=1$，则 $\int g_{\lambda,s}\le1$；  
- 若 $\int g_{\lambda,s}\le1$，取
$$
p(x)=g_{\lambda,s}(x)+\big(1-\int g_{\lambda,s}\big)\rho(x),
$$
即可得到可行密度。

因此消去 $p$ 后恰得到程序 (L)。

因此 (L) 的最优值不超过 $V_k^{\mathrm{raw}}(z)$，即 (8) 成立。证毕。

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

1. 先证积分约束绑定。若 $\int g^{\mathrm{up}}<1$，取
$$
0<\delta<-\log\!\int g^{\mathrm{up}}
$$
并令 $s_i'=s_i^{\mathrm{up}}+\delta$。则
$$
\int \max_i e^{s_i'-\lambda^{\mathrm{up}}\|x-a_i-m^{\mathrm{up}}\|_2}
=
e^\delta\int g^{\mathrm{up}}\le1
$$
仍可行，而目标值增加 $\delta$，与最优性矛盾。故 $\int g^{\mathrm{up}}=1$。

2. 因此 $p^{\mathrm{up},*}=g^{\mathrm{up}}$ 为概率密度。且对任意 $i,x$，
$$
\log p^{\mathrm{up},*}(x)
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
U(m^{\mathrm{up}})=\Phi(p^{\mathrm{up},*},m^{\mathrm{up}})
$$
，即 $p^{\mathrm{up},*}$ 为上界点位问题的最优预测分布。

3. 记 $\ell^{\mathrm{up}}(x):=-\log p^{\mathrm{up},*}(x)$，则
$$
\Phi(p^{\mathrm{up},*},m^{\mathrm{up}})
=
-\sup_{P\in\mathbb B_\varepsilon(\hat P_{x,m^{\mathrm{up}}})}\mathbb E_P[\ell^{\mathrm{up}}(X)].
$$
由 (H3)，$\ell^{\mathrm{up}}$ 满足 Theorem 4.2 的显式凸性条件，且在 $K=1$ 情形下为 concave loss。故可由 `Wasserstein.md` 的 Corollary 4.6 得到上式 supremum 可达，于是存在 $P^{\mathrm{up},*}$ 使上式取到，等价地得到 $\Phi$ 的 inner inf 取到。证毕。

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

1. 与推论 4 同理，若 $\int g^{\mathrm{low}}<1$ 可统一抬升 $s_i$ 提高目标，故最优时必有 $\int g^{\mathrm{low}}=1$，从而可取 $p^{\mathrm{low},*}=g^{\mathrm{low}}$，并得到其在等价松弛问题中的最优性。

2. 定理 3 仅给出
$$
V_k^{\mathrm{low}}(z)\le V_k^{\mathrm{raw}}(z)
$$
。若出现严格不等式
$$
V_k^{\mathrm{low}}(z)<V_k^{\mathrm{raw}}(z),
$$
则 (L) 的任一最优解都不可能对应原始问题的最优解值。故在未额外证明“下界与原值闭合”（等价地，定理 3 中放松步骤在最优点均取等）之前，不能从下界最优解推出原始问题的最优对抗分布。证毕。

---

## 9. 计算建议

1. 解一次下界凸程序 (L)，得到 $V_k^{\mathrm{low}}(z)$。  
2. 对若干 $m\in\{ \|m\|_2\le\gamma\}$ 解 (U-m)，取最小值得到上界近似。  
3. 报告区间 $[V_k^{\mathrm{low}}(z),\,V_k^{\mathrm{up}}(z)]$ 作为原始单步值证书。

