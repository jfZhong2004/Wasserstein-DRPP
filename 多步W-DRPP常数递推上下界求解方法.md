# 多步 W-DRPP 常数递推上下界求解方法

本文档只讨论我们刚才确认的这一类方法：

- 上界：“常数 continuation + 平移不变性” 的严格无 $z$ 常数递推；
- 下界：在 $\Gamma_0$ 保守化后，采用样本依赖半径 $\bar R_{t,i}$ 的严格无 $z$ 常数递推；
- 数值求解：对每个阶段子问题再做 $\tau$-平滑，以获得更稳定的光滑凸优化问题。

不讨论其它上界、下界、贪心递推、KR-Lipschitz 理论界或高斯次优策略。

---

## 1. 问题与记号

考虑多步 Wasserstein-DRPP 的严格问题，其价值函数记为
$$
V_t^{strict}(z)
=
\sup_{\hat p_t\in\mathcal F}
\inf_{\substack{f_t:\;\|f_t(\cdot)-\bar f_t(\cdot)\|_2^2\le \gamma_0(\cdot)}}
\inf_{P_w\in\mathbb B_{\varepsilon_t}(\hat P_{w,N}^{f_t})}
\mathbb E_{P_w}\!\left[
\log \hat p_t(f_t(z)+w)+V_{t+1}^{strict}(f_t(z)+w)
\right],
$$
终端条件为
$$
V_T^{strict}\equiv 0.
$$

为避免状态/控制记号混淆，本文把
$$
V_{t+1}^{strict}(x)
$$
视为已经定义在下一状态空间上的 continuation。若原始系统写作
$z=(x,u)$，则这里等价于已经固定下一步控制策略
$u_{t+1}=\pi_{t+1}(x_{t+1})$，或者已经把控制优化吸收到
$V_{t+1}^{strict}$ 的定义中。本文只处理分布鲁棒预测部分的递推，不额外讨论控制策略优化。

记训练样本为
$$
\{(\hat x_{t+1,i},\hat z_{t,i})\}_{i=1}^N.
$$
定义经验噪声样本
$$
\hat w_{t,i}:=\hat x_{t+1,i}-\bar f_t(\hat z_{t,i}),
$$
以及名义预测锚点
$$
a_{t,i}(z):=\bar f_t(z)+\hat w_{t,i}.
$$

下界构造中还会用到
$$
R_{t,i}(z):=\sqrt{\gamma_0(z)}+\sqrt{\gamma_0(\hat z_{t,i})}.
$$
若存在全局常数
$$
\Gamma_0\ge \gamma_0(z),\qquad \forall z\in\mathcal Z,
$$
则定义样本依赖的统一半径
$$
\bar R_{t,i}:=\sqrt{\Gamma_0}+\sqrt{\gamma_0(\hat z_{t,i})},
\qquad i=1,\dots,N,
$$
从而对任意 $z$ 都有
$$
R_{t,i}(z)\le \bar R_{t,i}.
$$

本文默认
$$
\mathcal X=\mathbb R^{d_x},
$$
因为后续消去 $z$ 依赖要用到全空间上的平移不变性。

为使后续从单步结论推广到全空间多步递推是严格的，本文补充采用以下假设。

**B1（全空间版本）.**
所有积分均相对于 $\mathbb R^{d_x}$ 上的 Lebesgue 测度；所有锚点
$a_{t,i}(z)$、$\hat w_{t,i}$ 均在 $\mathbb R^{d_x}$ 中。

**B2（半径与有限性）.**
$\varepsilon_t>0$，$\Gamma_0<\infty$，且所有
$\gamma_0(\hat z_{t,i})<\infty$。因此 (U1)、(L3) 及其平滑版本的有限可行解必有 $\lambda>0$；当 $\lambda=0$ 时全空间积分发散，视为不可行。

**B3（预测密度类）.**
存在正连续参考密度 $\rho$，满足
$$
\rho(x)>0,\qquad \int_{\mathbb R^{d_x}}\rho(x)\,dx=1.
$$
$\mathcal F$ 至少包含如下密度：若
$$
g(x)=e^{-v(x)}
\max_i\exp\!\big(s_i-\lambda\varphi_i(x)\big)
$$
满足 $\int g\le1$，则
$$
p(x)=g(x)+\left(1-\int_{\mathbb R^{d_x}}g(u)\,du\right)\rho(x)
$$
属于 $\mathcal F$。同时其中的 $p$ 满足
$\int_{\mathbb R^{d_x}}p(x)\,dx=1$、$\log p$ 下半连续。

**B4（continuation 正则性）.**
本文真正用于常数递推证明的是常数 continuation。若为表述方便写一般 continuation $v$，则
admissible 指 $v$ 连续，并且使 $-\log p-v$ 满足 Wasserstein 强对偶在
$\mathbb R^{d_x}$ 上所需的上半连续与可积/增长条件，同时保证有限维约束消去 $p$ 时构造出的密度仍属于 $\mathcal F$。常数 continuation 自动满足该条件。

**B5（全空间强对偶）.**
单步文档中在紧集 $\mathcal X$ 上使用的 Wasserstein 强对偶，在本文中替换为其
$\mathbb R^{d_x}$ 版本：对满足 B3-B4 的损失函数，经验分布中心有限且一阶矩有限时，固定损失的 Wasserstein 强对偶成立。

**B6（值函数可测性）.**
$V_t^{strict}$ 为实值可测函数，且本文出现的期望、上确界和下确界均有限；若允许扩展实值，则后续不等式只在两侧有定义的情形下使用。

---

## 2. 方法核心

完整的多步 Bellman 子问题一般依赖于当前状态 $z$，原因有两点：

- 锚点 $a_{t,i}(z)=\bar f_t(z)+\hat w_{t,i}$ 随 $z$ 平移；
- continuation 项 $V_{t+1}$ 会以权重 $e^{-V_{t+1}(x)}$ 进入积分约束。

因此，若直接处理函数型 continuation，就仍然是逐状态的动态规划问题。

本方法的关键是把 continuation 保守化为常数 $c$。此时约束中的权重 $e^{-c}$ 只是常数因子，再结合变量代换
$$
y:=x-\bar f_t(z),
$$
锚点中的 $\bar f_t(z)$ 会被完全消掉，阶段优化值就不再依赖 $z$。

要点如下：

- 无 $z$ 的来源不是 LSE；
- 无 $z$ 的来源是“常数 continuation + $\mathbb R^{d_x}$ 上的平移不变性”；
- $\tau$ 平滑只是把每个阶段子问题变成更稳定的光滑优化问题。

后续会反复使用如下单步 continuation 引理。它是
单步DRPP_OT强对偶严格证明.md 中定理 1 与定理 2 的直接推广：把单步证明里的
$\log p(x)$ 统一替换为 $\log p(x)+v(x)$。

**引理 1（带 continuation 的单步有限维化与下界）.**
在 B1-B6 下，对任意 admissible continuation $v$：

1. 名义上界算子满足精确有限维形式
   $$
   (\mathcal T_t^{up}v)(z)
   =
   \sup_{\lambda\ge0,\mathbf s}
   \left\{-\lambda\varepsilon_t+\frac1N\sum_i s_i\right\}
   $$
   $$
   \text{s.t.}\quad
   \int_{\mathbb R^{d_x}}e^{-v(x)}
   \max_i\exp\!\big(s_i-\lambda\|x-a_{t,i}(z)\|\big)\,dx
   \le 1.
   \tag{G1}
   $$

2. 对
   $$
   R_{t,i}(z)=\sqrt{\gamma_0(z)}+\sqrt{\gamma_0(\hat z_{t,i})},
   $$
   定义 P2 型算子
   $$
   (\mathcal P_t^{\mathbf R(z)}v)(z)
   :=
   \sup_{\lambda\ge0,\mathbf s}
   \left\{-\lambda\varepsilon_t+\frac1N\sum_i s_i\right\}
   $$
   $$
   \text{s.t.}\quad
   \int_{\mathbb R^{d_x}}e^{-v(x)}
   \max_i\exp\!\Big(
   s_i-\lambda[\|x-a_{t,i}(z)\|-R_{t,i}(z)]_+
   \Big)\,dx
   \le1.
   \tag{G2}
   $$
   则有
   $$
   (\mathcal P_t^{\mathbf R(z)}v)(z)
   \le
   (\mathcal T_t^{strict}v)(z).
   \tag{G3}
   $$

证明只需两步。第一步，用 B5 对固定 $p$ 和损失
$\log p+v$ 做 Wasserstein 对偶，得到 (G1)；第二步，沿用单步定理 2 的共享模型误差放松、弱对偶交换和最短距离计算，把
$\|x-a_i\|$ 换成 $[\|x-a_i\|-R_i]_+$，得到 (G2) 对严格耦合算子的下界性质 (G3)。

---

## 3. 多步上界：常数 continuation 递推

### 3.1 上界 Bellman 算子

令 $\gamma_0\equiv 0$，对应的上界 Bellman 算子为
$$
(\mathcal T_t^{up}v)(z)
:=
\sup_{\hat p_t\in\mathcal F}
\inf_{P\in\mathbb B_{\varepsilon_t}(\hat P_{N,z}^{pred})}
\mathbb E_P[\log \hat p_t(X)+v(X)],
$$
其中
$$
\hat P_{N,z}^{pred}
=
\frac1N\sum_{i=1}^N\delta_{a_{t,i}(z)}.
$$
由于 $\bar f_t$ 本身满足模型误差约束，严格问题中的内层 $\inf_{f_t}$ 至少包含名义模型这一种选择；把内层不确定模型集合缩小到 $f_t=\bar f_t$ 会使最坏情形值不减。因此由模糊集包含关系可知
$$
V_t^{strict}(z)\le (\mathcal T_t^{up}V_{t+1}^{strict})(z).
$$

### 3.2 阶段常数上界

定义阶段常数值
$$
u_t^{const}
:=
\sup_{\lambda\ge0,\mathbf s\in\mathbb R^N}
\left\{
-\lambda\varepsilon_t+\frac1N\sum_{i=1}^Ns_i
\right\}
$$
$$
\text{s.t.}\quad
\int_{\mathbb R^{d_x}}
\max_{1\le i\le N}
\exp\!\big(s_i-\lambda\|y-\hat w_{t,i}\|\big)\,dy
\le 1.
\tag{U1}
$$

**命题 1（常数 continuation 的上界阶段值）.**
对任意常数 $c\in\mathbb R$ 与任意状态 $z$，都有
$$
(\mathcal T_t^{up}c)(z)=c+u_t^{const}.
\tag{U2}
$$

**说明（U1 与单步文档中 P1 的关系）.**
单步文档 单步DRPP_OT强对偶严格证明.md 中的上界问题 P1 是对固定当前状态 $z$ 直接写出的原始单步形式：
$$
\sup_{\lambda\ge0,\mathbf s}
\left\{
-\lambda\varepsilon+\frac1N\sum_{i=1}^Ns_i
\right\}
$$
$$
\text{s.t.}\quad
\int_{\mathcal X}
\max_i\exp\!\big(s_i-\lambda\|x-a_i(z)\|\big)\,dx\le1,
$$
其中
$$
a_i(z)=\bar f_t(z)+\hat w_{t,i}.
$$
因此 P1 的问题数据确实由 $z$ 参数化。

若作变量代换
$$
y:=x-\bar f_t(z),
$$
则约束变为
$$
\int_{\mathcal X-\bar f_t(z)}
\max_i\exp\!\big(s_i-\lambda\|y-\hat w_{t,i}\|\big)\,dy\le1.
\tag{U1-pre}
$$
这说明：

- 若只假设 $\mathcal X\subset\mathbb R^{d_x}$ 是一般紧集，则积分域 $\mathcal X-\bar f_t(z)$ 仍依赖于 $z$，所以 P1 一般仍可视为关于 $z$ 的问题；
- 若进一步假设 $\mathcal X=\mathbb R^{d_x}$，则
  $$
  \mathbb R^{d_x}-\bar f_t(z)=\mathbb R^{d_x},
  $$
  积分域不变，(U1-pre) 就精确退化为 (U1)。

因此，U1 不是另一个不同的问题，而是 P1 在“全空间 + 去共同平移”条件下得到的去状态版本。

**证明思路：**

1. 对常数 continuation $v\equiv c$，有限维约束写成
   $$
   \int_{\mathbb R^{d_x}}e^{-c}
   \max_i\exp\!\big(s_i-\lambda\|x-a_{t,i}(z)\|\big)\,dx\le1.
   $$
2. 令 $r_i:=s_i-c$，目标函数变为
   $$
   c-\lambda\varepsilon_t+\frac1N\sum_i r_i.
   $$
3. 再令 $y=x-\bar f_t(z)$，并用 $a_{t,i}(z)=\bar f_t(z)+\hat w_{t,i}$，约束恰化为 (U1)。
4. 因而最优值等于 $c+u_t^{const}$，且右端与 $z$ 无关。

### 3.3 多步常数递推上界

定义
$$
U_T^{const}:=0,\qquad
U_t^{const}:=u_t^{const}+U_{t+1}^{const}.
$$
则显然
$$
U_t^{const}
=
\sum_{j=t}^{T-1}u_j^{const}.
\tag{U3}
$$

**定理 1（严格无 $z$ 的多步上界）.**
对所有 $t,z$ 都有
$$
V_t^{strict}(z)\le U_t^{const}.
\tag{U4}
$$

**证明思路：**

1. $\mathcal T_t^{up}$ 对 continuation 单调：若 $v_1\le v_2$，则
   $$
   \mathcal T_t^{up}v_1\le \mathcal T_t^{up}v_2.
   $$
2. 反向归纳。若已知 $V_{t+1}^{strict}(x)\le U_{t+1}^{const}$，则
   $$
   V_t^{strict}(z)
   \le (\mathcal T_t^{up}V_{t+1}^{strict})(z)
   \le (\mathcal T_t^{up}U_{t+1}^{const})(z)
   = U_{t+1}^{const}+u_t^{const}
   = U_t^{const}.
   $$

因此，上界不需要状态网格，也不需要函数逼近；只要逐阶段求出 $u_t^{const}$ 并求和即可。

---

## 4. 多步上界的 $\tau$-平滑版本

为了保持“上界方向”，必须使用**下逼近 max** 的平滑函数
$$
m_\tau^{-}(A_1,\dots,A_N)
:=
\frac1\tau\log\sum_{i=1}^N e^{\tau A_i}
-\frac{\log N}{\tau},
\qquad \tau>0.
$$
它满足
$$
m_\tau^{-}(A)\le \max_i A_i.
$$

定义平滑阶段值
$$
u_{t,\tau}^{const,up}
:=
\sup_{\lambda\ge0,\mathbf s\in\mathbb R^N}
\left\{
-\lambda\varepsilon_t+\frac1N\sum_{i=1}^Ns_i
\right\}
$$
$$
\text{s.t.}\quad
\int_{\mathbb R^{d_x}}
\exp\!\Big(
m_\tau^{-}(A_{t,1}^{up}(y),\dots,A_{t,N}^{up}(y))
\Big)\,dy
\le1,
\tag{U5}
$$
其中
$$
A_{t,i}^{up}(y):=s_i-\lambda\|y-\hat w_{t,i}\|.
$$

再定义
$$
U_{T,\tau}^{const,up}:=0,\qquad
U_{t,\tau}^{const,up}
:=
u_{t,\tau}^{const,up}+U_{t+1,\tau}^{const,up}.
$$

**定理 2（平滑上界）.**
对所有 $t,z$，有
$$
V_t^{strict}(z)\le U_t^{const}\le U_{t,\tau}^{const,up}.
\tag{U6}
$$
并且当 $\tau\to\infty$ 时，
$$
U_{t,\tau}^{const,up}\downarrow U_t^{const}.
\tag{U7}
$$

**理由：**

令
$$
\delta_\tau:=\frac{\log N}{\tau}.
$$
由 LSE 基本不等式，
$$
\max_i A_i-\delta_\tau
\le
m_\tau^{-}(A)
\le
\max_i A_i.
$$
因此平滑约束函数 $G_\tau^-$ 与原约束函数 $G$ 满足
$$
e^{-\delta_\tau}G\le G_\tau^-\le G.
$$
于是 $\{G\le1\}\subseteq\{G_\tau^-\le1\}$，所以
$$
u_t^{const}\le u_{t,\tau}^{const,up}.
$$
反过来，若 $(\lambda,s)$ 满足 $G_\tau^-\le1$，则 $G(\lambda,s)\le e^{\delta_\tau}$；把所有 $s_i$ 同时减去 $\delta_\tau$ 后得到原问题可行点，目标值下降 $\delta_\tau$。故
$$
u_{t,\tau}^{const,up}\le u_t^{const}+\delta_\tau.
$$
合并得
$$
0\le
u_{t,\tau}^{const,up}-u_t^{const}
\le
\delta_\tau.
\tag{U8}
$$
由于 $m_\tau^-$ 随 $\tau$ 单调上升，平滑可行域随 $\tau$ 收缩，阶段值单调下降；再由 (U8) 得到
$u_{t,\tau}^{const,up}\downarrow u_t^{const}$，逐阶段求和即得 (U7)。

---

## 5. 多步下界：$\Gamma_0$ 保守化后的常数递推

### 5.1 保守下界算子

下界侧先对模型不确定性做统一保守化。利用
$$
R_{t,i}(z)\le \bar R_{t,i},
$$
对 admissible continuation $v$ 定义比较算子
$$
(\mathcal T_t^{\bar{\mathbf R}}v)(z)
:=
\sup_{\lambda\ge0,\mathbf s\in\mathbb R^N}
\left\{
-\lambda\varepsilon_t+\frac1N\sum_{i=1}^Ns_i
\right\}
$$
$$
\text{s.t.}\quad
\int_{\mathbb R^{d_x}}e^{-v(x)}
\max_{1\le i\le N}
\exp\!\Big(
s_i-\lambda[\|x-a_{t,i}(z)\|-\bar R_{t,i}]_+
\Big)\,dx
\le1.
\tag{L1}
$$

该算子与引理 1 中的 $(\mathcal P_t^{\mathbf R(z)}v)(z)$ 的区别只在于把
$R_{t,i}(z)$ 换成了更大的 $\bar R_{t,i}$。由于
$$
\bar R_{t,i}\ge R_{t,i}(z),
$$
有
$$
[\|x-a_{t,i}(z)\|-\bar R_{t,i}]_+
\le
[\|x-a_{t,i}(z)\|-R_{t,i}(z)]_+.
$$
在 $\lambda\ge0$ 时，左侧指数核逐点不小于真实半径对应的指数核，因此 (L1) 的积分约束更严格，可行域更小。于是
$$
(\mathcal T_t^{\bar{\mathbf R}}v)(z)
\le
(\mathcal P_t^{\mathbf R(z)}v)(z).
\tag{L2a}
$$
再由引理 1 的 (G3) 得
$$
(\mathcal T_t^{\bar{\mathbf R}}v)(z)
\le
(\mathcal T_t^{strict}v)(z).
\tag{L2}
$$

### 5.2 阶段常数下界

定义阶段常数值
$$
\ell_t^{const}
:=
\sup_{\lambda\ge0,\mathbf s\in\mathbb R^N}
\left\{
-\lambda\varepsilon_t+\frac1N\sum_{i=1}^Ns_i
\right\}
$$
$$
\text{s.t.}\quad
\int_{\mathbb R^{d_x}}
\max_{1\le i\le N}
\exp\!\Big(
s_i-\lambda[\|y-\hat w_{t,i}\|-\bar R_{t,i}]_+
\Big)\,dy
\le1.
\tag{L3}
$$

**命题 2（常数 continuation 的下界阶段值）.**
对任意常数 $c\in\mathbb R$ 与任意状态 $z$，都有
$$
(\mathcal T_t^{\bar{\mathbf R}}c)(z)=c+\ell_t^{const}.
\tag{L4}
$$

证明与上界完全平行：先令 $r_i=s_i-c$，再作变量代换 $y=x-\bar f_t(z)$，即可把 $a_{t,i}(z)=\bar f_t(z)+\hat w_{t,i}$ 中的 $z$ 消掉。

### 5.3 多步常数递推下界

定义
$$
L_T^{const}:=0,\qquad
L_t^{const}:=\ell_t^{const}+L_{t+1}^{const},
$$
即
$$
L_t^{const}
=
\sum_{j=t}^{T-1}\ell_j^{const}.
\tag{L5}
$$

**定理 3（严格无 $z$ 的多步下界）.**
对所有 $t,z$ 都有
$$
L_t^{const}\le V_t^{strict}(z).
\tag{L6}
$$

**证明思路：**

1. 由 (L2)，有
   $$
   \mathcal T_t^{\bar{\mathbf R}}v\le \mathcal T_t^{strict}v.
   $$
2. $\mathcal T_t^{strict}$ 对 continuation 单调：若 $v_1\le v_2$，则对任意固定 $p,f,P$ 的期望也满足对应不等式，随后取 $\inf$ 与 $\sup$ 保持该序关系。
3. 常数 $L_{t+1}^{const}$ 满足 B4，因此命题 2 给出
   $$
   (\mathcal T_t^{\bar{\mathbf R}}L_{t+1}^{const})(z)
   =
   L_t^{const}.
   $$
4. 反向归纳。若 $L_{t+1}^{const}\le V_{t+1}^{strict}(x)$，则
   $$
   L_t^{const}
   =(\mathcal T_t^{\bar{\mathbf R}}L_{t+1}^{const})(z)
   \le (\mathcal T_t^{strict}L_{t+1}^{const})(z)
   \le (\mathcal T_t^{strict}V_{t+1}^{strict})(z)
   =V_t^{strict}(z).
   $$

这个下界是严格可证的，并且比把所有样本半径进一步并成一个公共 $\bar R_t$ 更紧，因为它保留了样本层面的半径差异 $\bar R_{t,i}$。

---

## 6. 多步下界的 $\tau$-平滑版本

为了保持“下界方向”，必须使用**上逼近 max** 的平滑函数
$$
m_\tau^{+}(A_1,\dots,A_N)
:=
\frac1\tau\log\sum_{i=1}^N e^{\tau A_i},
\qquad \tau>0.
$$
它满足
$$
\max_i A_i\le m_\tau^{+}(A).
$$

定义平滑阶段值
$$
\ell_{t,\tau}^{const,low}
:=
\sup_{\lambda\ge0,\mathbf s\in\mathbb R^N}
\left\{
-\lambda\varepsilon_t+\frac1N\sum_{i=1}^Ns_i
\right\}
$$
$$
\text{s.t.}\quad
\int_{\mathbb R^{d_x}}
\exp\!\Big(
m_\tau^{+}(A_{t,1}^{low}(y),\dots,A_{t,N}^{low}(y))
\Big)\,dy
\le1,
\tag{L7}
$$
其中
$$
A_{t,i}^{low}(y)
:=
s_i-\lambda[\|y-\hat w_{t,i}\|-\bar R_{t,i}]_+.
$$

再定义
$$
L_{T,\tau}^{const,low}:=0,\qquad
L_{t,\tau}^{const,low}
:=
\ell_{t,\tau}^{const,low}+L_{t+1,\tau}^{const,low}.
$$

**定理 4（平滑下界）.**
对所有 $t,z$，有
$$
L_{t,\tau}^{const,low}
\le
L_t^{const}
\le
V_t^{strict}(z).
\tag{L8}
$$
并且当 $\tau\to\infty$ 时，
$$
L_{t,\tau}^{const,low}\uparrow L_t^{const}.
\tag{L9}
$$

**理由：**

仍令 $\delta_\tau=\log N/\tau$。由
$$
\max_i A_i
\le
m_\tau^{+}(A)
\le
\max_i A_i+\delta_\tau
$$
可得约束函数满足
$$
G\le G_\tau^+\le e^{\delta_\tau}G.
$$
于是 $\{G_\tau^+\le1\}\subseteq\{G\le1\}$，所以
$$
\ell_{t,\tau}^{const,low}\le \ell_t^{const}.
$$
反过来，若 $(\lambda,s)$ 对原下界阶段问题可行，则把所有 $s_i$ 同时减去 $\delta_\tau$ 后有
$$
G_\tau^+(\lambda,s-\delta_\tau\mathbf 1)
=
e^{-\delta_\tau}G_\tau^+(\lambda,s)
\le
G(\lambda,s)
\le1,
$$
因此该点对平滑下界问题可行，目标值只下降 $\delta_\tau$。取上确界得
$$
\ell_t^{const}-\delta_\tau
\le
\ell_{t,\tau}^{const,low}
\le
\ell_t^{const}.
\tag{L10}
$$
由于 $m_\tau^+$ 随 $\tau$ 单调下降，平滑可行域随 $\tau$ 放宽，阶段值单调上升；再由 (L10) 得到
$\ell_{t,\tau}^{const,low}\uparrow \ell_t^{const}$，逐阶段求和即得 (L9)。

---

## 7. 求解流程

### 7.1 上界流程

1. 对每个阶段 $t=T-1,\dots,0$，求解阶段问题 (U1)，得到 $u_t^{const}$；
2. 若希望更平滑的数值优化，则改解 (U5)，得到 $u_{t,\tau}^{const,up}$；
3. 直接做求和递推：
   $$
   U_t^{const}=\sum_{j=t}^{T-1}u_j^{const},
   \qquad
   U_{t,\tau}^{const,up}=\sum_{j=t}^{T-1}u_{j,\tau}^{const,up}.
   $$

### 7.2 下界流程

1. 先给出全局模型误差上界 $\Gamma_0$；
2. 对每个阶段、每个样本计算
   $$
   \bar R_{t,i}=\sqrt{\Gamma_0}+\sqrt{\gamma_0(\hat z_{t,i})};
   $$
3. 对每个阶段求解 (L3)，得到 $\ell_t^{const}$；
4. 若希望更平滑的数值优化，则改解 (L7)，得到 $\ell_{t,\tau}^{const,low}$；
5. 直接做求和递推：
   $$
   L_t^{const}=\sum_{j=t}^{T-1}\ell_j^{const},
   \qquad
   L_{t,\tau}^{const,low}=\sum_{j=t}^{T-1}\ell_{j,\tau}^{const,low}.
   $$

### 7.3 计算特征

- 各阶段子问题彼此独立，可以并行；
- 不需要状态离散，不需要值函数拟合；
- 真正需要数值求解的只是每个时刻的单阶段凸优化；
- $\tau$ 平滑只影响阶段子问题的求解稳定性，不改变“常数递推”的结构。

---

## 8. 一维与高维实现建议

### 8.1 一维

若 $d_x=1$ 且 $\mathcal X=\mathbb R$：

- 上界阶段问题 (U1) / (U5) 可直接调用单步定理 1 的一维积分结构；
- 下界阶段问题 (L3) / (L7) 可直接调用单步定理 2 的一维平顶指数核积分结构。

区别只是：

- 单步文档里的锚点现在统一换成 $\hat w_{t,i}$；
- 下界半径统一换成 $\bar R_{t,i}$；
- 多步部分只剩下阶段求值后做常数求和。

### 8.2 高维

若 $d_x\ge 2$：

- 可对 (U5)、(L7) 使用数值积分 + 光滑梯度优化；
- 常见选择是自适应求积（低维）或重要性采样/Monte Carlo（高维）；
- 相比直接求完整 Bellman 问题，这里没有函数型 continuation，所以积分结构明显更简单。

---

## 9. $\tau=1$ 的显式与半显式公式

本节专门讨论
$$
\tau=1
$$
时的平滑常数递推界。此时：

- 上界侧使用的是 $m_1^{-}$，因此
  $$
  \exp(m_1^{-}(A_1,\dots,A_N))
  =
  \frac1N\sum_{i=1}^N e^{A_i};
  $$
- 下界侧使用的是 $m_1^{+}$，因此
  $$
  \exp(m_1^{+}(A_1,\dots,A_N))
  =
  \sum_{i=1}^N e^{A_i}.
  $$

因此，$\tau=1$ 会把阶段约束化成单步文档里已经有闭式积分的 Laplace 核或 Hinge-Laplace 核之和。

### 9.1 上界：任意维解析闭式

对上界平滑阶段值 (U5)，取 $\tau=1$ 后有
$$
\exp\!\Big(m_1^{-}(A_{t,1}^{up}(y),\dots,A_{t,N}^{up}(y))\Big)
=
\frac1N\sum_{i=1}^N e^{s_i-\lambda\|y-\hat w_{t,i}\|}.
$$
因此约束变为
$$
\frac1N\sum_{i=1}^N e^{s_i}\mathcal I_{d_x}^{Lap}(\lambda)\le1,
\tag{E1}
$$
其中
$$
\mathcal I_d^{Lap}(\lambda)
:=
\int_{\mathbb R^d}e^{-\lambda\|x\|}\,dx
=
\frac{C_d}{\lambda^d},
\qquad
C_d:=\frac{2\pi^{d/2}\Gamma(d)}{\Gamma(d/2)}.
$$

由于目标函数对 $s_i$ 对称，最优解满足
$$
s_1=\cdots=s_N=:s.
$$
于是 (E1) 化为
$$
e^s\mathcal I_{d_x}^{Lap}(\lambda)\le1,
$$
最优时取等，故
$$
s=d_x\log\lambda-\log C_{d_x}.
$$
阶段目标变成
$$
J_t^{up}(\lambda)
=
-\lambda\varepsilon_t+d_x\log\lambda-\log C_{d_x}.
$$
其一阶最优条件为
$$
-\varepsilon_t+\frac{d_x}{\lambda}=0,
$$
因此
$$
\lambda_{t,1}^{up,*}=\frac{d_x}{\varepsilon_t},
\qquad
s_{t,1}^{up,*}
=
d_x\log\frac{d_x}{\varepsilon_t}-\log C_{d_x}.
$$
从而得到任意维解析闭式
$$
u_{t,1}^{const,up}
=
-d_x+d_x\log\frac{d_x}{\varepsilon_t}-\log C_{d_x}.
\tag{E2}
$$

对应的多步上界就是
$$
U_{t,1}^{const,up}
=
\sum_{j=t}^{T-1}
\left(
-d_x+d_x\log\frac{d_x}{\varepsilon_j}-\log C_{d_x}
\right).
\tag{E3}
$$
特别地，若 $\varepsilon_j\equiv\varepsilon$，则
$$
U_{t,1}^{const,up}
=
(T-t)
\left(
-d_x+d_x\log\frac{d_x}{\varepsilon}-\log C_{d_x}
\right).
\tag{E4}
$$

> 上界侧在任意维都能闭式，且不需要额外假设样本半径一致，因为上界问题本身不含半径参数。

### 9.2 下界：一维

令 $d_x=1$。由
$$
\mathcal I_1^{Hinge}(\lambda,R)=2R+\frac{2}{\lambda},
$$
对下界平滑阶段值 (L7) 取 $\tau=1$，约束变为
$$
\sum_{i=1}^N e^{s_i}
\left(
2\bar R_{t,i}+\frac{2}{\lambda}
\right)
\le1.
\tag{E5}
$$

由 KKT 条件，
$$
s_i^*
=
-\log\!\left(
N\left(2\bar R_{t,i}+\frac{2}{\lambda^*}\right)
\right),
\qquad i=1,\dots,N.
\tag{E6}
$$
于是阶段值可写成
$$
\ell_{t,1}^{const,low}
=
\sup_{\lambda>0}
\left\{
-\lambda\varepsilon_t
-\log N
-\frac1N\sum_{i=1}^N
\log\!\left(2\bar R_{t,i}+\frac{2}{\lambda}\right)
\right\}.
\tag{E7}
$$

最优 $\lambda$ 满足单变量方程
$$
\varepsilon_t
=
\frac1N\sum_{i=1}^N
\frac{1}{\lambda(1+\bar R_{t,i}\lambda)}.
\tag{E8}
$$
因此，一维下界在一般情形下已经是**半显式**的：只需求解一个标量方程 (E8)。

若进一步满足“同阶段半径一致条件”
$$
\bar R_{t,1}=\cdots=\bar R_{t,N}=:\bar R_t,
\tag{E9}
$$
则 (E8) 化为
$$
\varepsilon_t=\frac{1}{\lambda(1+\bar R_t\lambda)}.
\tag{E10}
$$
此时：

- 若 $\bar R_t>0$，则
  $$
  \lambda_{t,1}^{low,*}
  =
  \frac{\sqrt{1+4\bar R_t/\varepsilon_t}-1}{2\bar R_t};
  \tag{E11}
  $$
- 若 $\bar R_t=0$，则
  $$
  \lambda_{t,1}^{low,*}=\frac{1}{\varepsilon_t}.
  \tag{E12}
  $$

代回 (E7) 得到一维闭式阶段值
$$
\ell_{t,1}^{const,low}
=
-\lambda_{t,1}^{low,*}\varepsilon_t
-\log\!\left(
2N\left(\bar R_t+\frac{1}{\lambda_{t,1}^{low,*}}\right)
\right).
\tag{E13}
$$
从而
$$
L_{t,1}^{const,low}
=
\sum_{j=t}^{T-1}\ell_{j,1}^{const,low}.
\tag{E14}
$$

特别地，当 $\bar R_t=0$ 时，
$$
\ell_{t,1}^{const,low}
=
-1-\log(2N\varepsilon_t).
\tag{E15}
$$

### 9.3 下界：高维

在一般维度 $d_x\ge2$ 下，仍有闭式积分
$$
\mathcal I_d^{Hinge}(\lambda,R)
=
V_dR^d
+S_{d-1}\sum_{k=0}^{d-1}\binom{d-1}{k}\frac{k!\,R^{d-1-k}}{\lambda^{k+1}},
$$
其中 $V_d$ 和 $S_{d-1}$ 分别是 $d$ 维单位球体积与 $(d-1)$ 维单位球面面积。

因此，对 (L7) 取 $\tau=1$ 后，阶段值可写成
$$
\ell_{t,1}^{const,low}
=
\sup_{\lambda>0}
\left\{
-\lambda\varepsilon_t
-\log N
-\frac1N\sum_{i=1}^N
\log \mathcal I_{d_x}^{Hinge}(\lambda,\bar R_{t,i})
\right\}.
\tag{E16}
$$

最优 $\lambda$ 满足单变量方程
$$
\varepsilon_t
=
\frac1N\sum_{i=1}^N
\frac{-\partial_\lambda \mathcal I_{d_x}^{Hinge}(\lambda,\bar R_{t,i})}
{\mathcal I_{d_x}^{Hinge}(\lambda,\bar R_{t,i})}.
\tag{E17}
$$
因此，高维下界在一般情形下是**半显式**的：目标函数与积分项都闭式，但 $\lambda$ 仍需通过一维方程求出。

若进一步满足同阶段半径一致条件 (E9)，则
$$
\ell_{t,1}^{const,low}
=
\sup_{\lambda>0}
\left\{
-\lambda\varepsilon_t
-\log N
-\log \mathcal I_{d_x}^{Hinge}(\lambda,\bar R_t)
\right\},
\tag{E18}
$$
且最优 $\lambda$ 满足
$$
\varepsilon_t
=
-\frac{d}{d\lambda}
\log \mathcal I_{d_x}^{Hinge}(\lambda,\bar R_t).
\tag{E19}
$$
对应多步下界为
$$
L_{t,1}^{const,low}
=
\sum_{j=t}^{T-1}\ell_{j,1}^{const,low}.
\tag{E20}
$$

需要强调的是：

- 一维在同阶段半径一致条件下可以继续化成闭式；
- 高维即使在同阶段半径一致条件下，通常也只能到 (E18)-(E19) 这种“闭式目标 + 一维根方程”的半显式形式；
- 因而“$\tau=1$ 后任意维都闭式”只对上界成立，对下界一般不成立。

---

## 10. 方法定位与限制

这套方法的定位很明确：

- 它给出的是**严格可证的无 $z$ 多步上界与下界**；
- 它不是完整 Bellman 方程的精确求解，而是对 continuation 做了常数保守化；
- 上界侧的保守化来自把未来值函数替换成阶段无关常数；
- 下界侧除了 continuation 常数化以外，还额外用了 $\Gamma_0$ 保守化；
- 因此这套方法适合作为理论夹逼、数值基线和快速评估工具，而不是原始多步问题的精确解。

特别强调：

- 严格链条依赖 B1-B6，尤其是全空间强对偶、$\varepsilon_t>0$、有限 $\Gamma_0$ 与常数 continuation 的可积性；
- 若 $\mathcal X$ 是一般紧集或有状态约束，变量代换后积分域会变成 $\mathcal X-\bar f_t(z)$，无 $z$ 常数递推一般不再成立；
- 若原始问题显式包含控制优化，则需先固定策略或把控制优化吸收到 $V_{t+1}$，否则本文的 Bellman 记号不完整；
- 上界平滑必须用 $m_\tau^-$，否则会破坏上界方向；
- 下界平滑必须用 $m_\tau^+$，否则会破坏下界方向；
- “能否得到无 $z$ 界”的关键不是 $\tau$，而是常数 continuation 后的平移不变性。

---

## 11. 最终结论

在当前 Wasserstein-DRPP 设定下，若只考虑我们刚才讨论的这一类方法，则可得到如下严格链条：

$$
L_{t,\tau}^{const,low}
\;\le\;
L_t^{const}
\;\le\;
V_t^{strict}(z)
\;\le\;
U_t^{const}
\;\le\;
U_{t,\tau}^{const,up},
\qquad \forall t,\ \forall z.
$$

并且当 $\tau\to\infty$ 时，
$$
L_{t,\tau}^{const,low}\uparrow L_t^{const},
\qquad
U_{t,\tau}^{const,up}\downarrow U_t^{const}.
$$

因此，这套方法给出了一类：

- 严格可证；
- 与当前状态 $z$ 无关；
- 逐阶段独立求解；
- 可用 $\tau$ 平滑稳定数值优化

的多步 W-DRPP 上下界构造方案。
