# 基于Wasserstein模糊集的DRPP问题思考与推导

非常棒的idea！将分布鲁棒概率预测（DRPP）从基于矩的模糊集（Moment-based Ambiguity Set）拓展到基于Wasserstein度量的模糊集，是一个非常有理论价值和实际意义的研究方向。正如你在 `DRPP.pdf` 中看到的，基于矩的模糊集往往会导致预测分布呈现高斯特性；而引入 `Wasserstein.pdf` 中的最优传输视角，可以直接利用数据的经验分布，这往往能保留数据中更丰富的多峰和非对称特征。

以下是我针对你提出的问题，对 **DRPP问题定义** 以及 **单步DRPP在 $\gamma_0(z) \equiv 0$ 时的优化求解** 的推导和思考。

## 1. Wasserstein-based 模糊集与 DRPP问题定义

为了将最优传输理论引入分布鲁棒概率预测（DRPP），我们需要重构模糊集。你提出的思路非常具有启发性：在实际应用中，基于观测到的历史数据集 $\mathcal{D}_N = \{ (\hat{x}_{k+1,i},\hat{z}_{k,i})\}_{i=1}^N$，我们可以利用经验分布直接构建 Wasserstein 球。

为了在数学表达上更加严谨且符合测度论和最优传输文献的惯例，我对你的原始定义符号进行了轻微的规范化（统一了随机变量与样本的符号、显式表达了经验测度对系统函数 $f_k$ 的依赖关系）：

**定义 1 (条件 Wasserstein 模糊集 / Conditional Wasserstein-based Ambiguity Set).**
在时刻 $k$，给定当前状态-控制对 $z_k = z$。设标称状态演化函数为 $\bar{f}_k$，模型不确定性上界为 $\gamma_0(z) \geq 0$。对于置信度 $\beta \in (0,1)$ 及样本量 $N$，基于 Wasserstein 度量的条件模糊集定义为：
$$
\mathcal{I}_{k}^{W}(z) := \left\{ P_{\pmb{x}_{k+1} | z} \in \mathcal{P}(\mathcal{X}) \; \middle| \;
\begin{aligned}
& \pmb{x}_{k+1} = f_{k}(z) + \pmb{w}_{k}, \\
& \|f_{k}(z) - \bar{f}_{k}(z)\|_2^2 \leq \gamma_{0}(z), \\
& P_{\pmb{w}_k} \in \mathbb{B}_{\varepsilon(\beta,N)} \left( \hat{P}_{w,N}^{f_k} \right)
\end{aligned}
\right\}
$$
其中：
* $\pmb{x}_{k+1}$ 和 $\pmb{w}_k$ 为随机变量，$P_{\pmb{w}_k}$ 是随机噪声 $\pmb{w}_k$ 的真实概率测度（分布）。
* $\hat{P}_{w,N}^{f_k} := \frac{1}{N} \sum_{i=1}^{N} \delta_{\hat{w}_{k,i}}$ 是由历史数据导出的**经验噪声分布**（由 Dirac 测度 $\delta$ 的均值构成）。
* $\hat{w}_{k,i} := \hat{x}_{k+1,i} - f_k(\hat{z}_{k,i})$ 是在给定的状态演化函数 $f_k$ 下，由第 $i$ 个数据点解算出的经验噪声样本（添加了小帽 $\hat{\cdot}$ 以与随机变量作区分）。
* $\mathbb{B}_{\varepsilon(\beta,N)}(\cdot)$ 表示 1-Wasserstein 度量下以经验分布为中心、半径为 $\varepsilon(\beta,N)$ 的闭球。半径 $\varepsilon$ 的大小可由统计学中的测度集中不等式（Measure Concentration Inequalities，见 `Wasserstein.pdf` Theorem 3.4）确定，以保证真实的未知噪声分布以 $1-\beta$ 的高概率落入该球内。

基于上述模糊集，多步 DRPP 的 Bellman 方程的单步核心问题（One-step DRPP）可以严格定义为如下的泛函极小极大（Max-Min）优化问题：
$$
\max_{\hat{p}_k \in \mathcal{F}} \inf_{P_{\pmb{x}_{k+1}|z} \in \mathcal{I}_k^W(z)} \mathbb{E}_{P} \left[ \log \hat{p}_k(\pmb{x}_{k+1}) \right]
$$
我们的目标是寻找一个最优的概率预测模型 $\hat{p}_k \in \mathcal{F}$，使得在最坏情况的真实数据分布下，预测得分（例如对数得分 Log-score）的期望最大化。

### Wasserstein-based 模糊集与 Conic 矩模糊集的对比与优势

结合《DRPP.pdf》中对 Conic 矩模糊集（Conic moment-based ambiguity set）的应用与《Wasserstein.pdf》中对数据驱动优化的推导，我们将两者的区别以及 Wasserstein 模糊集的实际含义和核心优势总结如下：

**1. 核心区别**
*   **Conic 模糊集**：仅利用数据的一阶和二阶矩（均值和协方差）信息，通过凸锥不等式约束来包络真实分布（参考 DRPP.pdf Definition 2）。这种方式本质上是对分布形状的宏观统计抽象。
*   **Wasserstein 模糊集**：直接以包含所有原始数据的经验分布 $\hat{P}_{w,N}^{f_k}$ 为中心，利用最优传输成本（Wasserstein metric）作为半径，定义概率测度空间中的“连续球体”。它不依赖于矩的提前提取，而是直接锚定每一个离散样本点。

**2. 实际含义**
Wasserstein 模糊集的实际含义是：在现实系统中，真实的未知数据生成分布（Data-generating distribution）可以通过对当前有限的观测样本进行一定程度的“质量搬运”和“扰动”得到。在给定置信度下，这个 Wasserstein 球代表了——考虑到样本的有限性（采样误差）后——由观测数据所能支撑的所有合理的真实概率分布的集合。

**3. 核心优势**
*   **打破“高斯化”束缚，完美保留数据特征**：在 DRPP 极小极大优化框架下，基于 Conic 矩模糊集的推导最终往往导致最优预测分布退化为单一的高斯分布（参考 DRPP.pdf Theorem 2 & 3），从而丢失了原始数据中的精细结构。而 Wasserstein 模糊集保留了全部经验样本，其导出的最优分布呈现出多峰指数核的逐点最大值（$\max \exp(\dots)$），能够完美捕捉系统噪声的**多峰（Multi-modal）、非对称（Asymmetric）以及重尾**等复杂真实的特性。
*   **强大的有限样本外推保证（Finite-sample Guarantees）**：基于现代统计学中的测度集中不等式（Measure Concentration，参考 Wasserstein.pdf Theorem 3.4），Wasserstein 半径 $\varepsilon$ 可以被严格设计，从而在数学上保证真实的未知分布以 $1-\beta$ 的确切概率落入模糊集中。这为预测模型的泛化能力（Out-of-sample performance）提供了强有力的理论背书，有效克服随机规划中的过拟合现象（Optimizer's curse）。
*   **纯数据驱动与非参数化**：无需预先计算或假设噪声具有标称的均值与协方差结构，直接利用历史数据 $\mathcal{D}_N$ 驱动，是一种更为自然、普适的纯数据驱动（Data-driven）且非参数化（Non-parametric）的鲁棒建模范式。

---

## 2. 深入思考与后续研究方向 (Ideas for your paper)

1. **与核密度估计（KDE）的深刻联系：**
   经典的基于数据的经验预测通常会使用核密度估计 $\hat{p}(x) = \frac{1}{N} \sum \exp(-\lambda\|x-x_i\|)$。而在你的 Wasserstein DRPP 框架下，基于极小极大的鲁棒对数损失，推导出的最优分布是 $\max \exp(\dots)$。这在最优传输和密度估计理论中是一个非常前沿的现象（类似 Wasserstein MLE），可以直接作为你论文的一大核心亮点（Highlight），与 `DRPP.pdf` 只能得到单一高斯分布形成强烈对比！

2. **当 $\gamma_0(z) \neq 0$ 时的拓展：**
   如果标称函数 $f_k$ 本身存在不确定性，那么相当于分布的中心 $\hat{x}_{i}^{pred}$ 本身也在一个有界集合内浮动。这将在内层再嵌套一个针对中心点的最优化（类似于 Robust Optimization 中的不确定性集合）。由于问题结构优良，这有可能通过在指数核中引入 $\gamma_0$ 的平移或膨胀来解析求解。

3. **计算的可行性：**
   上述约束条件中涉及积分 $\int \max \exp(\dots) dx$。在低维状态空间下，这个积分可以很容易地通过数值或半解析方法计算。即使在高维下，也可以寻找它的易计算的 Upper Bound。这为你下一步设计 tractable 的算法提供了坚实的基础。

---

## 3. 严谨的定理与证明 (Formal Theorem and Proof)

基于上述直观思考，我们将 $\gamma_0(z) \equiv 0$ 时的单步优化过程提炼为如下定理并给出严谨证明。

**定理 1 (Wasserstein 单步 DRPP 的有限维凸规约与最优预测分布).** 
*给定当前状态控制对 $z_k = z$，假设系统状态演化函数已知且精确，即限制不确定性边界 $\gamma_0(z) \equiv 0$。记历史数据给出的经验噪声样本为 $\hat{w}_{k,i} = \hat{x}_{k+1,i} - \bar{f}_k(\hat{z}_{k,i})$，并定义预测“锚点”为 $\hat{x}_{i}^{pred} = \bar{f}_k(z) + \hat{w}_{k,i}$，其中 $i = 1,\dots,N$。假设损失函数 $-\log \hat{p}_k(x)$ 在 $\mathcal{X}$ 上满足下半连续且对数凹（即允许使用最优传输强对偶定理），那么单步 DRPP 泛函极小极大问题：*
$$ \max_{\hat{p}_k \in \mathcal{F}} \inf_{P_{x_{k+1}|z} \in \mathcal{I}_k^W(z)} \mathbb{E}_{P} \left[ \log \hat{p}_k(\pmb{x}_{k+1}) \right] $$
*等价于如下有限维凸优化问题：*
$$ \max_{\lambda \geq 0, \mathbf{s} \in \mathbb{R}^N} \quad -\lambda \varepsilon + \frac{1}{N} \sum_{i=1}^N s_i $$
$$ \text{s.t.} \quad \int_{\mathcal{X}} \max_{1 \leq i \leq N} \exp \left( s_i - \lambda \| x - \hat{x}_{i}^{pred} \| \right) dx \leq 1 $$
*且取得最优解 $(\lambda^*, \mathbf{s}^*)$ 时，最坏情况下的最优预测概率密度函数（Worst-case Optimal Predictive PDF）唯一确定为多峰指数核函数的逐点最大值：*
$$ \hat{p}_k^*(x) = \max_{1 \leq i \leq N} \exp \left( s_i^* - \lambda^* \| x - \hat{x}_{i}^{pred} \| \right) $$

**证明.** 

**步骤1：内层分布鲁棒优化问题的对偶转换**
当 $\gamma_0(z) \equiv 0$ 时，下一时刻状态严格满足 $\pmb{x}_{k+1} = \bar{f}_k(z) + \pmb{w}_k$。根据模糊集 $\mathcal{I}_k^W(z)$ 的定义，真实噪声分布 $\mathbb{P}(\pmb{w}_k)$ 处于以经验分布 $\hat{P}_{w,N} = \frac{1}{N}\sum_{i=1}^N \delta_{\hat{w}_{k,i}}$ 为中心、半径为 $\varepsilon$ 的 Wasserstein 球 $\mathbb{B}_\varepsilon(\hat{P}_{w,N})$ 中。
给定一个固定的预测分布 $\hat{p}_k \in \mathcal{F}$，大自然（Adversary）面临的内层极小化问题为：
$$ \inf_{P_w \in \mathbb{B}_\varepsilon(\hat{P}_{w,N})} \mathbb{E}_{P_w} \left[ \log \hat{p}_k(\bar{f}_k(z) + w_k) \right] = - \sup_{P_w \in \mathbb{B}_\varepsilon(\hat{P}_{w,N})} \mathbb{E}_{P_w} \left[ -\log \hat{p}_k(\bar{f}_k(z) + w_k) \right] $$
记损失函数 $\ell(w) = -\log \hat{p}_k(\bar{f}_k(z) + w)$。我们将详细展开如何利用 `Wasserstein.pdf` 中的 Theorem 4.2 对上述最坏情况期望（上确界）进行重构，该定理的核心思想是基于 Kantorovich-Rubinstein 最优传输强对偶性。

根据 Wasserstein 距离的定义，真实分布 $P_w$ 与离散经验分布 $\hat{P}_{w,N}$ 之间的 1-Wasserstein 距离受限于 $\varepsilon$，等价于存在一个联合分布（最优传输计划） $\Pi(w, \hat{w})$，其边缘分布分别为 $P_w$ 和 $\hat{P}_{w,N}$，使得传输代价的期望满足：
$$ \int_{\mathcal{W} \times \mathcal{W}} \| w - \hat{w} \| d\Pi(w, \hat{w}) \leq \varepsilon $$
由于经验分布 $\hat{P}_{w,N}$ 仅在离散的 $N$ 个样本点 $\{\hat{w}_{k,i}\}_{i=1}^N$ 上具有均等的概率质量 $\frac{1}{N}$，联合分布 $\Pi$ 可以完美分解为条件分布的求和：在给定源样本点 $\hat{w}_{k,i}$ 的条件下，目标点 $w$ 的条件分布记为 $P_i(w)$。此时，任意合法的传输计划相当于寻找 $N$ 个局部的条件概率测度 $P_i$，使得总代价约束变为：
$$ \frac{1}{N} \sum_{i=1}^N \int_{\mathcal{W}} \| w - \hat{w}_{k,i} \| dP_i(w) \leq \varepsilon $$
同时，目标泛函对应的期望则变为边缘分布 $P_w$ 上的积分，即 $\frac{1}{N} \sum_{i=1}^N \int_{\mathcal{W}} \ell(w) dP_i(w)$。

我们将上述带有传输代价约束的无穷维优化问题写成拉格朗日形式。引入惩罚代价越界行为的非负对偶乘子（拉格朗日乘子） $\lambda \geq 0$。根据最优传输理论中的强对偶定理（由于分布空间 $\mathcal{P}(\mathcal{W})$ 是凸集且目标泛函与约束均为线性积分），我们可以交换极小化（对 $\lambda$）与极大化（对分布 $P_i$）的顺序，从而得到无约束形式：
$$ \sup_{P_w \in \mathbb{B}_\varepsilon} \mathbb{E}_{P_w} [\ell(w)] = \inf_{\lambda \geq 0} \sup_{P_i \in \mathcal{P}(\mathcal{W})} \left\{ \frac{1}{N} \sum_{i=1}^N \int_{\mathcal{W}} \ell(w) dP_i(w) - \lambda \left( \frac{1}{N} \sum_{i=1}^N \int_{\mathcal{W}} \| w - \hat{w}_{k,i} \| dP_i(w) - \varepsilon \right) \right\} $$
重新整理上式，将其解耦为 $N$ 个独立的条件分布优化问题：
$$ = \inf_{\lambda \geq 0} \left\{ \lambda \varepsilon + \frac{1}{N} \sum_{i=1}^N \sup_{P_i \in \mathcal{P}(\mathcal{W})} \int_{\mathcal{W}} \left( \ell(w) - \lambda \| w - \hat{w}_{k,i} \| \right) dP_i(w) \right\} $$
在最内层的优化中，对于每一个给定的源样本 $i$，我们需要选择一个最优的条件概率测度 $P_i$ 来最大化该积分。显然，将被积函数的最大值点赋予全概率（即选择集中在最劣点上的 Dirac 测度 $\delta$）即可达到积分的上确界。因此，无穷维概率分布空间上的上确界被巧妙地规约为定义域空间（状态空间）上的点态上确界：
$$ \sup_{P_i \in \mathcal{P}(\mathcal{W})} \int_{\mathcal{W}} \left( \ell(w) - \lambda \| w - \hat{w}_{k,i} \| \right) dP_i(w) = \sup_{w} \left( \ell(w) - \lambda \| w - \hat{w}_{k,i} \| \right) $$
通过这步严密的等价推导，我们得到了 Theorem 4.2 的最终结论，原分布域的上确界等价于如下有限维（只有标量 $\lambda$ 和单变量 $w$）对偶问题：
$$ \sup_{P_w \in \mathbb{B}_\varepsilon} \mathbb{E}_{P_w} [\ell(w)] = \inf_{\lambda \geq 0} \left\{ \lambda \varepsilon + \frac{1}{N} \sum_{i=1}^N \sup_{w} \left( \ell(w) - \lambda \| w - \hat{w}_{k,i} \| \right) \right\} $$

最后，我们在方程两边同时取负号转化为下确界，并将变量从噪声域 $w$ 映射到状态空间 $x = \bar{f}_k(z) + w$。由于绝对距离的平移不变性 $\| w - \hat{w}_{k,i} \| = \| x - (\bar{f}_k(z) + \hat{w}_{k,i}) \| = \| x - \hat{x}_{i}^{pred} \|$，原内层极小化问题被等价转化为：
$$ \sup_{\lambda \geq 0} \left\{ -\lambda \varepsilon + \frac{1}{N} \sum_{i=1}^N \inf_{x \in \mathcal{X}} \left( \log \hat{p}_k(x) + \lambda \| x - \hat{x}_{i}^{pred} \| \right) \right\} $$

**步骤2：极大极小问题的重构与辅助变量引入**
将步骤1得到的对偶等价形式代回原极小极大问题中，原问题转化为同时对 $\hat{p}_k$ 和 $\lambda$ 求解上确界：
$$ \sup_{\hat{p}_k \in \mathcal{F}, \lambda \geq 0} \left\{ -\lambda \varepsilon + \frac{1}{N} \sum_{i=1}^N \inf_{x \in \mathcal{X}} \left( \log \hat{p}_k(x) + \lambda \| x - \hat{x}_{i}^{pred} \| \right) \right\} $$
为了消除目标函数中的非平滑下确界（$\inf$）操作，对每一个经验样本 $i=1,\dots,N$ 引入实数辅助变量 $s_i \in \mathbb{R}$，满足：
$$ s_i \leq \inf_{x \in \mathcal{X}} \left( \log \hat{p}_k(x) + \lambda \| x - \hat{x}_{i}^{pred} \| \right) $$
该不等式等价于全局成立的下界约束：
$$ \log \hat{p}_k(x) \geq s_i - \lambda \| x - \hat{x}_{i}^{pred} \|, \quad \forall x \in \mathcal{X}, \forall i \in \{1,\dots,N\} $$
整合所有 $i$，等价于：
$$ \log \hat{p}_k(x) \geq \max_{1 \leq i \leq N} \left( s_i - \lambda \| x - \hat{x}_{i}^{pred} \| \right) \implies \hat{p}_k(x) \geq \max_{1 \leq i \leq N} \exp \left( s_i - \lambda \| x - \hat{x}_{i}^{pred} \| \right) $$

**步骤3：预测分布 $\hat{p}_k(x)$ 的解析最优结构与紧致性**
此时优化问题化为：
$$ \max_{\hat{p}_k \in \mathcal{F}, \lambda \geq 0, \mathbf{s} \in \mathbb{R}^N} \quad -\lambda \varepsilon + \frac{1}{N} \sum_{i=1}^N s_i $$
$$ \text{s.t.} \quad \hat{p}_k(x) \geq \max_{1 \leq i \leq N} \exp \left( s_i - \lambda \| x - \hat{x}_{i}^{pred} \| \right), \quad \int_{\mathcal{X}} \hat{p}_k(x) dx = 1 $$
注意到，目标函数是关于 $s_i$ 的严格单调递增函数。假设在某最优解下存在一个正测度集合使得 $\hat{p}_k^*(x) > \max_{i} \exp \left( s_i^* - \lambda^* \| x - \hat{x}_{i}^{pred} \| \right)$，那么 $\int_{\mathcal{X}} \max_{i} \exp(\dots) dx < 1$。这意味着我们可以对所有的 $s_i^*$ 增加一个充分小的正数 $\delta > 0$，仍能构造出合法的 $\hat{p}_k$ 使得积分等于1，这将导致目标函数值严格增加 $\delta$，从而与最优性矛盾。
因此，在取得最大值时，上述下界约束必然处处取等（几乎处处成立）：
$$ \hat{p}_k^*(x) = \max_{1 \leq i \leq N} \exp \left( s_i^* - \lambda^* \| x - \hat{x}_{i}^{pred} \| \right) $$
此时，关于无穷维泛函空间 $\mathcal{F}$ 的优化被完美降维，并可以由等式关系直接代入积分约束中。为了放宽优化难度，等式约束可安全写为不等式 $\int_{\mathcal{X}} \hat{p}_k(x) dx \leq 1$ （由于目标函数驱动 $s_i$ 增大，约束在最优点必定 bind）。

**步骤4：证明有限维问题为凸优化问题**
最后，我们证明规约后的有限维问题是凸优化问题。
目标函数 $J(\lambda, \mathbf{s}) = -\lambda \varepsilon + \frac{1}{N} \sum s_i$ 显然是仿射函数（因而既凸又凹）。
约束条件为 $g(\lambda, \mathbf{s}) := \int_{\mathcal{X}} \max_{1 \leq i \leq N} \exp \left( s_i - \lambda \| x - \hat{x}_{i}^{pred} \| \right) dx \leq 1$。
对于给定的 $x$，函数 $h_i(\lambda, \mathbf{s}) = s_i - \lambda \| x - \hat{x}_{i}^{pred} \|$ 是关于 $(\lambda, \mathbf{s})$ 的仿射函数。因为指数函数 $\exp(\cdot)$ 是递增且凸的，故 $\exp(h_i(\lambda, \mathbf{s}))$ 是凸函数（更是对数凸函数）。
有限个凸函数的逐点最大值 $\max_i \exp(h_i)$ 依然是凸函数。
因为非负凸函数的积分运算保持凸性，积分 $g(\lambda, \mathbf{s})$ 关于 $(\lambda, \mathbf{s})$ 是凸函数。
因此，我们要最大化一个仿射目标函数（等价于最小化其负值），且约束集由凸函数的不等式下水平集定义，这是一个标准的**凸优化问题**。该性质保证了我们能够使用如内点法等成熟算法高效求解全局最优，且不受局部极值的困扰。证毕。

---

## 4. 算法设计：Wasserstein Noise-DRPP (W-Noise-DRPP)

参考《DRPP.pdf》中针对状态演化函数无模糊性（即 $\gamma_0(z) \equiv 0$ 或 $\nu_k = \bar{\nu}_k$）时提出的 Noise-DRPP 算法，我们在 Wasserstein 框架下设计其对应的 W-Noise-DRPP 算法。

### 4.1 理论依据与对比
在基于矩的 Noise-DRPP 中（详见 DRPP.pdf Theorem 3），由于只约束了噪声的前二阶矩，大自然（Adversary）为了最大化预测者的对数损失，总是会选择最坏情况的高斯分布。因此，预测者针锋相对的最优策略也是输出一个方差被膨胀的单峰高斯预测分布 $\mathcal{N}(\bar{f}_k(z) + \bar{\mu}_k, \gamma_2\bar{\Sigma}_k)$。

然而，在 W-Noise-DRPP 中，模糊集约束在 Wasserstein 球内。根据前文**定理 1**的推导，当系统状态函数完全已知时，我们将基于经验数据集 $\mathcal{D}_N$ 中解算出的历史噪声 $\hat{w}_{k,i}$ 作为基准，将预测分布锚定在 $\hat{x}_{i}^{pred} = \bar{f}_k(z_k) + \hat{w}_{k,i}$。通过求解一个有限维凸优化问题，得到的最优预测分布不再是单一高斯，而是多个指数核函数的最大值包络。这赋予了 W-Noise-DRPP 在保留非高斯、多峰噪声特征上的绝对优势。

### 4.2 W-Noise-DRPP 算法流程

以下是结合 Wasserstein DRPP 特性设计的算法伪代码：

**Algorithm: Wasserstein Noise-DRPP (W-Noise-DRPP)**

**Input:** 预测时间步长 $T$，Wasserstein 半径 $\varepsilon(\beta, N)$，历史噪声经验样本 $\{\hat{w}_{k,i}\}_{i=1}^N$，控制策略 $\pi_{0:T-1}$，初始状态 $x_0$。

1: 初始化对数得分累加器 $Score \leftarrow 0$。

2: **for** $k = 0, \dots, T-1$ **do**

3: $\quad$ 观测当前状态 $x_k$，通过控制策略计算输入 $u_k \sim \pi_k(\cdot | x_k)$。

4: $\quad$ 定义当前状态-控制对 $z_k \leftarrow (x_k, u_k)$。

5: $\quad$ 计算标称状态演化：$\bar{x}_{k+1} = \bar{f}_k(z_k)$。

6: $\quad$ 基于历史经验噪声生成 $N$ 个预测锚点：$\hat{x}_{i}^{pred} = \bar{x}_{k+1} + \hat{w}_{k,i}, \quad \forall i \in \{1,\dots,N\}$。

7: $\quad$ 求解定理 1 中的有限维凸优化问题，获取最优参数 $(\lambda^*, \mathbf{s}^*)$：

$$ (\lambda^*, \mathbf{s}^*) = \arg\max_{\lambda \geq 0, \mathbf{s}} \left\{ -\lambda \varepsilon + \frac{1}{N} \sum_{i=1}^N s_i \right\} $$
$$ \text{s.t.} \quad \int_{\mathcal{X}} \max_{1 \leq i \leq N} \exp \left( s_i - \lambda \| x - \hat{x}_{i}^{pred} \| \right) dx \leq 1 $$

8: $\quad$ W-Noise-DRPP 输出多峰指数最优预测 PDF：
$$ \hat{p}_k^*(x) = \max_{1 \leq i \leq N} \exp \left( s_i^* - \lambda^* \| x - \hat{x}_{i}^{pred} \| \right) $$

9: $\quad$ 真实随机动力系统 (SDS) 演化生成下一时刻真实状态 $x_{k+1}$。

10: $\quad$ 更新得分：$Score \leftarrow Score + \log \hat{p}_k^*(x_{k+1})$。

11: **end for**

**Output:** 状态轨迹 $x_{0:T}$，预测分布序列 $\hat{p}_{0:T-1}^*$，累积得分 $Score$。

### 4.3 计算实现的可行性分析 (Tractability)

W-Noise-DRPP 算法的核心瓶颈在于**步骤 7** 中带有积分不等式约束的凸优化。为了使其在工程代码中高效运行，我们可以引入以下技巧：

1. **Log-Sum-Exp 平滑近似（Smooth Approximation）**：
   目标函数中的非平滑逐点最大值算子 $\max_i$ 可以使用 Log-Sum-Exp 函数进行平滑化下界逼近：
   $$ \max_{i} A_i \approx \frac{1}{\tau} \log \left( \sum_{i=1}^N \exp(\tau A_i) \right) $$
   这能让约束变为更易求导的光滑凸函数。
   
2. **积分的蒙特卡洛/数值离散化（Numerical Integration）**：
   在实际控制系统中，状态空间往往是有界的。可以通过在 $\mathcal{X}$ 上进行网格离散化或重要性采样，将连续积分约束转化为有限和约束。
   
3. **拉格朗日对偶与极速求解（Fast Solvers）**：
   由于该优化仅涉及标量 $\lambda$ 和 $N$ 维向量 $\mathbf{s}$，维度较小（$N+1$ 维），使用现有的内点法求解器（如 CVXPY 配合 MOSEK/SCS）可以做到毫秒级求解，完全能够满足模型预测控制（MPC）等在线控制任务的实时性要求。

## 5. 原始多步 DRPP 问题的严格定义 ($\gamma_0(z) \neq 0$)

在实际的非线性随机动力系统中，名义状态演化函数 $\bar{f}_k(z)$ 往往是不准确的。原始的多步 DRPP 并没有强制 $\gamma_0(z) \equiv 0$，而是允许系统存在由 $\gamma_0(z)$ 有界限制的认知不确定性（Epistemic Uncertainty），同时噪声存在由 Wasserstein 球半径 $\varepsilon$ 限制的偶然不确定性（Aleatoric Uncertainty）。

结合 `DRPP.pdf` 中的多步目标与 `Wasserstein.pdf` 中的数据驱动度量，原始的无额外限制的多步 DRPP 问题可以严格表述为如下的泛函极小极大问题：
$$
\max_{\mathcal{F} \in \mathfrak{F}} \inf_{\mathcal{P} \in \mathfrak{P}} \mathbb{E}_{\mathcal{P}} \left[ \sum_{k=0}^{T-1} \mathcal{L}(\mathcal{F}_k(z_k), \pmb{x}_{k+1}) \middle| z_0 \right]
$$
其中，系统真实联合分布 $\mathcal{P}$ 满足动态约束 $\pmb{x}_{k+1} = f_k(z_k) + \pmb{w}_k$，且其单步条件概率测度处于完整的混合模糊集中：
$$
\mathcal{I}_{k}^{W}(z) := \left\{ P_{\pmb{x}_{k+1} | z} \in \mathcal{P}(\mathcal{X}) \; \middle| \;
\begin{aligned}
& \pmb{x}_{k+1} = f_{k}(z) + \pmb{w}_{k}, \\
& \|f_{k}(z) - \bar{f}_{k}(z)\|_2^2 \leq \gamma_{0}(z), \\
& P_{\pmb{w}_k} \in \mathbb{B}_{\varepsilon(\beta,N)} \left( \hat{P}_{w,N}^{\bar{f}_k} \right)
\end{aligned}
\right\}
$$
这里为了保证历史经验分布的可计算性，我们将 Wasserstein 球的中心锚定在基于名义模型计算出的残差 $\hat{P}_{w,N}^{\bar{f}_k} = \frac{1}{N} \sum_{i=1}^N \delta_{\bar{w}_{k,i}}$ 上，其中 $\bar{w}_{k,i} = \hat{x}_{k+1,i} - \bar{f}_k(\hat{z}_{k,i})$。
利用动态规划原理，该多步问题在第 $k$ 步的核心可归结为求解包含完整不确定性的鲁棒 Bellman 方程：
$$
V_k^*(z) = \max_{\hat{p}_k} \inf_{\|\nu_k - \bar{\nu}_k\|_2^2 \leq \gamma_0(z)} \inf_{P_w \in \mathbb{B}_\varepsilon} \mathbb{E}_{P_w} \left[ \log \hat{p}_k(\nu_k + w) + V_{k+1}^*(\nu_k + w) \right]
$$
其中 $\nu_k = f_k(z)$ 是未知的真实系统函数在当前状态的取值，$\bar{\nu}_k = \bar{f}_k(z)$ 为名义预测中心。

## 6. 内层对偶与简化：向欧几里得空间的规约

在 $\gamma_0(z) \neq 0$ 时，自然界（Adversary）拥有双重控制权：不仅能移动整体状态分布的均值（选择最劣的 $\nu_k$），还能恶化数据的形状（选择最劣的 $P_w$）。这是一个无限维嵌套最优化问题。
根据 `Wasserstein.pdf` 中的 Theorem 4.2，我们可以首先将内层的 Wasserstein 极小化转化为关于变量 $\lambda \geq 0$ 的欧几里得空间对偶形式。令 $U(x) = \log \hat{p}_k(x) + V_{k+1}^*(x)$，大自然的最小化问题化为：
$$
\sup_{\lambda \geq 0} \left\{ -\lambda \varepsilon + \inf_{\|\Delta \nu\|_2 \leq \sqrt{\gamma_0}} \frac{1}{N} \sum_{i=1}^N \inf_{x_i} \left( U(x_i) + \lambda \|x_i - (\bar{\nu}_k + \bar{w}_{k,i} + \Delta \nu)\| \right) \right\}
$$
其中 $\Delta \nu = \nu_k - \bar{\nu}_k$，且 $\hat{x}_{i}^{pred} = \bar{\nu}_k + \bar{w}_{k,i}$ 是名义模型给出的第 $i$ 个预测锚点。

**保守近似与极小极大交换 (Conservative Approximation & Minimax Swap)**
上述公式中，大自然选择一个全局统一的漂移 $\Delta \nu$ 作用于所有样本。为了实现向欧几里得空间的完全规约并得到 Tractable 的凸优化模型，我们引入一个合理的**保守近似（下界放缩）**：允许大自然对每个锚点采用独立的恶化漂移。这在数学上等价于利用不等式 $\inf \sum \geq \sum \inf$，将 $\Delta \nu$ 的极小化移入求和号与 $x_i$ 的极小化内部：
$$
\inf_{\|\Delta \nu\|_2 \leq \sqrt{\gamma_0}} \inf_{x_i} \left( U(x_i) + \lambda \|x_i - \hat{x}_i^{pred} - \Delta \nu\| \right) = \inf_{x_i} \left( U(x_i) + \lambda \inf_{\|\Delta \nu\|_2 \leq \sqrt{\gamma_0}} \|x_i - \hat{x}_i^{pred} - \Delta \nu\| \right)
$$
注意到，点 $x_i - \hat{x}_i^{pred}$ 到原点半径为 $\sqrt{\gamma_0}$ 的球的最短距离有着优美的解析闭式解：
$$
\inf_{\|\Delta \nu\|_2 \leq \sqrt{\gamma_0}} \|x_i - \hat{x}_i^{pred} - \Delta \nu\| = \max \left( 0, \|x_i - \hat{x}_i^{pred}\| - \sqrt{\gamma_0} \right)
$$

**规约后的有限维优化问题 (Reduced Finite-Dimensional Problem)**
利用上述闭式解，并引入辅助变量 $\mathbf{s} \in \mathbb{R}^N$，我们可以将原始极小极大问题转化为欧几里得空间中的有限维凸优化问题。令 $V_{k+1}^*(x)$ 暂时被吸收或为0（考察单步），问题化为：
$$
\max_{\lambda \geq 0, \mathbf{s} \in \mathbb{R}^N} \quad -\lambda \varepsilon + \frac{1}{N} \sum_{i=1}^N s_i
$$
$$
\text{s.t.} \quad \int_{\mathcal{X}} \max_{1 \leq i \leq N} \exp \left( s_i - \lambda \max \left(0, \|x - \hat{x}_i^{pred}\| - \sqrt{\gamma_0} \right) \right) dx \leq 1
$$

此时，我们得到了限制条件解除后，**最优鲁棒概率预测密度的显式解析结构**：
$$
\hat{p}_k^*(x) = \max_{1 \leq i \leq N} \exp \left( s_i^* - \lambda^* \max \left(0, \|x - \hat{x}_i^{pred}\| - \sqrt{\gamma_0} \right) \right)
$$

### 核心理论洞见 (Theoretical Insights)："平顶"核函数 vs. 半径膨胀
这个解析结果提供了一个非常直观且深刻的理论洞察。
如果在传统思维下处理函数不确定性，人们可能会简单地将由于 $f_k$ 不准确带来的扰动 $\sqrt{\gamma_0}$ 粗暴地加到 Wasserstein 距离的半径中（即令新的 $\varepsilon' = \varepsilon + \sqrt{\gamma_0}$），这会退化为使用传统的拉普拉斯核 $\exp(-\lambda\|x - \hat{x}_i^{pred}\|)$ 并伴随一个更小的 $\lambda$。
然而，我们推导出的最优分布使用的是一种**带有“平顶”（Flat-top Plateau）的指数核**（即当 $\|x - \hat{x}_i^{pred}\| \leq \sqrt{\gamma_0}$ 时，核函数值保持恒定不衰减）。这证明了：
1. **分离不确定性本质**：Wasserstein 度量 $\varepsilon$ 应对的是分布的拖尾和形状扰动（Aleatoric），而 $\gamma_0$ 应对的是分布中心的绝对位移（Epistemic）。平顶核完美融合了这两者——在 $\sqrt{\gamma_0}$ 范围内预测分布不衰减以完全吸收中心漂移，在超出该范围后以 $\lambda$ 的速率衰减以抵抗分布扰动。
2. **更低的保守性**：这种结构比单纯的“膨胀半径”更紧致（Less Conservative），能为控制系统提供在保证鲁棒性的前提下，尽可能集中的置信域（Confidence Regions）。

## 7. 完全严谨的多步 DRPP：基于真实系统函数 $f$ 的耦合模糊集

正如前文所提及的，在第5节中为了计算与表达的便利性，我们将 Wasserstein 球的中心锚定在了基于标称函数 $\bar{f}_k$ 计算出的名义残差 $\hat{P}_{w,N}^{\bar{f}_k}$ 上。然而，从严格的物理和统计意义上讲，真实的数据生成过程是由未知的真实动力学函数 $f_k$ 驱动的。因此，最严谨的定义要求 Wasserstein 球的经验中心也必须基于未知的 $f_k$。

### 7.1 严格耦合的模糊集定义
我们将第5节中的模糊集升级为如下的严格耦合模糊集（Strictly Coupled Ambiguity Set）：
$$
\mathcal{I}_{k}^{W, strict}(z) := \left\{ P_{\pmb{x}_{k+1} | z} \in \mathcal{P}(\mathcal{X}) \; \middle| \;
\begin{aligned}
& \pmb{x}_{k+1} = f_{k}(z) + \pmb{w}_{k}, \\
& \|f_{k}(\cdot) - \bar{f}_{k}(\cdot)\|_2^2 \leq \gamma_{0}(\cdot), \quad (\text{对所有状态空间成立}) \\
& P_{\pmb{w}_k} \in \mathbb{B}_{\varepsilon(\beta,N)} \left( \hat{P}_{w,N}^{f_k} \right)
\end{aligned}
\right\}
$$
其中，经验分布中心 $\hat{P}_{w,N}^{f_k} = \frac{1}{N} \sum_{i=1}^N \delta_{\tilde{w}_{k,i}}$ 由**未知的真实历史残差**构成，即 $\tilde{w}_{k,i} = \hat{x}_{k+1,i} - f_k(\hat{z}_{k,i})$。
这就带来了一个更加复杂的挑战：大自然（Adversary）不仅可以通过选择 $f_k(z)$ 来改变当前的预测漂移，还可以通过改变 $f_k(\hat{z}_{k,i})$ 来**篡改（Shift）经验分布的中心**。此时，中心锚点与系统不确定性发生了深度的耦合。

### 7.2 内层对偶推导与”双重认知不确定性”

在极小极大框架下，大自然（Adversary）为了最小化我们的预测得分，会同时选择最劣的当前演化值 $\nu_k = f_k(z)$ 和最劣的历史评估值 $f_{k,i} = f_k(\hat{z}_{k,i})$。
令当前时刻的认知偏差为 $\Delta \nu = f_k(z) - \bar{f}_k(z)$，且 $\|\Delta \nu\|_2 \leq \sqrt{\gamma_0(z)}$；
令历史样本 $i$ 处的认知偏差为 $\Delta f_i = f_k(\hat{z}_{k,i}) - \bar{f}_k(\hat{z}_{k,i})$，且 $\|\Delta f_i\|_2 \leq \sqrt{\gamma_0(\hat{z}_{k,i})}$。

#### 步骤 A：真实预测锚点的解构

对于给定的历史数据对 $(\hat{z}_{k,i}, \hat{x}_{k+1,i})$，严谨定义下第 $i$ 个**真实预测锚点**（Strict Anchor Point）为 $\hat{x}_{i}^{strict} = f_k(z) + \tilde{w}_{k,i}$，其中 $\tilde{w}_{k,i} = \hat{x}_{k+1,i} - f_k(\hat{z}_{k,i})$ 是基于真实函数 $f_k$ 解算出的第 $i$ 个噪声样本。将其展开：
$$
\begin{aligned}
\hat{x}_{i}^{strict} &= f_k(z) + \tilde{w}_{k,i} \\
&= f_k(z) + \hat{x}_{k+1,i} - f_k(\hat{z}_{k,i}) \\
&= (\bar{f}_k(z) + \Delta \nu) + \hat{x}_{k+1,i} - (\bar{f}_k(\hat{z}_{k,i}) + \Delta f_i) \\
&= \underbrace{(\bar{f}_k(z) + \hat{x}_{k+1,i} - \bar{f}_k(\hat{z}_{k,i}))}_{\text{名义预测锚点 } \hat{x}_i^{pred}} + \underbrace{\Delta \nu - \Delta f_i}_{\text{耦合认知偏差}}
\end{aligned}
$$
这揭示了一个关键的理论现象：真实锚点相对于名义锚点的偏移 $\Delta \nu - \Delta f_i$ 是两个独立认知偏差的**叠加**——当前时刻的模型偏差 $\Delta \nu$ 将锚点向一个方向推移，而历史时刻的模型偏差 $\Delta f_i$ 则将锚点向**相反方向**推移（因为 $\Delta f_i$ 以负号出现）。大自然可以协调这两者，使锚点向最不利的方向发生最大偏移。

#### 步骤 B：对固定 $f_k$ 施加 Wasserstein 对偶

对于给定的系统函数 $f_k$（即固定 $\Delta \nu$ 和所有 $\Delta f_i$），Wasserstein 球的中心和锚点均已确定。此时，完全可以套用第3节定理 1 证明中步骤 1 的 Wasserstein 对偶转换。令 $U(x) = \log \hat{p}_k(x) + V_{k+1}^*(x)$（其中 $V_{k+1}^*$ 是后续步的值函数，单步时 $V_{k+1}^* \equiv 0$），经验锚点为 $\hat{x}_i^{strict} = \hat{x}_i^{pred} + \Delta\nu - \Delta f_i$。对偶结果为：
$$
\inf_{P_w \in \mathbb{B}_\varepsilon(\hat{P}_{w,N}^{f_k})} \mathbb{E}_{P_w} [U(\pmb{x}_{k+1})] = \sup_{\lambda \geq 0} \left\{ -\lambda\varepsilon + \frac{1}{N}\sum_{i=1}^N \inf_{x_i \in \mathcal{X}} \left( U(x_i) + \lambda \|x_i - \hat{x}_i^{strict}\| \right) \right\}
$$
将 $\hat{x}_i^{strict}$ 的表达式代入，得到关于 $\Delta\nu$ 和 $\{\Delta f_i\}$ 的显式依赖：
$$
= \sup_{\lambda \geq 0} \left\{ -\lambda\varepsilon + \frac{1}{N}\sum_{i=1}^N \inf_{x_i} \left( U(x_i) + \lambda \|x_i - \hat{x}_i^{pred} - \Delta\nu + \Delta f_i\| \right) \right\}
$$

#### 步骤 C：大自然的全局优化与保守近似

现在，大自然还需要选择最劣的 $f_k$（即选择 $\Delta\nu$ 和 $\{\Delta f_i\}$）来最小化上述对偶表达式。完整的鲁棒 Bellman 方程内层化为：
$$
\inf_{\substack{\|\Delta\nu\| \leq \sqrt{\gamma_0(z)} \\ \|\Delta f_i\| \leq \sqrt{\gamma_0(\hat{z}_{k,i})}}} \sup_{\lambda \geq 0} \left\{ -\lambda\varepsilon + \frac{1}{N}\sum_{i=1}^N \inf_{x_i} \left( U(x_i) + \lambda \|x_i - \hat{x}_i^{pred} - \Delta\nu + \Delta f_i\| \right) \right\}
$$

**困难所在：** 上式中，大自然必须选择一个全局统一的 $\Delta\nu$ 同时作用于所有 $N$ 个样本。这使得 $\Delta\nu$ 与求和号耦合，导致问题难以分解。

**保守近似（Conservative Approximation）：** 完全类比第 6 节的处理方式，我们允许大自然对每个样本 $i$ 独立地选择最劣的偏移 $\Delta\nu_i$（而非共用一个全局 $\Delta\nu$）。由于给予大自然更多的自由度只会使结果更悲观（对预测者更不利），数学上对应不等式：
$$
\inf_{\text{共用 } \Delta\nu} \sum_{i=1}^N g_i(\Delta\nu) \;\geq\; \sum_{i=1}^N \inf_{\text{独立 } \Delta\nu_i} g_i(\Delta\nu_i)
$$
这是因为左边要求所有 $g_i$ 共享同一个 $\Delta\nu$ 作折中，而右边允许每项各自选择最优的 $\Delta\nu_i$，后者优化空间更大，所以每项的最小值之和更小，即右边 $\leq$ 左边。

施加此近似后，问题分解为 $N$ 个独立的子问题。对于第 $i$ 个子问题：
$$
\inf_{\|\Delta\nu_i\| \leq \sqrt{\gamma_0(z)},\; \|\Delta f_i\| \leq \sqrt{\gamma_0(\hat{z}_{k,i})}} \inf_{x_i} \left( U(x_i) + \lambda \|x_i - \hat{x}_i^{pred} - \Delta\nu_i + \Delta f_i\| \right)
$$

#### 步骤 D：双认知偏差下的最短距离闭式解

上述子问题的核心在于计算：对固定的 $x_i$，大自然通过协调 $\Delta\nu_i$ 和 $\Delta f_i$ 使得距离 $\|x_i - \hat{x}_i^{pred} - \Delta\nu_i + \Delta f_i\|$ **尽可能小**（因为 $\lambda \geq 0$，距离越小则目标函数越小，对预测者越不利）。

> **关键辨析：** 这里大自然是在**最小化**距离，而非最大化。直觉上：大自然希望将锚点 $\hat{x}_i^{strict}$ 尽可能地”拉近”到使 $U(x_i)$ 很低的区域——让最坏情况的真实数据落在预测分布 $\hat{p}_k$ 赋予概率很低的地方。在对偶形式中，这体现为压缩距离惩罚项 $\lambda\|\cdot\|$，使大自然付出更小的传输代价就能把概率质量搬到预测盲区。

令 $d = x_i - \hat{x}_i^{pred}$，并引入复合偏移 $\delta = \Delta\nu_i - \Delta f_i$。问题化为：
$$
\inf_{\delta \in \mathcal{S}} \|d - \delta\|
$$
其中可行域 $\mathcal{S} = \{\Delta\nu_i - \Delta f_i : \|\Delta\nu_i\| \leq r_1,\; \|\Delta f_i\| \leq r_2\}$，$r_1 = \sqrt{\gamma_0(z)}$，$r_2 = \sqrt{\gamma_0(\hat{z}_{k,i})}$。

**Minkowski 和引理：** 集合 $\mathcal{S}$ 恰好是两个闭球的 Minkowski 和。具体地，$\Delta\nu_i$ 取遍半径为 $r_1$ 的球 $\mathbb{B}_{r_1}$，$-\Delta f_i$ 取遍半径为 $r_2$ 的球 $\mathbb{B}_{r_2}$（因为 $\|\Delta f_i\| \leq r_2$ 意味着 $\|-\Delta f_i\| \leq r_2$），所以：
$$
\mathcal{S} = \mathbb{B}_{r_1} + \mathbb{B}_{r_2} = \mathbb{B}_{r_1 + r_2}
$$
即 $\mathcal{S}$ 本身是一个以原点为中心、半径为 $R_i := r_1 + r_2 = \sqrt{\gamma_0(z)} + \sqrt{\gamma_0(\hat{z}_{k,i})}$ 的闭球。

因此，$\inf_{\delta \in \mathcal{S}} \|d - \delta\|$ 就是求点 $d$ 到闭球 $\mathbb{B}_{R_i}$ 的最短距离，其解析闭式解为：
$$
\inf_{\|\Delta\nu_i\| \leq \sqrt{\gamma_0(z)},\; \|\Delta f_i\| \leq \sqrt{\gamma_0(\hat{z}_{k,i})}} \|x_i - \hat{x}_i^{pred} - \Delta\nu_i + \Delta f_i\| = \max\!\left(0,\; \|x_i - \hat{x}_i^{pred}\| - R_i \right)
$$
其中 $R_i = \sqrt{\gamma_0(z)} + \sqrt{\gamma_0(\hat{z}_{k,i})}$。

> **几何直觉：** 当 $x_i$ 距离名义锚点 $\hat{x}_i^{pred}$ 在 $R_i$ 范围以内时，大自然可以通过协调两个偏差 $\Delta\nu_i$ 和 $\Delta f_i$ 将真实锚点完全搬到 $x_i$ 所在位置（使距离为零），从而完全消除传输代价惩罚。当距离超过 $R_i$ 时，大自然只能将锚点推到离 $x_i$ 最近的球面上，剩余距离 $\|x_i - \hat{x}_i^{pred}\| - R_i$ 成为不可消除的传输代价。

这正是”**双重认知不确定性的累加效应（Double Epistemic Uncertainty）**”的精确数学体现：两个独立的认知偏差球的 Minkowski 和使总”免费搬运”半径从第 6 节的 $\sqrt{\gamma_0(z)}$ 扩大为 $\sqrt{\gamma_0(z)} + \sqrt{\gamma_0(\hat{z}_{k,i})}$。

### 7.3 向欧几里得空间的完全规约：有限维凸优化定理

将步骤 D 的闭式解代回对偶问题，并执行与第 3 节定理 1 完全相同的辅助变量引入和紧致性论证（步骤 2-4），我们得到如下严格耦合模糊集下的有限维凸规约定理。

**定理 2（严格耦合 Wasserstein 单步 DRPP 的有限维凸规约与最优预测分布，保守近似）.**
*给定当前状态控制对 $z_k = z$，历史数据 $\mathcal{D}_N$。名义函数为 $\bar{f}_k$，模型不确定性上界为 $\gamma_0(\cdot)$。定义名义预测锚点 $\hat{x}_i^{pred} = \bar{f}_k(z) + \hat{x}_{k+1,i} - \bar{f}_k(\hat{z}_{k,i})$，以及第 $i$ 个样本的复合认知不确定性半径 $R_i = \sqrt{\gamma_0(z)} + \sqrt{\gamma_0(\hat{z}_{k,i})}$。那么，在保守近似（允许大自然对每个样本独立选择偏移）下，基于严格耦合模糊集 $\mathcal{I}_k^{W,strict}(z)$ 的单步 DRPP 泛函极小极大问题：*
$$
\max_{\hat{p}_k \in \mathcal{F}} \inf_{P_{\pmb{x}_{k+1}|z} \in \mathcal{I}_k^{W,strict}(z)} \mathbb{E}_{P} \left[\log \hat{p}_k(\pmb{x}_{k+1})\right]
$$
*的一个保守（对预测者悲观的）下界由如下欧几里得空间中的有限维凸优化问题给出：*
$$
\max_{\lambda \geq 0,\; \mathbf{s} \in \mathbb{R}^N} \quad -\lambda\varepsilon + \frac{1}{N}\sum_{i=1}^N s_i
$$
$$
\text{s.t.} \quad \int_{\mathcal{X}} \max_{1 \leq i \leq N} \exp\!\left( s_i - \lambda \max\!\left(0,\; \|x - \hat{x}_i^{pred}\| - R_i \right) \right) dx \leq 1
$$
*且取得最优解 $(\lambda^*, \mathbf{s}^*)$ 时，最优鲁棒预测概率密度函数为平顶指数核的逐点最大值：*
$$
\hat{p}_k^*(x) = \max_{1 \leq i \leq N} \exp\!\left( s_i^* - \lambda^* \max\!\left(0,\; \|x - \hat{x}_i^{pred}\| - \underbrace{\left(\sqrt{\gamma_0(z)} + \sqrt{\gamma_0(\hat{z}_{k,i})}\right)}_{R_i:\;\text{扩展的平顶半径}} \right) \right)
$$

**证明概要.** 基于步骤 B-D 的推导，保守近似后内层问题等价于：
$$
\sup_{\lambda \geq 0} \left\{ -\lambda\varepsilon + \frac{1}{N}\sum_{i=1}^N \inf_{x_i} \left( U(x_i) + \lambda\max(0, \|x_i - \hat{x}_i^{pred}\| - R_i) \right) \right\}
$$
将此代回外层 $\max_{\hat{p}_k}$，引入辅助变量 $s_i \leq \inf_{x_i}(\cdots)$，并利用与定理 1 步骤 3 完全相同的紧致性论证（目标函数关于 $s_i$ 严格单调递增，迫使下界约束处处取等），即可消去泛函空间 $\mathcal{F}$ 上的优化，完成从无穷维到有限维 $(\lambda, \mathbf{s}) \in \mathbb{R}^{N+1}$ 的规约。凸性证明与定理 1 步骤 4 完全平行：$\max(0, \|x - \hat{x}_i^{pred}\| - R_i)$ 作为 $x$ 的凸函数（截断范数），经仿射变换、指数复合、逐点最大值和积分后保持凸性。$\square$

**与定理 1 的对比：** 当 $\gamma_0(\cdot) \equiv 0$ 时，所有 $R_i = 0$，$\max(0, \|x - \hat{x}_i^{pred}\| - 0) = \|x - \hat{x}_i^{pred}\|$，定理 2 精确退化为定理 1。因此定理 2 是定理 1 在一般化模型不确定性设定下的推广。

### 7.4 核心理论洞见与物理解释

#### “平顶”核函数：两类不确定性的最优融合

在定理 2 的最优预测密度中，每个锚点 $\hat{x}_i^{pred}$ 对应的指数核函数具有如下分段结构：
$$
K_i(x) = \exp\!\left(s_i^* - \lambda^* \max(0, \|x - \hat{x}_i^{pred}\| - R_i)\right) = \begin{cases} e^{s_i^*}, & \text{若 } \|x - \hat{x}_i^{pred}\| \leq R_i \quad (\text{平顶区}) \\ e^{s_i^*} \cdot e^{-\lambda^*(\|x - \hat{x}_i^{pred}\| - R_i)}, & \text{若 } \|x - \hat{x}_i^{pred}\| > R_i \quad (\text{指数衰减区}) \end{cases}
$$

这种”平顶指数核”是理论推导的自然产物，而非人为设计。它优雅地将两类本质不同的不确定性**分工应对**：

1. **平顶区吸收认知不确定性（Epistemic）：** 在半径 $R_i = \sqrt{\gamma_0(z)} + \sqrt{\gamma_0(\hat{z}_{k,i})}$ 内，预测密度保持恒定不衰减。这反映了一个物理事实：大自然可以通过选择真实系统函数 $f_k$ 的值（在 $\gamma_0$ 允许的范围内），将真实锚点移动到名义锚点周围 $R_i$ 范围内的任意位置。既然我们无法区分这些位置中哪个是真实的，最优策略就是对它们赋予相同的概率密度——这就是平顶区。

2. **指数衰减区抵抗偶然不确定性（Aleatoric）：** 超出 $R_i$ 的区域，即使大自然用尽了系统函数的不确定性预算，仍然无法将锚点移动到 $x$ 附近。此时剩余的偏离只能由噪声分布的扰动（Wasserstein 球内的质量搬运）来实现，代价以 $\lambda^*$ 的速率线性增长，对应指数核的衰减。

#### 与”膨胀半径”粗暴近似的本质区别

如果在传统思维下处理函数不确定性，人们可能会简单地将 $R_i$ 加到 Wasserstein 半径中（令 $\varepsilon' = \varepsilon + R_i$），退化为使用普通拉普拉斯核 $\exp(-\lambda'\|x - \hat{x}_i^{pred}\|)$。然而平顶核严格优于这种粗暴近似：膨胀半径方法在所有距离上均匀”变软”（$\lambda'$ 更小），而平顶核在 $R_i$ 内保持完全坚挺、在 $R_i$ 外以原始速率 $\lambda^*$ 衰减，提供更紧致的置信域。

#### 样本自适应的不确定性量化

平顶半径 $R_i$ 是**样本依赖的（Sample-dependent）**：不同的历史数据点 $\hat{z}_{k,i}$ 处的模型不确定性 $\gamma_0(\hat{z}_{k,i})$ 可以不同。这意味着：
*   如果某个历史数据点采集自模型准确的区域（$\gamma_0(\hat{z}_{k,i}) \approx 0$），其对应锚点可信度高，平顶区小，预测密度集中。
*   如果某个历史数据点采集自模型极不准确的区域（$\gamma_0(\hat{z}_{k,i})$ 很大），其对应锚点不可信，平顶区大，预测密度扩散。

这一机制**自动防止**控制系统对不可靠的历史残差产生过度拟合或过度自信，完美契合安全控制中”在不确定区域降低置信度”的直觉原则。

#### 第 5、6 节的定位与退化关系

本节的严谨推导也明确了第 5、6 节简化方法的适用条件：当假设历史数据集在模型相对准确的区域收集（即 $\gamma_0(\hat{z}_{k,i}) \approx 0$），严格耦合问题退化为第 6 节中以 $\sqrt{\gamma_0(z)}$ 为唯一平顶半径的形式；进一步令 $\gamma_0(z) \equiv 0$，则退化为第 3 节定理 1 的标准拉普拉斯核。这条完整的退化链为论文从”极致严格的理论模型”到”紧凑高效的实际算法”的过渡提供了无缝的逻辑链条。