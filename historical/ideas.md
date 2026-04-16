参照DRPP的论文，我提出了一种新的Wasserstein-based的模糊集（Ambiguity Set）。这个模糊集的定义如下：

在实际应用中，一般只能观测到（输入，输出）对，那么是否可以用经验分布来定义模糊集？就像下面这个样子：

$$\mathcal{I}_{k}^{W}\left(z | \bar{f}_{k}, \gamma_{0}, \{ \hat{x}_{k+1,i},\hat{z}_{k,i}\}_{i=1}^N ,\beta\right) \\  :=\left\{P_{\pmb{x}_{k+1} | \pmb{z}_{k}}(\cdot|z) \middle| \begin{array}{c}
\pmb{x}_{k+1}=f_{k}(z)+\pmb{w}_{k}, \\
\left\|f_{k}(z)-\bar{f}_{k}(z)\right\|_{2}^{2} \leq \gamma_{0}(z), \\
\mathbb{P}(\omega_k) \in \mathbb{B}_{\varepsilon (\beta,N)}\left(\frac{1}{N}\Sigma_{i=1}^{N}\delta\left(\omega-(x_{k+1,i}-f_k(z_{k,i}))\right)\right)
\end{array}\right\}$$

这样定义的模糊集包含nominal的函数$f$以及对函数$f$偏差的限制$\gamma$，$\{ \hat{x}_{k+1,i},\hat{y}_{k,i}\}_{i=1}^N$是数据集中的数据，其中逗号后面的$i$是数据集中样本的索引（一共N个数据），以及置信度$\beta$；

模糊集定义的前两行和原来的一致，第三行是在$f$由第二行确定之后，由噪声的经验分布定义的一个wasserstein ball，其中半径$\varepsilon$是$\beta$和$N$的函数（这在论文中有详细论述），这个球的半径保证真实分布有$1-\beta$的概率在球中。

# 问题

上述定义参考了Wasserstein论文中的相关结果。现在我希望能够基于这样的一个定义，给出一个Wasserstein-based的模糊集的**DRPP问题的定义**，并且希望能够给出限制$\gamma_0(z)\equiv0$（也就是真实的函数和nominal的函数完全一致），并且仅仅进行单步DRPP（详见DRPP论文）时的**优化问题的解**。


gemini.md的Wasserstein 单步 DRPP 的有限维凸规约与最优预测分布定理证明的步骤1，你是怎么使用Wasserstein.pdf的thm 4.2的（怎么套进这个公式，然后怎么用基于       Kantorovich-Rubinstein 对偶定理的拓展得到下面的式子），请你改得详细一点