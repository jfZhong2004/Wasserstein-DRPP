实验要求：

请你参考实验方案设计.md，为我设计一个一维W-DRPP实验文档的markdown文件，要求能够直接参照这一文档，以此作为框架，生成所需实验的所有代码。要求：基准动力系统仅考虑原实验方案中的系统C，噪声分布库可以原样保留。

历史数据集生成方式、样本量取值范围和原来一致。请你单独生成一个文件夹存储数据集。步数为0,1,...,31。其中系统采取采取两种控制方式，一种为完全没有控制（u为0），另一种采取LQR控制（Q和R全部为单位阵，需要你自行调用工具或自行实现）。

对比方法采取DRPP.pdf（若无法读取，请读取DRPP.md）中的Noise-DRPP（后续我会再扩展Eigen-DRPP，现在还不需要实现，但需要保留接口），KDE，Empirical，Oracle，W-DRPP中的上下界。

仅开展原实验方案设计中的实验2，使用系统C，但是\gamma_0取一个合理的函数（参考DRPP.pdf），0-31步预测，进行M=1000次蒙特卡洛重复，性能指标仅评估log-score。

实验结果展示：需要在一个文件夹中保存实验结果（每次实验要建立一个带有时间戳的文件夹），不仅有实验条件描述、实验结果；还要有可视化：每一时间步基于不同算法的评分，画成几条图线展示在同一个图像中。

如果有其他实验细节，可以参考DRPP中开展实验的方法。

 一维W-DRPP实验设计方案.md已存在，请你直接原文件内容（要求）以下进行文档撰写。请你先生成实验文档，评估实验要求的合理性及缺陷，是否有未提及的要求，不要直接开始写代码。如有疑问请提出。

请在下面写下正式实验设计方案，不要删掉上面的内容：

---

## 一、实验要求合理性评估（先评估，后实施）

### 1.1 总体合理性

当前要求是可执行且目标明确的，尤其是：

1. 只做系统 C，范围收敛，利于快速闭环；
2. 仅保留实验 2（噪声分布敏感性），避免任务发散；
3. 固定指标为 Log-Score，便于统一比较；
4. 明确了两种控制策略（Zero / PID）和 Monte Carlo 重复次数（M=1000）；
5. 明确要求结果目录按时间戳存档，便于复现实验与追溯。

### 1.2 主要缺陷与风险点

1. **PID 参数若不先固定，会导致结果不可比**  
   系统 C 为非线性系统，PID 参数 \((K_p,K_i,K_d)\) 若按实验配置分别手调，会引入额外偏差，需先统一整定并冻结。

2. **\(\gamma_0(z)\) 对下界结果高度敏感**  
   本轮已固定采用 DRPP.pdf 形式，后续实现中不得再改动该函数。

3. **Oracle 口径已固定**  
   本轮统一定义为“使用真实噪声分布 \(P_w\) 的一步预测器”，后续实现不得替换为其他近似口径。

4. **KDE 带宽若不固定会引入额外波动**  
   本轮建议固定为“稳健 Silverman 规则”并写入配置，避免人为调参。

5. **0–31 步的评价口径需明确**  
   建议定义为 \(k=0,\dots,31\) 共 32 个一步预测评分，并按步统计均值与置信区间。

### 1.3 本轮固定约定（按本次新指令执行）

1. **控制策略固定为 Zero + PID**，LQR 暂不纳入本轮实验。  
2. **PID 参数先统一整定，再冻结用于全部实验配置**。建议采用“两阶段自动整定”：
   - 阶段 A（粗搜）：在标称模型上网格搜索 \((K_p,K_i,K_d)\)（固定范围见 2.2.4）；
   - 阶段 B（细化）：在粗搜最优点附近做局部连续优化（如 Nelder-Mead）；
   - 整定与正式实验统一采用控制限幅 \(u_{\max}=4\)（即 \(u_k\in[-4,4]\)）并启用 anti-windup；
   - 目标函数：
   \[
   J_{\text{PID}}=\sum_{k=0}^{31}\left(x_{k+1}^2+0.1\,u_k^2\right),
   \]
   选取使 \(J_{\text{PID}}\) 最小的一组参数作为全局固定 PID 参数。  
3. **\(\gamma_0(z)\) 固定采用 DRPP.pdf 形式**（与 `DRPP.md` 一致）：
   \[
   \gamma_0(z)=\min\{0.3\|z\|_2,\ 5\}\cdot \Delta t^2,\quad z=(x,u),\ \Delta t=1.
   \]
4. **Oracle 定义**：使用实验配置下的真实噪声分布 \(P_w\) 构造真实一步条件密度。  
5. **KDE**：Gaussian kernel + 稳健 Silverman 带宽（见 2.6.4，固定公式）。  
6. **噪声尺度统一**：W1–W6 全部固定为方差 1（统一数量级，仅保留形状差异）。  
7. **W-DRPP 求解器**：直接复用现有 `src/solvers/drpp_1d_exact_solver.py`（不重复实现）。
8. **Wasserstein 半径**：严格按 `Wassersteein.pdf` / `Wassersteein.md` 的 Theorem 3.4 与式 (8) 计算，固定 \(\beta=0.05\)。

---

## 二、正式实验设计方案（可直接转化为代码实现）

> 本方案严格针对“系统 C + 实验 2（噪声分布敏感性）”。

## 2.1 实验目标

在系统 C 下比较以下方法在不同噪声分布（W1–W6）中的一步概率预测性能（Log-Score）：

- Moment-DRPP（Noise-DRPP）
- KDE
- Empirical
- Oracle
- W-DRPP Upper（定理 1，\(\gamma_0\equiv 0\)）
- W-DRPP Lower（定理 2，\(\gamma_0\neq 0\)）

并保留 **Eig-DRPP 接口占位**（仅接口，不实现算法）。

## 2.2 动力系统与控制

### 2.2.1 真实系统（System C）
\[
x_{k+1}=\sin(x_k)+0.5u_k+w_k,\quad k=0,\dots,31.
\]

### 2.2.2 标称模型（用于构造经验噪声）
\[
\bar f_k(z)=x_k-\frac{x_k^3}{6}+0.5u_k,\quad z=(x_k,u_k).
\]

### 2.2.3 控制策略

1. **Zero control**：\(u_k=0\)。  
2. **PID control**：  
   - 控制目标：跟踪 \(x_{\text{ref}}=0\)；  
   - 误差定义：\(e_k=x_{\text{ref}}-x_k\)；  
   - 控制律：
   \[
   u_k=\operatorname{sat}\!\Big(K_p e_k + K_i\sum_{t=0}^{k}e_t + K_d(e_k-e_{k-1})\Big),
   \]
   其中 \(\operatorname{sat}(\cdot)\) 固定为
   \[
   \operatorname{sat}(v)=\min\{4,\max\{-4,v\}\},
   \]
   并固定启用 anti-windup。

### 2.2.4 PID 参数确定流程（先于正式实验执行）

1. 在标称模型
\[
\bar f_k(z)=x_k-\frac{x_k^3}{6}+0.5u_k
\]
上进行 32 步滚动仿真；  
2. 先做**阶段 A 粗搜**（推荐固定范围）：
   \[
   K_p\in[0,4]\ \text{(step }0.25),\quad
   K_i\in[0,1.2]\ \text{(step }0.1),\quad
   K_d\in[0,1.5]\ \text{(step }0.1).
   \]
3. 以
\[
J_{\text{PID}}=\sum_{k=0}^{31}(x_{k+1}^2+0.1u_k^2)
\]
为整定目标，先粗搜后细化；  
4. 再做**阶段 B 细化**：以粗搜最优点为初值，采用 Nelder-Mead，在局部盒约束
   \[
   K_p\pm0.4,\ K_i\pm0.2,\ K_d\pm0.2
   \]
   （并裁剪回阶段 A 全局范围）内优化。  
5. 固定主随机种子 `seed_master=20260412` 与单一初值 \(x_0=2.0\) 进行整定评估；  
6. 得到单组 \((K_p,K_i,K_d)\) 后，冻结用于全部后续实验（Zero 控制组除外）。

## 2.3 噪声分布库（保持形状一致，统一方差=1）

- W1: \(\mathcal N(0,1)\)  
- W2: 基础分布 \(W2_{\text{base}}=0.5\mathcal N(-2,0.5^2)+0.5\mathcal N(2,0.5^2)\)，使用
  \[
  W2=\frac{W2_{\text{base}}}{\sqrt{4.25}}
  \]
  使 \(\mathrm{Var}(W2)=1\)。  
- W3: 基础分布 \(W3_{\text{base}}=0.3\mathcal N(-1,0.3^2)+0.7\mathcal N(2,1^2)\)，使用
  \[
  W3=\frac{W3_{\text{base}}}{\sqrt{2.617}}
  \]
  使 \(\mathrm{Var}(W3)=1\)。  
- W4: Student-t(\(\nu=3\)) 缩放到方差 1  
- W5: \(U[-\sqrt{3},\sqrt{3}]\)（本身方差 1）  
- W6: 基础分布
  \[
  W6_{\text{base}}=\frac{1}{3}\mathcal N(-3,0.3^2)+\frac{1}{3}\mathcal N(0,0.3^2)+\frac{1}{3}\mathcal N(3,0.3^2),
  \]
  使用
  \[
  W6=\frac{W6_{\text{base}}}{\sqrt{6.09}}
  \]
  使 \(\mathrm{Var}(W6)=1\)。

说明：统一方差是强制约束，用于保证不同噪声配置的比较聚焦在“分布形状差异”而非“能量大小差异”。

## 2.4 历史数据生成与样本量

### 2.4.1 样本量
\[
N\in\{10,20,50,100,200,500,1000\}.
\]

### 2.4.2 每个配置的数据集生成

对每个 `(control_mode, noise_id, N)`：

1. 采样 \(\hat z_{k,i}=(\hat x_{k,i},\hat u_{k,i})\)；
2. 用真实系统推进得到 \(\hat x_{k+1,i}\)；
3. 计算经验噪声
   \[
   \hat w_{k,i}=\hat x_{k+1,i}-\bar f_k(\hat z_{k,i}).
   \]
4. 为 \(k=0,\dots,31\) 分步保存 \(\hat w_{k,1:N}\) 与锚点 \(\hat x^{pred}_{k,i}=\bar f_k(z_k)+\hat w_{k,i}\)。

## 2.5 \(\gamma_0(z)\) 设定（用于 W-DRPP Lower）

固定采用 DRPP.pdf 的饱和型函数：
\[
\gamma_0(z)=\min\{0.3\|z\|_2,\ 5\}\cdot \Delta t^2,\quad \Delta t=1.
\]
对应单步半径项（1D）：
\[
R_i(z)=\sqrt{\gamma_0(z)}+\sqrt{\gamma_0(\hat z_{k,i})}.
\]

说明：该项在本轮实验中不再作为可调超参数，直接按上式实现。

### 2.5.1 Wasserstein 半径 \(\varepsilon_N(\beta)\)（严格按 Theorem 3.4 / 式 (8)）

固定置信参数 \(\beta=0.05\)（即 95% 覆盖概率），按文献式 (8)：
\[
\varepsilon_N(\beta)=
\begin{cases}
\left(\dfrac{\log(c_1/\beta)}{c_2 N}\right)^{1/\max\{m,2\}}, & N \ge \dfrac{\log(c_1/\beta)}{c_2},\\[8pt]
\left(\dfrac{\log(c_1/\beta)}{c_2 N}\right)^{1/a}, & N < \dfrac{\log(c_1/\beta)}{c_2}.
\end{cases}
\]

本实验是一维问题（\(m=1\)），因此上式第一支指数为 \(1/2\)。

实现要求：

1. 对每个样本量 \(N\) 计算一次 \(\varepsilon_N(0.05)\)；  
2. 同一 \(N\) 下所有方法/所有 Monte Carlo 复用该半径；  
3. 在 `experiment_config.yaml` 中显式记录 `beta, a, c1, c2, epsilon_N`。  
4. 常数 `a, c1, c2` 的实现方式固定为：**按文献 [21]（Theorem 2）证明中的保守常数公式自动计算**（不手动指定常数）。

## 2.6 方法实现与接口规范

### 2.6.1 方法清单

1. `noise_drpp`（Moment-DRPP / Noise-DRPP）  
2. `kde`  
3. `empirical`  
4. `oracle`  
5. `wdrpp_upper`（尖顶核，Thm1）  
6. `wdrpp_lower`（平顶核，Thm2）  
7. `eig_drpp`（占位接口，返回 NotImplemented 或跳过标记）

### 2.6.2 统一预测接口（建议）

```python
class Predictor:
    def fit(self, historical_dataset, config): ...
    def predict_pdf(self, x_grid, context): ...
    def logpdf(self, x, context): ...
```

其中 `context` 至少包含 \((k,x_k,u_k)\)。

### 2.6.3 W-DRPP 求解器复用规范

直接调用现有 `src/solvers/drpp_1d_exact_solver.py`：

- `wdrpp_upper`：调用 `solve_drpp_1d_exact(centers, epsilon, radii=None)`；
- `wdrpp_lower`：调用 `solve_drpp_1d_exact(centers, epsilon, radii=R_i)`；
- 统一使用返回对象 `DRPP1DSolution` 的 `pdf(x)` / `log_pdf(x)` 接口评估打分。

### 2.6.4 KDE 带宽推荐（本轮固定）

采用 Gaussian kernel，并在每个时间步 \(k\) 基于历史噪声样本 \(\{\hat w_{k,i}\}_{i=1}^{N}\) 计算：
\[
h_k = 0.9 \cdot \min\!\left(\hat\sigma_k,\ \frac{\operatorname{IQR}_k}{1.34}\right)\cdot N^{-1/5}.
\]

数值稳健处理：

1. 若 \(\hat\sigma_k\) 与 IQR 同时接近 0，则设 \(h_k = 10^{-3}\)；  
2. 对每个 \(k\) 仅计算一次 \(h_k\)，并在该配置下全部 Monte Carlo 试验中复用。  

说明：该规则对非高斯/重尾样本比普通 Silverman 更稳健，且无需手工调参。

### 2.6.5 Oracle 实现口径（本轮固定）

Oracle 必须直接使用真实噪声分布 \(P_w\)：
\[
p^{\text{oracle}}_k(x\mid z_k)=p_w\!\left(x-f_{\text{true}}(z_k)\right),
\]
其中 \(f_{\text{true}}(z_k)=\sin(x_k)+0.5u_k\)。

对 Log-Score 评估时，统一使用
\[
\log p^{\text{oracle}}_k(x_{k+1})
\]
作为 Oracle 分数，不允许用 KDE/经验分布近似替代 Oracle。

## 2.7 评价指标与统计口径

仅使用 Log-Score：
\[
\mathrm{LS}_{k}^{(m)}=\log \hat p_k^{(m)}(x_{k+1}^{(m)}),
\]
\[
\overline{\mathrm{LS}}_k=\frac1M\sum_{m=1}^{M}\mathrm{LS}_{k}^{(m)},\quad M=1000,\ k=0,\dots,31\ (\text{共 }32\text{ 步}).
\]

输出两类统计：

1. **逐步曲线**：每种方法一条 \(\overline{\mathrm{LS}}_k\) 曲线；  
2. **整体汇总**：\(\frac1{32}\sum_k \overline{\mathrm{LS}}_k\)。

## 2.8 Monte Carlo 试验流程

对每个 `(control_mode, noise_id)`：

1. 固定 \(N=100\)（对齐原实验 2 主设置），固定主随机种子 `seed_master=20260412`；  
2. 重复 \(m=1,\dots,1000\)：  
   - 从同一单值初始状态 \(x_0=2.0\) 起步；
   - 第 \(m\) 次重复的随机种子由主种子派生（如 `seed_m = seed_master + m`）；
   - 运行 \(k=0,\dots,31\) 共 32 步真实轨迹；
   - 每步由各方法给出一步预测并记录 log-score；  
3. 统计各方法逐步均值曲线与总体均值。

## 2.9 文件与目录规范（必须落盘）

### 2.9.1 数据集目录（单独文件夹）

```text
datasets_1d_wdrpp/
  systemC/
    control_zero/
      noise_W1/
      ...
      noise_W6/
        N_010/
        N_020/
        ...
        N_1000/
          step_00.csv
          ...
          step_31.csv
    control_pid/
      noise_W1/
      ...
      noise_W6/
        N_010/
        N_020/
        ...
        N_1000/
          step_00.csv
          ...
          step_31.csv
```

### 2.9.2 结果目录（每次实验时间戳）

```text
results_1d_wdrpp/
  run_YYYYMMDD_HHMMSS/
    experiment_config.yaml
    requirement_trace.md
    summary_table.csv
    per_step_scores.csv
    figures/
      score_curves_control_zero_noise_W1.png
      ...
      score_curves_control_pid_noise_W6.png
```

其中：

- `experiment_config.yaml`：记录系统、控制、噪声、N、M、`x0=2.0`、`seed_master=20260412`、`beta=0.05`、`a/c1/c2`、`epsilon_N`、`gamma0`、随机种子派生规则；
- `requirement_trace.md`：记录本次运行如何对应本方案要求；
- `per_step_scores.csv`：列建议为  
  `step, method, mean_logscore, std_logscore, control_mode, noise_id`。

## 2.10 可视化要求（强制）

每个 `(control_mode, noise_id)` 生成一张图：

- 横轴：`step = 0..31`
- 纵轴：`mean log-score`
- 多条线：`noise_drpp, kde, empirical, oracle, wdrpp_upper, wdrpp_lower`
- 图例与标题完整，标明 `M=1000`、`N=100`。

## 2.11 代码落地建议（用于后续“生成所有代码”）

建议代码模块结构：

```text
src/
  systems/system_c.py
  controllers/{zero.py,pid.py}
  controllers/pid_tuning.py
  noise_lib/noise_w1_w6.py
  data/build_dataset.py
  predictors/
    noise_drpp.py
    kde.py
    empirical.py
    oracle.py
    wdrpp_upper.py
    wdrpp_lower.py
    eig_drpp_interface.py
  eval/logscore.py
  runner/exp2_systemC.py
  viz/plot_per_step_scores.py
```

其中 `predictors/wdrpp_upper.py` 与 `predictors/wdrpp_lower.py` 必须直接导入并复用 `src/solvers/drpp_1d_exact_solver.py`，禁止重复拷贝实现。

并通过单一入口脚本运行，例如：

```bash
python -m src.runner.exp2_systemC --m 1000 --steps 32 --save_dir results_1d_wdrpp
```

---

## 三、当前未提及但必须补充的执行细节（建议在代码阶段固定）

1. 初始状态固定为单值 \(x_0=2.0\)；  
2. 统一主随机种子固定为 `seed_master=20260412`（各子任务种子按固定规则派生）；  
3. Oracle 仅允许按真实噪声分布解析密度实现（不使用采样近似）；  
4. W-DRPP 求解器超参数（容差、最大迭代数、失败重试策略）；  
5. 对异常值（logpdf 下溢）统一截断规则（如 `logpdf >= -1e6`）。

---

## 四、结论

该设计满足“先写实验文档、评估合理性与缺陷、暂不写代码”的要求，并已给出可直接映射为代码工程的目录结构、数据结构、统计口径与可视化规范。后续实现阶段可按本文件逐模块落地。
