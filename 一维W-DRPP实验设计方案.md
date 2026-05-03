# 一维 W-DRPP 实验设计方案（当前实现适配版）

本文档用于描述当前 `src/` 代码实际实现的一维 W-DRPP 实验流程。它替代早期“生成代码前”的设计稿，重点同步以下修改：

- 控制方式固定为 Zero control 与 PID control，不再使用 LQR。
- 对比方法以当前代码为准：Nominal、Noise-DRPP、KDE、Oracle、W-DRPP Upper、W-DRPP Lower。
- `Empirical` 不再作为当前主实验方法；`Eig-DRPP` 只保留接口占位，不参与当前结果图。
- 常规实验包含 W1-W6 六种噪声。
- 新增 W7 adversary 压力测试：按控制方式和 step pooled W1-W6 历史残差，构造 common adversary，当前目标 predictor 为 `wdrpp_upper`。
- W-DRPP 求解器支持 `exact` 与 `lse` 两种模式。
- 结果目录 `results_1d_wdrpp/` 已作为实验输出目录，并已从 Git 跟踪中移除。

---

## 一、实验目标

当前实验只考虑 System C 的一维系统，目标是比较不同概率预测方法在多种噪声分布和两种控制方式下的一步 log-score 表现。

常规实验回答：

1. 在 W1-W6 不同噪声形状下，各预测器的一步概率预测性能如何；
2. Zero control 与 PID control 是否会改变不同方法的相对表现；
3. W-DRPP upper/lower 相比 Nominal、Noise-DRPP、KDE 和 Oracle 的表现如何。

W7 adversary 压力测试回答：

1. 若在 Wasserstein 球内构造一个针对 `wdrpp_upper` 的 common adversary，同一 adversarial 分布下各方法的 log-score 如何；
2. W-DRPP 在“普通噪声平均评估”和“受限 Wasserstein adversary 压力测试”中的表现是否一致；
3. 当前求解器和绘图流程在 adversarial 场景下是否数值稳定。

---

## 二、动力系统与控制

### 2.1 真实系统 System C

真实系统为

$$
x_{k+1}=\sin(x_k)+0.5u_k+w_k,\qquad k=0,\dots,31.
$$

其中 $w_k$ 为真实噪声。

### 2.2 标称模型

标称模型用于计算经验残差：

$$
\bar f_k(x_k,u_k)=x_k-\frac{x_k^3}{6}+0.5u_k.
$$

经验残差定义为

$$
\hat w_{k,i}
=
\hat x_{k+1,i}-\bar f_k(\hat x_{k,i},\hat u_{k,i}).
$$

### 2.3 控制方式

当前代码固定两种控制方式：

1. Zero control：$u_k=0$。
2. PID control：跟踪目标为 $x_{\mathrm{ref}}=0$，控制输入带限幅 $u_k\in[-4,4]$。

PID 控制律为

$$
u_k=\operatorname{sat}\left(
K_p e_k
+K_i\sum_{t=0}^{k}e_t
+K_d(e_k-e_{k-1})
\right),
$$

其中

$$
e_k=x_{\mathrm{ref}}-x_k,\qquad
\operatorname{sat}(v)=\min\{4,\max\{-4,v\}\}.
$$

PID 参数由 `src/controllers/pid_tuning.py` 自动整定后冻结，当前配置记录在每次运行的 `experiment_config.yaml` 中。

---

## 三、噪声分布库

常规实验保留 W1-W6 六种噪声。当前代码中噪声实现位于 `src/noise_lib/noise_w1_w6.py`。

统一原则：

- W1-W6 均统一到方差约为 1；
- 实验比较重点是噪声形状差异，而不是噪声能量差异；
- W4 使用截断 Student-t 后再方差标准化，避免极端尾部导致数值不稳定。

噪声类型：

- W1：标准高斯噪声。
- W2：对称双峰混合高斯，标准化到方差 1。
- W3：非对称混合高斯，标准化到方差 1。
- W4：截断 Student-t 噪声，标准化到方差 1。
- W5：均匀噪声，方差为 1。
- W6：三峰混合高斯，标准化到方差 1。

Oracle 只对 W1-W6 有定义，因为这些噪声有明确真实分布。

---

## 四、历史数据集

### 4.1 样本量

历史数据集支持以下样本量：

$$
N\in\{10,20,50,100,200,500,1000\}.
$$

当前主实验默认使用

$$
N_{\mathrm{main}}=100.
$$

### 4.2 数据集目录

默认数据集目录为

```text
datasets_1d_wdrpp/
```

目录结构为

```text
datasets_1d_wdrpp/
  systemC/
    control_zero/
      noise_W1/
        N_010/
          step_00.csv
          ...
          step_31.csv
      ...
      noise_W6/
    control_pid/
      noise_W1/
      ...
      noise_W6/
```

每个 `step_kk.csv` 至少包含当前状态、控制输入、下一状态和经验残差等字段。

### 4.3 数据集生成与复用

默认运行会先生成历史数据集，再运行评估。

若已有数据集，可使用：

```powershell
python -m src.runner.exp2_systemC --skip-dataset-build
```

如果修改了系统动力学、控制器、噪声定义、随机种子或样本量，不应复用旧数据集。

---

## 五、Wasserstein 半径与模型误差半径

### 5.1 Wasserstein 半径

Wasserstein 半径由 `src/radius/wasserstein_radius.py` 计算，采用 Theorem 3.4 / Eq.(8) 风格的有限样本半径：

$$
\varepsilon_N(\beta)=
\begin{cases}
\left(\dfrac{\log(c_1/\beta)}{c_2 N}\right)^{1/\max\{m,2\}},
& N \ge \dfrac{\log(c_1/\beta)}{c_2},\\[8pt]
\left(\dfrac{\log(c_1/\beta)}{c_2 N}\right)^{1/a},
& N < \dfrac{\log(c_1/\beta)}{c_2}.
\end{cases}
$$

当前默认参数为：

- $\beta=0.05$；
- $m=1$；
- $a=1.5$；
- $c_1=2.0$；
- $c_2=1.0$。

这些参数可以通过命令行覆盖：

```powershell
python -m src.runner.exp2_systemC --beta 0.05 --a 1.5 --c1 2.0 --c2 1.0
```

### 5.2 模型误差函数

W-DRPP lower 使用模型误差函数

$$
\gamma_0(z)=\min\{0.3\|z\|_2,5\},
\qquad z=(x,u).
$$

当前代码中 `dt=1`，因此不额外乘时间步缩放。

一维 lower 的样本半径形式为

$$
R_i(z)=\sqrt{\gamma_0(z)}+\sqrt{\gamma_0(\hat z_{k,i})}.
$$

代码实现中，为减少重复求解，`wdrpp_lower` 会对当前半径做量化缓存，量化粒度由 `lower_r_quant` 控制，默认 `0.02`。

---

## 六、预测方法

当前主实验使用以下 predictor。

### 6.1 Nominal

`nominal` 使用固定高斯噪声模型：

$$
x_{k+1}\mid z_k \sim \mathcal N(\bar f_k(z_k),1).
$$

实现文件：`src/predictors/nominal.py`。

### 6.2 Noise-DRPP

`noise_drpp` 是 Moment-DRPP / Noise-DRPP baseline。当前实现基于每个 step 的历史残差均值和方差构造单峰高斯预测器，并使用方差放大参数 `gamma2_noise_drpp`。

实现文件：`src/predictors/noise_drpp.py`。

### 6.3 KDE

`kde` 使用历史残差构造 Gaussian kernel density estimator。带宽采用稳健 Silverman 规则。

实现文件：`src/predictors/kde.py`。

### 6.4 Oracle

`oracle` 使用 W1-W6 的真实噪声分布计算真实一步条件密度。

Oracle 只参与 W1-W6 常规实验，不参与 W7 adversary，因为 W7 不是一个预先给定的真实噪声分布。

实现文件：`src/predictors/oracle.py`。

### 6.5 W-DRPP Upper

`wdrpp_upper` 使用尖顶核，对应 $\gamma_0\equiv0$ 的上界构造。

实现入口：`src/predictors/wdrpp.py` 中的 `WDRPPUpperPredictor`。

求解器由 `--wdrpp-solver-mode` 控制：

- `exact`：调用 `src/solvers/drpp_1d_exact_solver.py`；
- `lse`：调用 `src/solvers/drpp_lse_solver.py`。

### 6.6 W-DRPP Lower

`wdrpp_lower` 使用平顶核，并包含模型误差半径。

实现入口：`src/predictors/wdrpp.py` 中的 `WDRPPLowerPredictor`。

求解器同样由 `--wdrpp-solver-mode` 控制。

### 6.7 Eig-DRPP 接口

当前只保留 `src/predictors/eig_drpp_interface.py` 作为接口占位，不纳入主实验曲线和汇总表。

### 6.8 不再纳入当前主实验的方法

早期方案中的 `empirical` 目前未作为当前主实验方法实现和绘图。若后续需要恢复，应新增 `src/predictors/empirical.py` 并补齐统一 `logpdf` 接口。

---

## 七、常规 W1-W6 实验流程

对每个控制方式

$$
\text{control\_mode}\in\{\text{zero},\text{pid}\}
$$

和每个噪声

$$
\text{noise\_id}\in\{W1,W2,W3,W4,W5,W6\}
$$

执行以下步骤：

1. 读取或生成对应历史数据集；
2. 构造 `nominal`、`noise_drpp`、`kde`、`oracle`、`wdrpp_upper`、`wdrpp_lower`；
3. 从 $x_0=2.0$ 开始生成 Monte Carlo 测试轨迹；
4. 对每个 step $k=0,\dots,31$ 记录各 predictor 的 log-score；
5. 汇总均值、标准差、2.5% 分位数和 97.5% 分位数；
6. 每个 `(control_mode, noise_id)` 输出一张逐步曲线图。

默认 Monte Carlo 次数：

$$
M=1000.
$$

运行命令：

```powershell
python -m src.runner.exp2_systemC --m 1000 --steps 32 --n-main 100
```

---

## 八、W7 Adversary 压力测试

### 8.1 定位

W7 adversary 是一个额外压力测试，不是自然噪声分布。

当前 W7 的定义是：

- 按控制方式分别构造；
- 按 step 分别构造；
- 每个 step pooled W1-W6 的历史残差；
- 在 pooled 经验分布的 Wasserstein 球内寻找最坏残差分布；
- 当前 common adversary 的目标 predictor 固定为 `wdrpp_upper`；
- 用同一个 adversarial 分布评价所有非 Oracle 方法。

因此 W7 图不应解释为“第七种真实噪声分布下的普通泛化性能”，而应解释为“针对 `wdrpp_upper` 构造的 Wasserstein stress test”。

### 8.2 Pooled 数据

对固定控制方式和固定 step，令

$$
\mathcal D_k^{pool}
=
\mathcal D_k^{W1}\cup\mathcal D_k^{W2}\cup\cdots\cup\mathcal D_k^{W6}.
$$

若主样本量为 $N_{\mathrm{main}}$，则 pooled 样本量为

$$
N_{pool}=6N_{\mathrm{main}}.
$$

该设计由命令行参数控制：

```powershell
--adversary-source-noises W1,W2,W3,W4,W5,W6
```

后续若希望只使用部分噪声构造 W7，例如 W2/W3/W6，可改为：

```powershell
python -m src.runner.exp2_systemC --adversary-source-noises W2,W3,W6
```

### 8.3 支撑区间

为避免 adversary 在全空间上把少量质量搬到极远尾部，当前代码使用有限残差支撑区间。

对固定 step，支撑区间由 pooled 残差范围加 padding 得到：

$$
[\min_i \hat w_i - B_k,\ \max_i \hat w_i + B_k].
$$

其中 padding 近似为

$$
B_k
=
c_B\left(\operatorname{std}(\hat w)+\varepsilon_{pool}+r_k\right),
$$

默认 $c_B=2.0$，由 `--adversary-support-scale` 控制。

### 8.4 离散 Wasserstein LP

给定经验源点 $\hat w_i$ 和目标网格点 $y_j$，令 $\pi_{ij}$ 表示从 $\hat w_i$ 搬运到 $y_j$ 的质量。当前 W7 使用线性规划：

$$
\min_{\pi_{ij}\ge0}
\sum_{i=1}^{N_s}\sum_{j=1}^{M_g}
\pi_{ij}\ell_{\hat p,z}(y_j)
$$

subject to

$$
\sum_{j=1}^{M_g}\pi_{ij}=\frac1{N_s},
\qquad i=1,\dots,N_s,
$$

$$
\sum_{i=1}^{N_s}\sum_{j=1}^{M_g}
\pi_{ij}|y_j-\hat w_i|
\le
\varepsilon_{pool}.
$$

其中损失为目标 predictor 的 log-score：

$$
\ell_{\hat p,z}(w)
=
\log \hat p(\bar f(z)+w\mid z).
$$

这里使用最小化，是因为 log-score 越低表示预测越差。LP 得到的最坏网格分布为

$$
q_j^{adv}=\sum_i\pi_{ij}^*.
$$

### 8.5 当前 W7 输出

W7 每个控制方式输出一张图：

```text
score_curves_control_zero_noise_W7_adversary.png
score_curves_control_pid_noise_W7_adversary.png
```

W7 图中包含：

- `nominal`
- `noise_drpp`
- `kde`
- `wdrpp_upper`
- `wdrpp_lower`

W7 不包含 `oracle`。

W7 的逐步 score 是对离散 adversarial 分布的加权期望，不是 Monte Carlo 重复统计。因此当前实现中 W7 的 `q025_logscore` 与 `q975_logscore` 等于均值。

### 8.6 W7 数值注意事项

W7 对求解器更敏感。若使用 `--wdrpp-solver-mode exact`，小样本或尖锐 adversary 网格可能导致 `wdrpp_upper` 求解失败，并出现极端正 log-score。此时图像会被异常值拉伸，不能解释为方法优势。

正式分析 W7 前必须检查：

- 结果目录是否完整生成了 `per_step_scores.csv` 与 `summary_table.csv`；
- 图中是否出现 $10^6$ 或 $10^9$ 量级异常 log-score；
- WDRPP 求解器是否返回失败解；
- W7 的结果是否只作为 stress test，而非普通噪声实验。

建议 W7 首先使用 LSE 模式：

```powershell
python -m src.runner.exp2_systemC --wdrpp-solver-mode lse --adversary-grid-size 400 --adversary-source-samples 120
```

---

## 九、评价指标与统计口径

唯一评价指标为 log-score。

对普通 W1-W6 Monte Carlo 实验，单次评分为

$$
\mathrm{LS}_{k}^{(m)}
=
\log \hat p_k(x_{k+1}^{(m)}\mid x_k^{(m)},u_k^{(m)}).
$$

逐步均值为

$$
\overline{\mathrm{LS}}_k
=
\frac1M\sum_{m=1}^{M}\mathrm{LS}_{k}^{(m)}.
$$

总体汇总分数为

$$
\overline{\mathrm{LS}}
=
\frac1{32}\sum_{k=0}^{31}\overline{\mathrm{LS}}_k.
$$

对 W7 adversary，逐步分数为

$$
\mathrm{ALS}_{k}(\hat p)
=
\sum_j q_{k,j}^{adv}
\log \hat p_k(\bar f_k(z_k)+y_j\mid z_k).
$$

---

## 十、输出目录与文件

默认结果目录为

```text
results_1d_wdrpp/run_YYYYMMDD_HHMMSS/
```

主要输出文件：

- `experiment_config.yaml`：本次实验配置。
- `requirement_trace.md`：本次运行对应的需求追踪。
- `per_step_scores.csv`：逐步分数。
- `summary_table.csv`：总体汇总。
- `figures/*.png`：曲线图。

`per_step_scores.csv` 当前列为：

```text
control_mode,noise_id,step,method,mean_logscore,std_logscore,q025_logscore,q975_logscore
```

`summary_table.csv` 当前列为：

```text
control_mode,noise_id,method,overall_mean_logscore
```

常规 W1-W6 图像共 12 张：

$$
2\ \text{control modes}\times 6\ \text{noise types}=12.
$$

若启用 W7 adversary，会额外生成 2 张图，因此总图像数为 14 张。

---

## 十一、代码结构

当前代码结构如下：

```text
src/
  config.py
  systems/
    system_c.py
  controllers/
    zero.py
    pid.py
    pid_tuning.py
  noise_lib/
    noise_w1_w6.py
  data/
    build_dataset.py
  predictors/
    common.py
    nominal.py
    noise_drpp.py
    kde.py
    oracle.py
    wdrpp.py
    wdrpp_upper.py
    wdrpp_lower.py
    eig_drpp_interface.py
  solvers/
    drpp_1d_exact_solver.py
    drpp_lse_solver.py
  radius/
    wasserstein_radius.py
  eval/
    logscore.py
  viz/
    plot_per_step_scores.py
  runner/
    exp2_systemC.py
```

核心入口：

```powershell
python -m src.runner.exp2_systemC
```

---

## 十二、推荐运行配置

### 12.1 常规正式实验

```powershell
python -m src.runner.exp2_systemC --m 1000 --steps 32 --n-main 100 --wdrpp-solver-mode lse
```

说明：当前 W7 已默认启用。若要只做 W1-W6，添加 `--skip-adversary`。

### 12.2 只做代码连通性检查

```powershell
python -m src.runner.exp2_systemC --m 10 --n-main 10 --wdrpp-solver-mode lse --adversary-grid-size 30 --adversary-source-samples 10
```

该配置只能验证代码能跑通，不能作为正式实验结论。

### 12.3 复用已有数据集

```powershell
python -m src.runner.exp2_systemC --skip-dataset-build --m 1000 --n-main 100 --wdrpp-solver-mode lse
```

---

## 十三、当前实现与早期设计的差异

1. 早期设计提到 LQR，当前实现使用 PID。
2. 早期设计包含 Empirical，当前主实验未实现该方法。
3. 早期设计要求 Eig-DRPP 预留接口，当前已保留接口但不参与实验。
4. 早期设计只覆盖 W1-W6，当前增加 W7 adversary。
5. 早期设计只强调 exact 求解器，当前代码支持 `exact` 与 `lse`。
6. 早期设计未区分训练种子和评估种子，当前固定为 `dataset_seed_master=20260412` 与 `eval_seed_master=30260412`。
7. 早期设计未说明 W7 没有 Oracle，当前明确 W7 不绘制 Oracle。
8. 早期设计未说明 incomplete run 风险；当前要求若结果目录缺少 CSV，则不能视为完整实验。

---

## 十四、实验解释原则

解释 W1-W6 时，应优先看 `summary_table.csv` 的总体均值，再结合逐步曲线分析特定 step 的行为。

解释 W7 时，必须说明：

- W7 是针对 `wdrpp_upper` 的 common adversary；
- W7 不是真实噪声分布；
- W7 不存在 Oracle；
- W7 分数是加权期望，不是 Monte Carlo 平均；
- 若图中出现极端正值，优先怀疑求解器失败。

论文或报告中不应把 W7 的 common adversary 结果表述为“所有方法在同一个自然噪声分布下的泛化性能”。更准确的表述是：

> 在由 pooled W1-W6 经验残差诱导的 Wasserstein 球内，构造一个针对 W-DRPP upper 的受限最坏残差分布，并在该共同压力场景下比较各方法的 log-score。

