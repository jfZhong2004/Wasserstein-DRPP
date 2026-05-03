# W-DRPP 一维实验与理论文档说明

本仓库用于整理 DRPP / W-DRPP 的理论推导文档，并实现 System C 下的一维 W-DRPP 实验流程。当前可运行代码集中在 `src/`，主实验入口是 `src.runner.exp2_systemC`。

## 1. 环境准备

建议使用 Python 3.10 或更高版本。

安装依赖：

```powershell
pip install -r requirements.txt
```

当前 `requirements.txt` 包含：

```text
numpy
scipy
matplotlib
```

## 2. 快速运行

在仓库根目录执行：

```powershell
python -m src.runner.exp2_systemC --m 1000 --steps 32 --n-main 100
```

该命令会完成以下流程：

- 生成或读取历史数据集；
- 对 `zero` 和 `pid` 两种控制方式分别运行实验；
- 对 W1-W6 六种噪声分别评估各预测器；
- 默认额外运行 W7 adversary stress test；
- 输出逐步 log-score、汇总表和曲线图。

如果已有数据集，只想复用已有数据：

```powershell
python -m src.runner.exp2_systemC --skip-dataset-build --m 1000 --n-main 100
```

如果希望跳过 W7 adversary：

```powershell
python -m src.runner.exp2_systemC --skip-adversary
```

## 3. W-DRPP 求解器模式

W-DRPP predictor 支持两种求解模式，由 `--wdrpp-solver-mode` 控制。

使用一维 exact 求解器：

```powershell
python -m src.runner.exp2_systemC --wdrpp-solver-mode exact
```

使用 LSE 松弛求解器：

```powershell
python -m src.runner.exp2_systemC --wdrpp-solver-mode lse --wdrpp-lse-integration closed_form
```

使用 LSE 松弛并采用 MC 积分：

```powershell
python -m src.runner.exp2_systemC --wdrpp-solver-mode lse --wdrpp-lse-integration mc --wdrpp-lse-mc-samples 2000
```

实际使用建议：

- 常规 W1-W6 一维实验可以先用 `exact`，但需要检查求解器是否成功收敛。
- W7 adversary 场景下，`exact` 在小样本或尖锐 adversary 网格下可能出现数值失败，建议优先使用 `lse` 跑通诊断实验。
- 若图中出现 `1e9` 量级的异常正 log-score，通常不是统计结果，而是求解器失败解被用于绘图。

## 4. 常用命令行参数

主入口：

```powershell
python -m src.runner.exp2_systemC [options]
```

核心实验参数：

- `--m`：Monte Carlo 评估次数，默认 `1000`。
- `--steps`：预测步数，默认 `32`；当前实验设计固定要求 `32`。
- `--n-main`：主训练样本量，默认 `100`。
- `--dataset-seed-master`：训练数据主随机种子，默认 `20260412`。
- `--eval-seed-master`：评估轨迹主随机种子，默认 `30260412`。
- `--datasets-root`：数据集根目录，默认 `datasets_1d_wdrpp`。
- `--results-root`：结果根目录，默认 `results_1d_wdrpp`。
- `--skip-dataset-build`：跳过历史数据集生成，直接读取已有数据。

Wasserstein 半径参数：

- `--beta`：置信参数，默认 `0.05`。
- `--a`：轻尾指数参数，默认 `1.5`。
- `--c1`：Theorem 3.4 / Eq.(8) 中的常数，默认 `2.0`。
- `--c2`：Theorem 3.4 / Eq.(8) 中的常数，默认 `1.0`。

W-DRPP 求解参数：

- `--wdrpp-solver-mode`：`exact` 或 `lse`，默认 `exact`。
- `--wdrpp-lse-integration`：`closed_form` 或 `mc`，默认 `closed_form`。
- `--wdrpp-lse-mc-samples`：LSE-MC 积分采样数，默认 `2000`。
- `--wdrpp-lse-mc-seed`：LSE-MC 积分随机种子，默认 `20260501`。

W7 adversary 参数：

- `--skip-adversary`：跳过 W7 adversary。
- `--adversary-source-noises`：构造 W7 时 pooled 的源噪声，默认 `W1,W2,W3,W4,W5,W6`。
- `--adversary-grid-size`：一维 adversary LP 的目标网格大小，默认 `400`。
- `--adversary-source-samples`：每步 adversary LP 使用的 pooled 源残差点上限，默认 `120`。
- `--adversary-support-scale`：adversary 支撑区间 padding 系数，默认 `2.0`。

## 5. 输出文件

默认数据集目录：

```text
datasets_1d_wdrpp/
```

默认结果目录：

```text
results_1d_wdrpp/run_YYYYMMDD_HHMMSS/
```

结果目录主要内容：

- `experiment_config.yaml`：本次实验配置快照。
- `requirement_trace.md`：实验需求追踪记录。
- `per_step_scores.csv`：逐步、逐方法 log-score 统计。
- `summary_table.csv`：每个控制方式、噪声、方法的总体均值 log-score。
- `figures/*.png`：逐步 score curve 图。

注意：`results_1d_wdrpp/` 已写入 `.gitignore`，默认不再同步到 Git。

## 6. `src/` 代码结构

`src/config.py`：

集中定义实验配置，包括步数、初始状态、随机种子、样本量、W-DRPP 求解器设置、W7 adversary 设置、数据目录和结果目录。

`src/runner/exp2_systemC.py`：

实验主入口。负责生成数据集、构造预测器、运行 Monte Carlo 评估、运行 W7 adversary、写出 CSV 与图片。

`src/data/build_dataset.py`：

历史数据集生成和读取。默认数据结构为 `datasets_1d_wdrpp/systemC/control_{mode}/noise_{Wj}/N_xxx/step_kk.csv`。

`src/systems/system_c.py`：

System C 的系统动力学与名义漂移函数。

`src/controllers/zero.py`：

零控制策略。

`src/controllers/pid.py`：

PID 控制策略。

`src/controllers/pid_tuning.py`：

PID 参数自动整定逻辑。

`src/noise_lib/noise_w1_w6.py`：

W1-W6 噪声分布定义。W4 使用截断 Student-t 后方差标准化的规则。

`src/predictors/common.py`：

预测器公共数据结构和工具函数，例如 `StepDataset`、`PredictorBase`、残差变换、Gaussian mixture logpdf。

`src/predictors/nominal.py`：

Nominal baseline。使用固定高斯噪声模型。

`src/predictors/noise_drpp.py`：

Noise-DRPP / Moment-DRPP baseline。当前实现为基于残差均值和方差的单峰高斯预测器，并带有方差放大参数。

`src/predictors/kde.py`：

KDE baseline。使用历史残差构造核密度估计。

`src/predictors/oracle.py`：

Oracle predictor。仅用于 W1-W6 已知真实噪声分布的评估；W7 adversary 没有 oracle。

`src/predictors/wdrpp.py`：

W-DRPP predictor 的公共接口，包含 `WDRPPUpperPredictor`、`WDRPPLowerPredictor` 以及 `gamma0_value`。

`src/predictors/wdrpp_upper.py` 和 `src/predictors/wdrpp_lower.py`：

W-DRPP upper / lower 相关兼容入口或拆分模块。

`src/predictors/eig_drpp_interface.py`：

EIG-DRPP 接口占位或外部接口适配代码。

`src/solvers/drpp_1d_exact_solver.py`：

一维 exact 求解器。核心思想是构造上包络线并使用闭式积分求解尖顶和平顶模型。

`src/solvers/drpp_lse_solver.py`：

加性 LSE 松弛求解器。支持 `d>=1` 的接口；一维场景可用闭式积分，高维或复杂场景可切换 MC 积分。

`src/radius/wasserstein_radius.py`：

Wasserstein 半径计算，当前对应 Theorem 3.4 / Eq.(8) 风格的有限样本半径。

`src/eval/logscore.py`：

log-score 评估工具。

`src/viz/plot_per_step_scores.py`：

逐步 log-score 曲线绘图。纵轴默认使用稳健分位数范围，避免极端点压缩其他曲线。

`src/docs/代码使用说明.md`：

较早版本的代码使用说明。根目录 README 是面向整个仓库的总说明，内容在该文档基础上扩展。

## 7. W1-W6 与 W7 adversary

W1-W6 是常规噪声实验。每个控制方式、每个噪声分别训练和评估 predictor，并绘制对应曲线。

W7 adversary 是新增的 stress test，不是一个固定外部真实噪声分布。当前实现逻辑为：

- 对每个控制方式分别处理；
- 对每个 step 分别处理；
- pooled W1-W6 历史残差作为经验中心；
- 构造 common adversary；
- adversary 目标方法是 `wdrpp_upper`；
- 在一维支撑网格上求解离散 Wasserstein transport LP；
- 用该 adversarial distribution 评价所有非 Oracle predictor；
- W7 不绘制 Oracle，因为不存在真实 W7 噪声分布。

解释 W7 图时需要注意：

- W7 是“攻击 wdrpp_upper 的共同 adversary”，不是公平自然噪声。
- W7 曲线不能直接解释为普通泛化表现。
- 若 `n_main` 很小、LP 网格很粗或 exact 求解器失败，图像可能主要反映数值问题。
- 建议正式展示前检查 `per_step_scores.csv`、`summary_table.csv`、求解器状态和异常量级。

## 8. 根目录重要文件

`DRPP.md`：

原始 DRPP 相关论文或文档整理，包含 conic ambiguity set 版本的实验与 adversarial 设计参考。

`Wasserstein.md`：

Wasserstein 距离、收敛率或相关理论材料。

`单步DRPP_OT强对偶严格证明.md`：

单步 DRPP 与 OT 强对偶相关的严格证明文档。

`单步DRPP优化求解.md`：

单步 DRPP 优化求解方法，包括上界、下界和可计算形式。

`多步W-DRPP常数递推上下界求解方法.md`：

多步 W-DRPP 的常数递推上下界方法。重点讨论常数 continuation、无状态递推、上下界平滑和显式/半显式公式。

`多步W-DRPP价值函数上下界分析.md`：

多步 W-DRPP 价值函数上下界分析文档。

`实验方案设计.md`：

实验方案总设计文档。包含 W1-W6、W7 adversary、支撑区间、计算方法和实验解释逻辑。

`一维W-DRPP实验设计方案.md`：

一维 W-DRPP 实验的早期或专项设计方案。

`theoremA_1d_visualizer.py`：

定理 A 或单步上界相关的一维可视化脚本。

`theoremB_1d_visualizer.py`：

定理 B 或单步下界相关的一维可视化脚本。

`requirements.txt`：

Python 依赖列表。

`.gitignore`：

Git 忽略规则。当前至少忽略 `results_1d_wdrpp/`。

`historical/`：

历史推导、旧定义、伪证或临时笔记。只作为追溯参考，不建议作为当前正式结论引用。

`try/`：

试验性推导或临时材料。

## 9. 复现实验注意事项

训练和评估种子应保持不同，默认分别为 `20260412` 和 `30260412`。

当前实验固定 `steps=32`，如果命令行传入其他值，程序会报错。

`datasets_1d_wdrpp/` 会保存历史数据集；如果修改噪声定义、控制器或系统动力学，建议不要复用旧数据集。

`results_1d_wdrpp/` 保存实验输出，并已从 Git 跟踪中移除；需要分享结果时应单独打包或挑选关键图表。

如果运行中断，结果目录可能只有部分图片而没有 `per_step_scores.csv` 或 `summary_table.csv`。这种目录不能视为完整实验结果。

如果 WDRPP 曲线出现极端正 log-score，应优先检查求解器是否失败，而不是直接解释为方法优势。

