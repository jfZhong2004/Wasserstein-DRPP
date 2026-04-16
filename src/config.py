from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class RadiusConfig:
    """
    Wasserstein 半径配置（Theorem 3.4 / Eq. (8)）。

    说明：
    - beta 固定为 0.05（95% 置信）；
    - m=1 对应一维噪声；
    - a>1 为轻尾指数参数；
    - c1,c2 为浓缩不等式常数（可根据参考文献实现调整）。
    """

    beta: float = 0.05
    m: int = 1
    a: float = 1.5
    c1: float = 2.0
    c2: float = 1.0


@dataclass(frozen=True)
class PIDTuningConfig:
    """PID 参数自动整定配置。"""

    kp_min: float = 0.0
    kp_max: float = 4.0
    kp_step: float = 0.25
    ki_min: float = 0.0
    ki_max: float = 1.2
    ki_step: float = 0.1
    kd_min: float = 0.0
    kd_max: float = 1.5
    kd_step: float = 0.1
    local_kp_delta: float = 0.4
    local_ki_delta: float = 0.2
    local_kd_delta: float = 0.2
    objective_u_weight: float = 0.1


@dataclass(frozen=True)
class ExperimentConfig:
    """
    一维实验总配置。
    """

    steps: int = 32  # k = 0..31
    x0: float = 2.0
    dataset_seed_master: int = 20260412
    eval_seed_master: int = 30260412
    n_values: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200, 500, 1000])
    n_main: int = 100
    m_monte_carlo: int = 1000
    dt: float = 1.0
    u_max: float = 4.0
    logpdf_floor: float = -1e6
    nominal_noise_mean: float = 0.0
    nominal_noise_var: float = 1.0
    w4_truncation_L: float = 8.0
    gamma2_noise_drpp: float = 3.0
    lower_r_quant: float = 0.02
    plot_y_q_low: float = 0.02
    plot_y_q_high: float = 0.98
    plot_y_padding_ratio: float = 0.1
    datasets_root: Path = Path("datasets_1d_wdrpp")
    results_root: Path = Path("results_1d_wdrpp")
    radius: RadiusConfig = field(default_factory=RadiusConfig)
    pid_tuning: PIDTuningConfig = field(default_factory=PIDTuningConfig)

    @property
    def control_modes(self) -> List[str]:
        return ["zero", "pid"]

    @property
    def noise_ids(self) -> List[str]:
        return ["W1", "W2", "W3", "W4", "W5", "W6"]

