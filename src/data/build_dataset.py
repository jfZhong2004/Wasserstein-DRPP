from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.config import ExperimentConfig
from src.controllers.pid import PIDController
from src.controllers.pid_tuning import PIDGains
from src.controllers.zero import ZeroController
from src.noise_lib.noise_w1_w6 import NoiseModel
from src.predictors.common import StepDataset
from src.systems.system_c import nominal_drift, true_next_state


@dataclass(frozen=True)
class HistoricalDatasetMeta:
    control_mode: str
    noise_id: str
    n: int


def _make_controller(control_mode: str, pid_gains: PIDGains, cfg: ExperimentConfig):
    if control_mode == "zero":
        return ZeroController()
    if control_mode == "pid":
        return PIDController(
            kp=pid_gains.kp,
            ki=pid_gains.ki,
            kd=pid_gains.kd,
            u_max=cfg.u_max,
        )
    raise ValueError(f"Unknown control_mode: {control_mode}")


def build_historical_datasets(
    cfg: ExperimentConfig,
    noise_lib: Dict[str, NoiseModel],
    pid_gains: PIDGains,
) -> None:
    """
    生成并落盘历史数据集：
    datasets_1d_wdrpp/systemC/control_xxx/noise_Wi/N_xxx/step_kk.csv
    """
    root = cfg.datasets_root / "systemC"
    root.mkdir(parents=True, exist_ok=True)

    for control_mode in cfg.control_modes:
        for noise_id in cfg.noise_ids:
            noise = noise_lib[noise_id]
            for n in cfg.n_values:
                out_dir = root / f"control_{control_mode}" / f"noise_{noise_id}" / f"N_{n:03d}"
                out_dir.mkdir(parents=True, exist_ok=True)

                # step-wise collector
                rows_by_step: List[List[List[float]]] = [[] for _ in range(cfg.steps)]

                for traj_id in range(n):
                    seed = (
                        cfg.dataset_seed_master
                        + traj_id
                        + 100_000 * cfg.noise_ids.index(noise_id)
                        + 1_000_000 * cfg.control_modes.index(control_mode)
                    )
                    rng = np.random.default_rng(seed)
                    ctrl = _make_controller(control_mode, pid_gains, cfg)
                    x = cfg.x0
                    for k in range(cfg.steps):
                        u = float(ctrl.control(x, k))
                        w = float(noise.sample(rng, 1)[0])
                        x_next = float(true_next_state(x, u, w))
                        w_hat = x_next - nominal_drift(x, u)
                        rows_by_step[k].append([traj_id, x, u, x_next, w_hat])
                        x = x_next

                for k in range(cfg.steps):
                    file_path = out_dir / f"step_{k:02d}.csv"
                    with file_path.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(["traj_id", "x_k", "u_k", "x_next", "w_hat"])
                        writer.writerows(rows_by_step[k])


def load_historical_dataset(
    cfg: ExperimentConfig,
    control_mode: str,
    noise_id: str,
    n: int,
) -> Dict[int, StepDataset]:
    """
    读取某个配置下某个样本量的 step-wise 历史数据集。
    """
    base = cfg.datasets_root / "systemC" / f"control_{control_mode}" / f"noise_{noise_id}" / f"N_{n:03d}"
    if not base.exists():
        raise FileNotFoundError(f"Dataset directory not found: {base}")

    out: Dict[int, StepDataset] = {}
    for k in range(cfg.steps):
        file_path = base / f"step_{k:02d}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {file_path}")
        arr = np.genfromtxt(file_path, delimiter=",", names=True, dtype=float)
        # 防止 n=1 时 shape 异常
        if arr.shape == ():
            arr = np.array([arr], dtype=arr.dtype)
        out[k] = StepDataset(
            x_k=np.asarray(arr["x_k"], dtype=float),
            u_k=np.asarray(arr["u_k"], dtype=float),
            x_next=np.asarray(arr["x_next"], dtype=float),
            w_hat=np.asarray(arr["w_hat"], dtype=float),
        )
    return out

