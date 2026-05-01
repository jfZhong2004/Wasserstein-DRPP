from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def plot_score_curves(
    steps: List[int],
    method_to_mean: Dict[str, np.ndarray],
    method_to_q025: Dict[str, np.ndarray],
    method_to_q975: Dict[str, np.ndarray],
    title: str,
    output_path: Path,
    y_q_low: float = 0.02,
    y_q_high: float = 0.98,
    y_padding_ratio: float = 0.1,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    step_arr = np.asarray(steps, dtype=float)
    if step_arr.size >= 2:
        diffs = np.diff(np.sort(step_arr))
        diffs = diffs[diffs > 0]
        min_step = float(np.min(diffs)) if diffs.size > 0 else 1.0
    else:
        min_step = 1.0
    cap_half_width = 0.16 * min_step
    for method, values in method_to_mean.items():
        line, = plt.plot(steps, values, label=method, linewidth=1.8)
        q025 = np.asarray(method_to_q025.get(method, np.array([])), dtype=float).reshape(-1)
        q975 = np.asarray(method_to_q975.get(method, np.array([])), dtype=float).reshape(-1)
        if q025.size == len(steps) and q975.size == len(steps):
            plt.vlines(
                steps,
                q025,
                q975,
                color=line.get_color(),
                alpha=0.55,
                linewidth=1.0,
            )
            plt.hlines(
                q025,
                step_arr - cap_half_width,
                step_arr + cap_half_width,
                color=line.get_color(),
                alpha=0.55,
                linewidth=1.0,
            )
            plt.hlines(
                q975,
                step_arr - cap_half_width,
                step_arr + cap_half_width,
                color=line.get_color(),
                alpha=0.55,
                linewidth=1.0,
            )
    plt.xlabel("Step")
    plt.ylabel("Mean log-score")
    plt.title(title)
    # 使用稳健分位数设置纵轴，避免个别极端点压缩其余曲线可读性。
    # 这里把均值曲线和95%分位杠都纳入范围统计，避免被裁掉。
    all_values = []
    for arr in method_to_mean.values():
        vals = np.asarray(arr, dtype=float).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            all_values.append(vals)
    for arr in method_to_q025.values():
        vals = np.asarray(arr, dtype=float).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            all_values.append(vals)
    for arr in method_to_q975.values():
        vals = np.asarray(arr, dtype=float).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            all_values.append(vals)
    if all_values:
        merged = np.concatenate(all_values)
        q_low = float(np.quantile(merged, y_q_low))
        q_high = float(np.quantile(merged, y_q_high))
        if q_high <= q_low:
            center = 0.5 * (q_low + q_high)
            half = 1.0
            ymin, ymax = center - half, center + half
        else:
            pad = max((q_high - q_low) * y_padding_ratio, 1e-6)
            ymin, ymax = q_low - pad, q_high + pad
        plt.ylim(ymin, ymax)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

