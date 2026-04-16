from __future__ import annotations


class ZeroController:
    """零控制策略：u_k = 0。"""

    def control(self, x: float, k: int) -> float:  # noqa: ARG002
        return 0.0

