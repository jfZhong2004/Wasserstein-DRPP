from __future__ import annotations

from src.predictors.common import PredictorBase


class EigDRPPPlaceholder(PredictorBase):
    """
    Eig-DRPP 预留接口（当前不实现）。
    """

    def __init__(self) -> None:
        self.name = "eig_drpp"

    def logpdf(self, step: int, x_next: float, x_k: float, u_k: float) -> float:  # noqa: ARG002
        raise NotImplementedError("Eig-DRPP is reserved for future extension.")

