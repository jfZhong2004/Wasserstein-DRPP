from __future__ import annotations

from dataclasses import dataclass


def saturate(value: float, limit: float) -> float:
    """对控制输入做对称限幅。"""
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


@dataclass
class PIDController:
    """
    一维 PID 控制器（含简单 anti-windup）。
    """

    kp: float
    ki: float
    kd: float
    u_max: float
    x_ref: float = 0.0
    integral: float = 0.0
    prev_error: float = 0.0
    initialized: bool = False

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def control(self, x: float, k: int) -> float:  # noqa: ARG002
        e = self.x_ref - x
        de = 0.0 if not self.initialized else (e - self.prev_error)

        # 先尝试积分，再在饱和时做 anti-windup（饱和即回退积分）。
        tentative_integral = self.integral + e
        u_unsat = self.kp * e + self.ki * tentative_integral + self.kd * de
        u = saturate(u_unsat, self.u_max)

        if u == u_unsat:
            self.integral = tentative_integral

        self.prev_error = e
        self.initialized = True
        return u

