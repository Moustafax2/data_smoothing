"""PID smoother class."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smoothing.interface import SmootherInterface


@dataclass
class _PidAxis:
    kp: float
    ki: float
    kd: float
    integral_limit: float
    dt: float
    output: float | None = None
    integral: float = 0.0
    previous_error: float = 0.0

    def reset(self) -> None:
        self.output = None
        self.integral = 0.0
        self.previous_error = 0.0

    def update(self, value: float) -> float:
        if self.output is None:
            self.output = value
            return value

        error = value - self.output
        self.integral = float(np.clip(self.integral + error * self.dt, -self.integral_limit, self.integral_limit))
        derivative = (error - self.previous_error) / self.dt
        control = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.output = self.output + control
        self.previous_error = error
        return self.output


class PidSmoother(SmootherInterface):
    name = "pid"

    def __init__(
        self,
        kp: float = 0.18,
        ki: float = 0.002,
        kd: float = 0.08,
        integral_limit: float = 25.0,
        dt: float = 1.0,
    ) -> None:
        if kp < 0.0 or ki < 0.0 or kd < 0.0:
            raise ValueError("PID gains must be >= 0.")
        if integral_limit <= 0.0:
            raise ValueError("PID integral_limit must be > 0.")
        if dt <= 0.0:
            raise ValueError("PID dt must be > 0.")
        self.x_axis = _PidAxis(kp, ki, kd, integral_limit, dt)
        self.y_axis = _PidAxis(kp, ki, kd, integral_limit, dt)

    def reset(self) -> None:
        self.x_axis.reset()
        self.y_axis.reset()

    def smooth_points(self, x: float, y: float) -> tuple[float, float]:
        return self.x_axis.update(x), self.y_axis.update(y)
