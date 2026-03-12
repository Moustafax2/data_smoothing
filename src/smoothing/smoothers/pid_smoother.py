"""PID smoother class."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smoothing.tracked_base import TrackedPointSmoother


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


class PidSmoother(TrackedPointSmoother):
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
        super().__init__()
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.dt = dt

    def _create_axis_smoother(self) -> _PidAxis:
        return _PidAxis(
            kp=self.kp,
            ki=self.ki,
            kd=self.kd,
            integral_limit=self.integral_limit,
            dt=self.dt,
        )
