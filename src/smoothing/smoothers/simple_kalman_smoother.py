"""Simple Kalman smoother class."""

from __future__ import annotations

from dataclasses import dataclass

from smoothing.interface import SmootherInterface


@dataclass
class _SimpleKalmanAxis:
    process_variance: float
    measurement_variance: float
    estimate: float | None = None
    error_covariance: float = 1.0

    def reset(self) -> None:
        self.estimate = None
        self.error_covariance = 1.0

    def update(self, value: float) -> float:
        if self.estimate is None:
            self.estimate = value
            return value

        self.error_covariance += self.process_variance
        kalman_gain = self.error_covariance / (self.error_covariance + self.measurement_variance)
        self.estimate = self.estimate + kalman_gain * (value - self.estimate)
        self.error_covariance = (1.0 - kalman_gain) * self.error_covariance
        return self.estimate


class SimpleKalmanSmoother(SmootherInterface):
    name = "simple_kalman"

    def __init__(self, process_variance: float = 1e-2, measurement_variance: float = 20.0) -> None:
        if process_variance <= 0.0:
            raise ValueError("Kalman process_variance must be > 0.")
        if measurement_variance <= 0.0:
            raise ValueError("Kalman measurement_variance must be > 0.")
        self.x_axis = _SimpleKalmanAxis(process_variance, measurement_variance)
        self.y_axis = _SimpleKalmanAxis(process_variance, measurement_variance)

    def reset(self) -> None:
        self.x_axis.reset()
        self.y_axis.reset()

    def smooth_points(self, x: float, y: float) -> tuple[float, float]:
        return self.x_axis.update(x), self.y_axis.update(y)
