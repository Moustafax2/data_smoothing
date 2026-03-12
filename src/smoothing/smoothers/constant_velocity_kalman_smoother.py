"""Constant-velocity Kalman smoother class."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from smoothing.tracked_base import TrackedPointSmoother


@dataclass
class _ConstantVelocityAxis:
    process_variance: float
    measurement_variance: float
    dt: float
    state: np.ndarray | None = field(default=None, init=False)
    covariance: np.ndarray | None = field(default=None, init=False)

    def update(self, value: float) -> float:
        if self.state is None:
            self.state = np.array([[value], [0.0]], dtype=float)
            self.covariance = np.eye(2, dtype=float)
            return value

        assert self.covariance is not None
        transition = np.array([[1.0, self.dt], [0.0, 1.0]], dtype=float)
        measurement_matrix = np.array([[1.0, 0.0]], dtype=float)
        process_noise = self.process_variance * np.array(
            [[self.dt**4 / 4.0, self.dt**3 / 2.0], [self.dt**3 / 2.0, self.dt**2]],
            dtype=float,
        )
        measurement_noise = np.array([[self.measurement_variance]], dtype=float)
        identity = np.eye(2, dtype=float)

        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process_noise

        measurement = np.array([[value]], dtype=float)
        innovation = measurement - measurement_matrix @ self.state
        innovation_covariance = measurement_matrix @ self.covariance @ measurement_matrix.T + measurement_noise
        kalman_gain = self.covariance @ measurement_matrix.T @ np.linalg.inv(innovation_covariance)

        self.state = self.state + kalman_gain @ innovation
        self.covariance = (identity - kalman_gain @ measurement_matrix) @ self.covariance
        return float(self.state[0, 0])


class ConstantVelocityKalmanSmoother(TrackedPointSmoother):
    name = "cv_kalman"

    def __init__(self, process_variance: float = 1e-2, measurement_variance: float = 20.0, dt: float = 1.0):
        if process_variance <= 0.0:
            raise ValueError("Constant-velocity Kalman process_variance must be > 0.")
        if measurement_variance <= 0.0:
            raise ValueError("Constant-velocity Kalman measurement_variance must be > 0.")
        if dt <= 0.0:
            raise ValueError("Constant-velocity Kalman dt must be > 0.")
        super().__init__()
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.dt = dt

    def _create_axis_smoother(self) -> _ConstantVelocityAxis:
        return _ConstantVelocityAxis(
            process_variance=self.process_variance,
            measurement_variance=self.measurement_variance,
            dt=self.dt,
        )
