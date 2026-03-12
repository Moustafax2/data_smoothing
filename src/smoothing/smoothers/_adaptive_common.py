"""Shared helpers for adaptive smoother classes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import savgol_filter


@dataclass
class AdaptiveAxisFilter:
    outlier_window: int = 7
    outlier_gate: float = 2.2
    process_variance: float = 7e-2
    measurement_variance: float = 18.0
    base_smoothing: float = 0.68
    state: np.ndarray | None = field(default=None, init=False)
    covariance: np.ndarray | None = field(default=None, init=False)
    previous_output: float | None = field(default=None, init=False)
    history: list[float] = field(default_factory=list, init=False)
    innovation_history: list[float] = field(default_factory=list, init=False)

    def reset(self) -> None:
        self.state = None
        self.covariance = None
        self.previous_output = None
        self.history.clear()
        self.innovation_history.clear()

    def update(self, value: float) -> float:
        if self.state is None:
            self.state = np.array([[value], [0.0]], dtype=float)
            self.covariance = np.eye(2, dtype=float)
            self.previous_output = value
            self.history.append(value)
            return value

        assert self.covariance is not None
        assert self.previous_output is not None

        dt = 1.0
        transition = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
        measurement_matrix = np.array([[1.0, 0.0]], dtype=float)
        process_noise = self.process_variance * np.array(
            [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]],
            dtype=float,
        )
        base_measurement_noise = np.array([[self.measurement_variance]], dtype=float)
        identity = np.eye(2, dtype=float)

        predicted_state = transition @ self.state
        predicted_covariance = transition @ self.covariance @ transition.T + process_noise

        recent = np.asarray(self.history[-self.outlier_window :], dtype=float)
        recent_innovations = np.asarray(self.innovation_history[-self.outlier_window :], dtype=float)
        if recent_innovations.size > 0:
            innovation_median = float(np.median(recent_innovations))
            innovation_mad = float(np.median(np.abs(recent_innovations - innovation_median)))
            innovation_sigma = 1.4826 * innovation_mad + 1e-6
        elif recent.size > 1:
            diffs = np.diff(recent)
            diff_median = float(np.median(diffs))
            diff_mad = float(np.median(np.abs(diffs - diff_median)))
            innovation_sigma = max(1.4826 * diff_mad, self.measurement_variance**0.5 * 0.35) + 1e-6
        else:
            innovation_sigma = self.measurement_variance**0.5

        predicted_value = float(predicted_state[0, 0])
        clipped_value = predicted_value + np.clip(
            value - predicted_value,
            -self.outlier_gate * innovation_sigma,
            self.outlier_gate * innovation_sigma,
        )
        measurement = np.array([[clipped_value]], dtype=float)
        innovation = measurement - measurement_matrix @ predicted_state
        innovation_scale = abs(float(innovation[0, 0])) / (2.5 * innovation_sigma + 1e-6)
        adaptive_measurement_noise = base_measurement_noise * (1.0 + min(innovation_scale, 6.0))

        if abs(float(innovation[0, 0])) > 6.5 * innovation_sigma and len(self.history) > self.outlier_window:
            self.state = predicted_state
            self.covariance = predicted_covariance
            estimate = float(self.state[0, 0])
        else:
            innovation_covariance = (
                measurement_matrix @ predicted_covariance @ measurement_matrix.T
                + adaptive_measurement_noise
            )
            kalman_gain = predicted_covariance @ measurement_matrix.T @ np.linalg.inv(innovation_covariance)
            self.state = predicted_state + kalman_gain @ innovation
            self.covariance = (identity - kalman_gain @ measurement_matrix) @ predicted_covariance
            estimate = float(self.state[0, 0])

        motion_ratio = min(abs(float(self.state[1, 0])) / 4.0, 1.0)
        smoothing_alpha = min(self.base_smoothing + 0.12 * motion_ratio, 0.9)
        output = smoothing_alpha * estimate + (1.0 - smoothing_alpha) * self.previous_output
        self.previous_output = output
        self.history.append(value)
        self.innovation_history.append(float(value - predicted_value))
        return output


def apply_fixed_lag_refinement(values: np.ndarray, lag_frames: int, polyorder: int) -> np.ndarray:
    """Apply a small future-aware local polynomial refinement."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return data.copy()

    window_length = min(2 * lag_frames + 1, len(data) if len(data) % 2 == 1 else len(data) - 1)
    if window_length < 3:
        return data.copy()

    adjusted_polyorder = min(polyorder, window_length - 1)
    refined = savgol_filter(
        data,
        window_length=window_length,
        polyorder=adjusted_polyorder,
        mode="interp",
    )
    output = refined.copy()
    output[-lag_frames:] = data[-lag_frames:]
    return output
