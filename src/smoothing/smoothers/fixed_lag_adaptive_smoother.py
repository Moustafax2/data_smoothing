"""Fixed-lag adaptive smoother class."""

from __future__ import annotations

from collections import deque

import numpy as np

from smoothing.smoothers._adaptive_common import AdaptiveAxisFilter, apply_fixed_lag_refinement
from smoothing.tracked_base import TrackedPointSmoother


class _FixedLagAdaptiveAxis:
    def __init__(
        self,
        lag_frames: int,
        polyorder: int,
        adaptive_kwargs: dict[str, float | int],
    ) -> None:
        self.lag_frames = lag_frames
        self.polyorder = polyorder
        self.base_axis = AdaptiveAxisFilter(**adaptive_kwargs)
        self.buffer: deque[float] = deque(maxlen=2 * lag_frames + 1)

    def update(self, value: float) -> float:
        base_value = self.base_axis.update(value)
        self.buffer.append(base_value)
        if len(self.buffer) < self.buffer.maxlen:
            return base_value
        window = np.asarray(self.buffer, dtype=float)
        refined = apply_fixed_lag_refinement(window, lag_frames=self.lag_frames, polyorder=self.polyorder)
        return float(refined[self.lag_frames])


class FixedLagAdaptiveSmoother(TrackedPointSmoother):
    name = "fixed_lag_adaptive"

    def __init__(
        self,
        lag_frames: int = 4,
        polyorder: int = 2,
        outlier_window: int = 9,
        outlier_gate: float = 2.5,
        process_variance: float = 0.16,
        measurement_variance: float = 12.0,
        base_smoothing: float = 0.84,
        min_smoothing: float = 0.66,
        max_smoothing: float = 0.96,
        turn_responsiveness: float = 0.75,
        turn_velocity_damping: float = 0.35,
        acceleration_responsiveness: float = 0.10,
        skip_update_gate: float = 8.0,
    ) -> None:
        if lag_frames < 1:
            raise ValueError("Fixed-lag lag_frames must be >= 1.")
        if polyorder < 1:
            raise ValueError("Fixed-lag polyorder must be >= 1.")
        super().__init__()
        self.lag_frames = lag_frames
        self.polyorder = polyorder
        self.adaptive_kwargs: dict[str, float | int] = {
            "outlier_window": outlier_window,
            "outlier_gate": outlier_gate,
            "process_variance": process_variance,
            "measurement_variance": measurement_variance,
            "base_smoothing": base_smoothing,
            "min_smoothing": min_smoothing,
            "max_smoothing": max_smoothing,
            "turn_responsiveness": turn_responsiveness,
            "turn_velocity_damping": turn_velocity_damping,
            "acceleration_responsiveness": acceleration_responsiveness,
            "skip_update_gate": skip_update_gate,
        }

    def _create_axis_smoother(self) -> _FixedLagAdaptiveAxis:
        return _FixedLagAdaptiveAxis(
            lag_frames=self.lag_frames,
            polyorder=self.polyorder,
            adaptive_kwargs=self.adaptive_kwargs,
        )

    def smooth_sequence(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_data = np.asarray(x_values, dtype=float)
        if y_values is None:
            y_data = np.zeros_like(x_data, dtype=float)
        else:
            y_data = np.asarray(y_values, dtype=float)

        x_base_axis = AdaptiveAxisFilter(**self.adaptive_kwargs)
        y_base_axis = AdaptiveAxisFilter(**self.adaptive_kwargs)
        x_base = np.array([x_base_axis.update(float(v)) for v in x_data], dtype=float)
        y_base = np.array([y_base_axis.update(float(v)) for v in y_data], dtype=float)
        top = np.zeros((len(x_data), 2), dtype=float)
        bottom = np.column_stack(
            (
                apply_fixed_lag_refinement(x_base, lag_frames=self.lag_frames, polyorder=self.polyorder),
                apply_fixed_lag_refinement(y_base, lag_frames=self.lag_frames, polyorder=self.polyorder),
            )
        )
        return top, bottom
