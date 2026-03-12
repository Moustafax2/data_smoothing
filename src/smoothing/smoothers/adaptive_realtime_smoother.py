"""Adaptive real-time smoother class."""

from __future__ import annotations

from smoothing.smoothers._adaptive_common import AdaptiveAxisFilter
from smoothing.tracked_base import TrackedPointSmoother


class AdaptiveRealtimeSmoother(TrackedPointSmoother):
    name = "hybrid_realtime"

    def __init__(
        self,
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
        if outlier_window < 3:
            raise ValueError("Adaptive outlier_window must be >= 3.")
        if outlier_gate <= 0.0:
            raise ValueError("Adaptive outlier_gate must be > 0.")
        if process_variance <= 0.0:
            raise ValueError("Adaptive process_variance must be > 0.")
        if measurement_variance <= 0.0:
            raise ValueError("Adaptive measurement_variance must be > 0.")
        if not 0.0 < base_smoothing < 1.0:
            raise ValueError("Adaptive base_smoothing must be in the range (0, 1).")
        if not 0.0 < min_smoothing < 1.0:
            raise ValueError("Adaptive min_smoothing must be in the range (0, 1).")
        if not 0.0 < max_smoothing <= 1.0:
            raise ValueError("Adaptive max_smoothing must be in the range (0, 1].")
        if min_smoothing > max_smoothing:
            raise ValueError("Adaptive min_smoothing must be <= max_smoothing.")
        if turn_responsiveness < 0.0:
            raise ValueError("Adaptive turn_responsiveness must be >= 0.")
        if not 0.0 <= turn_velocity_damping <= 1.0:
            raise ValueError("Adaptive turn_velocity_damping must be in the range [0, 1].")
        if acceleration_responsiveness < 0.0:
            raise ValueError("Adaptive acceleration_responsiveness must be >= 0.")
        if skip_update_gate <= 0.0:
            raise ValueError("Adaptive skip_update_gate must be > 0.")
        super().__init__()
        self.outlier_window = outlier_window
        self.outlier_gate = outlier_gate
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.base_smoothing = base_smoothing
        self.min_smoothing = min_smoothing
        self.max_smoothing = max_smoothing
        self.turn_responsiveness = turn_responsiveness
        self.turn_velocity_damping = turn_velocity_damping
        self.acceleration_responsiveness = acceleration_responsiveness
        self.skip_update_gate = skip_update_gate

    def _create_axis_smoother(self) -> AdaptiveAxisFilter:
        return AdaptiveAxisFilter(
            outlier_window=self.outlier_window,
            outlier_gate=self.outlier_gate,
            process_variance=self.process_variance,
            measurement_variance=self.measurement_variance,
            base_smoothing=self.base_smoothing,
            min_smoothing=self.min_smoothing,
            max_smoothing=self.max_smoothing,
            turn_responsiveness=self.turn_responsiveness,
            turn_velocity_damping=self.turn_velocity_damping,
            acceleration_responsiveness=self.acceleration_responsiveness,
            skip_update_gate=self.skip_update_gate,
        )
