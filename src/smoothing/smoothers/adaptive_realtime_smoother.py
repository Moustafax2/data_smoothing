"""Adaptive real-time smoother class."""

from __future__ import annotations

from smoothing.smoothers._adaptive_common import AdaptiveAxisFilter
from smoothing.tracked_base import TrackedPointSmoother


class AdaptiveRealtimeSmoother(TrackedPointSmoother):
    name = "hybrid_realtime"

    def __init__(
        self,
        outlier_window: int = 7,
        outlier_gate: float = 2.2,
        process_variance: float = 7e-2,
        measurement_variance: float = 18.0,
        base_smoothing: float = 0.68,
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
        super().__init__()
        self.outlier_window = outlier_window
        self.outlier_gate = outlier_gate
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.base_smoothing = base_smoothing

    def _create_axis_smoother(self) -> AdaptiveAxisFilter:
        return AdaptiveAxisFilter(
            outlier_window=self.outlier_window,
            outlier_gate=self.outlier_gate,
            process_variance=self.process_variance,
            measurement_variance=self.measurement_variance,
            base_smoothing=self.base_smoothing,
        )
