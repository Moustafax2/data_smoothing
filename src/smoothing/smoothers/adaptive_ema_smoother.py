"""Adaptive smoother with EMA pre-pass."""

from __future__ import annotations

from smoothing.smoothers._adaptive_common import AdaptiveAxisFilter
from smoothing.tracked_base import TrackedPointSmoother


class _AdaptiveEmaAxis:
    def __init__(
        self,
        ema_alpha: float,
        outlier_window: int,
        outlier_gate: float,
        process_variance: float,
        measurement_variance: float,
        base_smoothing: float,
    ) -> None:
        self.ema_alpha = ema_alpha
        self.ema_value: float | None = None
        self.adaptive = AdaptiveAxisFilter(
            outlier_window=outlier_window,
            outlier_gate=outlier_gate,
            process_variance=process_variance,
            measurement_variance=measurement_variance,
            base_smoothing=base_smoothing,
        )

    def update(self, value: float) -> float:
        if self.ema_value is None:
            self.ema_value = value
        else:
            self.ema_value = self.ema_alpha * value + (1.0 - self.ema_alpha) * self.ema_value
        return self.adaptive.update(self.ema_value)


class AdaptiveEmaSmoother(TrackedPointSmoother):
    name = "adaptive_ema"

    def __init__(
        self,
        ema_alpha: float = 0.5,
        outlier_window: int = 7,
        outlier_gate: float = 2.2,
        process_variance: float = 7e-2,
        measurement_variance: float = 18.0,
        base_smoothing: float = 0.68,
    ) -> None:
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("Adaptive EMA alpha must be in the range (0, 1].")
        super().__init__()
        self.ema_alpha = ema_alpha
        self.outlier_window = outlier_window
        self.outlier_gate = outlier_gate
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.base_smoothing = base_smoothing

    def _create_axis_smoother(self) -> _AdaptiveEmaAxis:
        return _AdaptiveEmaAxis(
            ema_alpha=self.ema_alpha,
            outlier_window=self.outlier_window,
            outlier_gate=self.outlier_gate,
            process_variance=self.process_variance,
            measurement_variance=self.measurement_variance,
            base_smoothing=self.base_smoothing,
        )
