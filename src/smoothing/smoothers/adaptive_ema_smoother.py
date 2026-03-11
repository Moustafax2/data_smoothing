"""Adaptive smoother with EMA pre-pass."""

from __future__ import annotations

from smoothing.interface import SmootherInterface
from smoothing.smoothers.adaptive_realtime_smoother import AdaptiveRealtimeSmoother
from smoothing.smoothers.ema_smoother import EmaSmoother


class AdaptiveEmaSmoother(SmootherInterface):
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
        self.ema = EmaSmoother(alpha=ema_alpha)
        self.adaptive = AdaptiveRealtimeSmoother(
            outlier_window=outlier_window,
            outlier_gate=outlier_gate,
            process_variance=process_variance,
            measurement_variance=measurement_variance,
            base_smoothing=base_smoothing,
        )

    def reset(self) -> None:
        self.ema.reset()
        self.adaptive.reset()

    def smooth_points(self, x: float, y: float) -> tuple[float, float]:
        ema_x, ema_y = self.ema.smooth_points(x, y)
        return self.adaptive.smooth_points(ema_x, ema_y)
