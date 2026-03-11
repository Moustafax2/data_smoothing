"""Fixed-lag adaptive smoother with EMA pre-pass."""

from __future__ import annotations

import numpy as np

from smoothing.interface import SmootherInterface
from smoothing.smoothers._adaptive_common import apply_fixed_lag_refinement
from smoothing.smoothers.adaptive_ema_smoother import AdaptiveEmaSmoother


class FixedLagAdaptiveEmaSmoother(SmootherInterface):
    name = "fixed_lag_adaptive_ema"

    def __init__(self, lag_frames: int = 4, polyorder: int = 2, ema_alpha: float = 0.5) -> None:
        if lag_frames < 1:
            raise ValueError("Fixed-lag EMA lag_frames must be >= 1.")
        if polyorder < 1:
            raise ValueError("Fixed-lag EMA polyorder must be >= 1.")
        self.lag_frames = lag_frames
        self.polyorder = polyorder
        self.base = AdaptiveEmaSmoother(ema_alpha=ema_alpha)

    def reset(self) -> None:
        self.base.reset()

    def smooth_points(self, x: float, y: float) -> tuple[float, float]:
        return self.base.smooth_points(x, y)

    def smooth_sequence(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        base_x, base_y = self.base.smooth_sequence(x_values, y_values)
        return (
            apply_fixed_lag_refinement(base_x, lag_frames=self.lag_frames, polyorder=self.polyorder),
            apply_fixed_lag_refinement(base_y, lag_frames=self.lag_frames, polyorder=self.polyorder),
        )
