"""Fixed-lag adaptive smoother with EMA pre-pass."""

from __future__ import annotations

from collections import deque

import numpy as np

from smoothing.smoothers._adaptive_common import AdaptiveAxisFilter, apply_fixed_lag_refinement
from smoothing.smoothers.adaptive_ema_smoother import AdaptiveEmaSmoother
from smoothing.tracked_base import TrackedPointSmoother


class _FixedLagAdaptiveEmaAxis:
    def __init__(self, lag_frames: int, polyorder: int, ema_alpha: float) -> None:
        self.lag_frames = lag_frames
        self.polyorder = polyorder
        self.ema_alpha = ema_alpha
        self.ema_value: float | None = None
        self.base_axis = AdaptiveAxisFilter()
        self.buffer: deque[float] = deque(maxlen=2 * lag_frames + 1)

    def update(self, value: float) -> float:
        if self.ema_value is None:
            self.ema_value = value
        else:
            self.ema_value = self.ema_alpha * value + (1.0 - self.ema_alpha) * self.ema_value
        base_value = self.base_axis.update(self.ema_value)
        self.buffer.append(base_value)
        if len(self.buffer) < self.buffer.maxlen:
            return base_value
        window = np.asarray(self.buffer, dtype=float)
        refined = apply_fixed_lag_refinement(window, lag_frames=self.lag_frames, polyorder=self.polyorder)
        return float(refined[self.lag_frames])


class FixedLagAdaptiveEmaSmoother(TrackedPointSmoother):
    name = "fixed_lag_adaptive_ema"

    def __init__(self, lag_frames: int = 4, polyorder: int = 2, ema_alpha: float = 0.5) -> None:
        if lag_frames < 1:
            raise ValueError("Fixed-lag EMA lag_frames must be >= 1.")
        if polyorder < 1:
            raise ValueError("Fixed-lag EMA polyorder must be >= 1.")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("Fixed-lag EMA alpha must be in the range (0, 1].")
        super().__init__()
        self.lag_frames = lag_frames
        self.polyorder = polyorder
        self.ema_alpha = ema_alpha

    def _create_axis_smoother(self) -> _FixedLagAdaptiveEmaAxis:
        return _FixedLagAdaptiveEmaAxis(
            lag_frames=self.lag_frames,
            polyorder=self.polyorder,
            ema_alpha=self.ema_alpha,
        )

    def smooth_sequence(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        base = AdaptiveEmaSmoother(ema_alpha=self.ema_alpha)
        top, bottom = base.smooth_sequence(x_values, y_values)
        bottom = np.column_stack(
            (
                apply_fixed_lag_refinement(bottom[:, 0], lag_frames=self.lag_frames, polyorder=self.polyorder),
                apply_fixed_lag_refinement(bottom[:, 1], lag_frames=self.lag_frames, polyorder=self.polyorder),
            )
        )
        return top, bottom
