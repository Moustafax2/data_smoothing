"""EMA smoother class."""

from __future__ import annotations

from dataclasses import dataclass

from smoothing.tracked_base import TrackedPointSmoother


@dataclass
class _EmaAxis:
    alpha: float
    value: float | None = None

    def update(self, point: float) -> float:
        if self.value is None:
            self.value = point
            return point
        self.value = self.alpha * point + (1.0 - self.alpha) * self.value
        return self.value


class EmaSmoother(TrackedPointSmoother):
    name = "ema"

    def __init__(self, alpha: float = 0.2) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("EMA alpha must be in the range (0, 1].")
        super().__init__()
        self.alpha = alpha

    def _create_axis_smoother(self) -> _EmaAxis:
        return _EmaAxis(alpha=self.alpha)
