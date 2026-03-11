"""EMA smoother class."""

from __future__ import annotations

from smoothing.interface import SmootherInterface


class EmaSmoother(SmootherInterface):
    name = "ema"

    def __init__(self, alpha: float = 0.2) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("EMA alpha must be in the range (0, 1].")
        self.alpha = alpha
        self._x: float | None = None
        self._y: float | None = None

    def reset(self) -> None:
        self._x = None
        self._y = None

    def smooth_points(self, x: float, y: float) -> tuple[float, float]:
        if self._x is None or self._y is None:
            self._x = x
            self._y = y
            return x, y

        self._x = self.alpha * x + (1.0 - self.alpha) * self._x
        self._y = self.alpha * y + (1.0 - self.alpha) * self._y
        return self._x, self._y
