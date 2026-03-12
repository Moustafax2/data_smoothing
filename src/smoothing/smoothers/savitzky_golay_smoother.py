"""Savitzky-Golay smoother class."""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.signal import savgol_filter

from smoothing.tracked_base import TrackedPointSmoother


class _SavitzkyGolayAxis:
    def __init__(self, window_length: int, polyorder: int) -> None:
        self.window_length = window_length
        self.polyorder = polyorder
        self.buffer: deque[float] = deque(maxlen=window_length)

    def update(self, value: float) -> float:
        self.buffer.append(value)
        if len(self.buffer) < self.window_length:
            return value
        window = np.asarray(self.buffer, dtype=float)
        filtered = savgol_filter(window, window_length=self.window_length, polyorder=self.polyorder, mode="interp")
        return float(filtered[self.window_length // 2])


class SavitzkyGolaySmoother(TrackedPointSmoother):
    name = "savgol"

    def __init__(self, window_length: int = 51, polyorder: int = 2) -> None:
        if window_length < 3 or window_length % 2 == 0:
            raise ValueError("Savitzky-Golay window_length must be an odd integer >= 3.")
        if polyorder < 1 or polyorder >= window_length:
            raise ValueError("Savitzky-Golay polyorder must be >= 1 and < window_length.")
        super().__init__()
        self.window_length = window_length
        self.polyorder = polyorder

    def _create_axis_smoother(self) -> _SavitzkyGolayAxis:
        return _SavitzkyGolayAxis(window_length=self.window_length, polyorder=self.polyorder)

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

        adjusted_window = min(self.window_length, len(x_data) if len(x_data) % 2 == 1 else len(x_data) - 1)
        if adjusted_window < 3:
            top = np.zeros((len(x_data), 2), dtype=float)
            bottom = np.column_stack((x_data, y_data))
            return top, bottom

        adjusted_polyorder = min(self.polyorder, adjusted_window - 1)
        smooth_x = savgol_filter(x_data, window_length=adjusted_window, polyorder=adjusted_polyorder, mode="interp")
        smooth_y = savgol_filter(y_data, window_length=adjusted_window, polyorder=adjusted_polyorder, mode="interp")
        top = np.zeros((len(x_data), 2), dtype=float)
        bottom = np.column_stack((smooth_x, smooth_y))
        return top, bottom
