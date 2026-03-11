"""Savitzky-Golay smoother class."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from smoothing.interface import SmootherInterface


class SavitzkyGolaySmoother(SmootherInterface):
    name = "savgol"

    def __init__(self, window_length: int = 51, polyorder: int = 2) -> None:
        if window_length < 3 or window_length % 2 == 0:
            raise ValueError("Savitzky-Golay window_length must be an odd integer >= 3.")
        if polyorder < 1 or polyorder >= window_length:
            raise ValueError("Savitzky-Golay polyorder must be >= 1 and < window_length.")
        self.window_length = window_length
        self.polyorder = polyorder

    def smooth_points(self, x: float, y: float) -> tuple[float, float]:
        return x, y

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
            return x_data.copy(), y_data.copy()

        adjusted_polyorder = min(self.polyorder, adjusted_window - 1)
        return (
            savgol_filter(x_data, window_length=adjusted_window, polyorder=adjusted_polyorder, mode="interp"),
            savgol_filter(y_data, window_length=adjusted_window, polyorder=adjusted_polyorder, mode="interp"),
        )
