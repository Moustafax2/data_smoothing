"""Shared smoothing interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SmootherInterface(ABC):
    """Frame-by-frame point smoother contract."""

    name: str = "base"

    def reset(self) -> None:
        """Reset internal state before processing a new sequence."""

    @abstractmethod
    def smooth_points(self, x: float, y: float) -> tuple[float, float]:
        """Smooth one frame of point data."""

    def smooth_sequence(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Default batch helper built from smooth_points."""
        x_data = np.asarray(x_values, dtype=float)
        if y_values is None:
            y_data = np.zeros_like(x_data, dtype=float)
        else:
            y_data = np.asarray(y_values, dtype=float)

        self.reset()
        smooth_x = np.empty_like(x_data, dtype=float)
        smooth_y = np.empty_like(y_data, dtype=float)

        for idx, (x_point, y_point) in enumerate(zip(x_data, y_data)):
            smooth_x[idx], smooth_y[idx] = self.smooth_points(float(x_point), float(y_point))

        return smooth_x, smooth_y
