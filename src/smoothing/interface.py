"""Shared smoothing interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

Point = tuple[float, float]


class SmootherInterface(ABC):
    """Per-player frame-by-frame point smoother contract."""

    name: str = "base"

    def reset(self) -> None:
        """Reset internal state before processing a new sequence."""

    @abstractmethod
    def smooth_points(
        self,
        tracking_id: int,
        top_point: Point,
        bottom_point: Point,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Smooth one frame of top and bottom points for a single tracked player."""

    def cleanup_old_players(self, current_tracking_ids: set[int]) -> None:
        """Drop state for players that are no longer present."""

    def smooth_sequence(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch helper that feeds one synthetic track through smooth_points."""
        x_data = np.asarray(x_values, dtype=float)
        if y_values is None:
            y_data = np.zeros_like(x_data, dtype=float)
        else:
            y_data = np.asarray(y_values, dtype=float)

        self.reset()
        smooth_top = np.empty((len(x_data), 2), dtype=float)
        smooth_bottom = np.empty((len(x_data), 2), dtype=float)

        for idx, (x_point, y_point) in enumerate(zip(x_data, y_data)):
            top_point, bottom_point = self.smooth_points(
                0,
                (0.0, 0.0),
                (float(x_point), float(y_point)),
            )
            smooth_top[idx] = top_point
            smooth_bottom[idx] = bottom_point

        return smooth_top, smooth_bottom
