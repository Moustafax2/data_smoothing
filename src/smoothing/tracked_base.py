"""Shared multi-track smoother helpers."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from smoothing.interface import Point, SmootherInterface


class TrackedPointSmoother(SmootherInterface):
    """Base class for smoothers that maintain independent state per tracking ID."""

    def __init__(self) -> None:
        self._player_states: dict[int, dict[str, object]] = {}

    @abstractmethod
    def _create_axis_smoother(self) -> object:
        """Create a fresh axis smoother with update(value) -> float."""

    def _create_player_state(self) -> dict[str, object]:
        return {
            "top_x": self._create_axis_smoother(),
            "top_y": self._create_axis_smoother(),
            "bottom_x": self._create_axis_smoother(),
            "bottom_y": self._create_axis_smoother(),
        }

    def reset(self) -> None:
        self._player_states.clear()

    def cleanup_old_players(self, current_tracking_ids: set[int]) -> None:
        self._player_states = {
            track_id: state
            for track_id, state in self._player_states.items()
            if track_id in current_tracking_ids
        }

    def smooth_points(
        self,
        tracking_id: int,
        top_point: Point,
        bottom_point: Point,
    ) -> tuple[np.ndarray, np.ndarray]:
        player_state = self._player_states.setdefault(int(tracking_id), self._create_player_state())

        top_x = player_state["top_x"].update(float(top_point[0]))
        top_y = player_state["top_y"].update(float(top_point[1]))
        bottom_x = player_state["bottom_x"].update(float(bottom_point[0]))
        bottom_y = player_state["bottom_y"].update(float(bottom_point[1]))

        return (
            np.asarray([top_x, top_y], dtype=float),
            np.asarray([bottom_x, bottom_y], dtype=float),
        )
