"""Alpha-beta smoother class."""

from __future__ import annotations

from dataclasses import dataclass

from smoothing.interface import SmootherInterface


@dataclass
class _AlphaBetaAxis:
    alpha: float
    beta: float
    dt: float
    position: float | None = None
    velocity: float = 0.0

    def reset(self) -> None:
        self.position = None
        self.velocity = 0.0

    def update(self, value: float) -> float:
        if self.position is None:
            self.position = value
            return value

        predicted_position = self.position + self.velocity * self.dt
        residual = value - predicted_position
        self.position = predicted_position + self.alpha * residual
        self.velocity = self.velocity + (self.beta / self.dt) * residual
        return self.position


class AlphaBetaSmoother(SmootherInterface):
    name = "alpha_beta"

    def __init__(self, alpha: float = 0.18, beta: float = 0.004, dt: float = 1.0) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("Alpha-beta alpha must be in the range (0, 1].")
        if beta < 0.0:
            raise ValueError("Alpha-beta beta must be >= 0.")
        if dt <= 0.0:
            raise ValueError("Alpha-beta dt must be > 0.")
        self.x_axis = _AlphaBetaAxis(alpha, beta, dt)
        self.y_axis = _AlphaBetaAxis(alpha, beta, dt)

    def reset(self) -> None:
        self.x_axis.reset()
        self.y_axis.reset()

    def smooth_points(self, x: float, y: float) -> tuple[float, float]:
        return self.x_axis.update(x), self.y_axis.update(y)
