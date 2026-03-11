"""One Euro smoother class."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smoothing.interface import SmootherInterface


@dataclass
class _OneEuroAxis:
    min_cutoff: float
    beta: float
    derivative_cutoff: float
    frame_rate: float
    previous_value: float | None = None
    previous_derivative: float = 0.0
    previous_output: float | None = None

    def reset(self) -> None:
        self.previous_value = None
        self.previous_derivative = 0.0
        self.previous_output = None

    def _alpha(self, cutoff: float) -> float:
        dt = 1.0 / self.frame_rate
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, value: float) -> float:
        if self.previous_value is None or self.previous_output is None:
            self.previous_value = value
            self.previous_output = value
            return value

        derivative = (value - self.previous_value) * self.frame_rate
        derivative_alpha = self._alpha(self.derivative_cutoff)
        derivative_hat = derivative_alpha * derivative + (1.0 - derivative_alpha) * self.previous_derivative
        cutoff = self.min_cutoff + self.beta * abs(derivative_hat)
        signal_alpha = self._alpha(cutoff)
        output = signal_alpha * value + (1.0 - signal_alpha) * self.previous_output

        self.previous_value = value
        self.previous_derivative = derivative_hat
        self.previous_output = output
        return output


class OneEuroSmoother(SmootherInterface):
    name = "one_euro"

    def __init__(
        self,
        min_cutoff: float = 0.35,
        beta: float = 0.008,
        derivative_cutoff: float = 0.6,
        frame_rate: float = 30.0,
    ) -> None:
        if min_cutoff <= 0.0:
            raise ValueError("One Euro min_cutoff must be > 0.")
        if beta < 0.0:
            raise ValueError("One Euro beta must be >= 0.")
        if derivative_cutoff <= 0.0:
            raise ValueError("One Euro derivative_cutoff must be > 0.")
        if frame_rate <= 0.0:
            raise ValueError("One Euro frame_rate must be > 0.")

        self.x_axis = _OneEuroAxis(min_cutoff, beta, derivative_cutoff, frame_rate)
        self.y_axis = _OneEuroAxis(min_cutoff, beta, derivative_cutoff, frame_rate)

    def reset(self) -> None:
        self.x_axis.reset()
        self.y_axis.reset()

    def smooth_points(self, x: float, y: float) -> tuple[float, float]:
        return self.x_axis.update(x), self.y_axis.update(y)
