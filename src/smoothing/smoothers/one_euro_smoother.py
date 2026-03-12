"""One Euro smoother class."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smoothing.tracked_base import TrackedPointSmoother


@dataclass
class _OneEuroAxis:
    min_cutoff: float
    beta: float
    derivative_cutoff: float
    frame_rate: float
    previous_value: float | None = None
    previous_derivative: float = 0.0
    previous_output: float | None = None

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


class OneEuroSmoother(TrackedPointSmoother):
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
        super().__init__()
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self.frame_rate = frame_rate

    def _create_axis_smoother(self) -> _OneEuroAxis:
        return _OneEuroAxis(
            min_cutoff=self.min_cutoff,
            beta=self.beta,
            derivative_cutoff=self.derivative_cutoff,
            frame_rate=self.frame_rate,
        )
