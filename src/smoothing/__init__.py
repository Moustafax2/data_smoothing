"""Utilities for experimenting with point smoothing methods."""

from smoothing.data import generate_noisy_x_series
from smoothing.filters import available_smoothers, smooth_all_series, smooth_series
from smoothing.interface import SmootherInterface
from smoothing.runner import SmootherRunResult, benchmark_smoothers

__all__ = [
    "benchmark_smoothers",
    "available_smoothers",
    "generate_noisy_x_series",
    "smooth_all_series",
    "smooth_series",
    "SmootherInterface",
    "SmootherRunResult",
]
