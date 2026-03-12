"""Helpers for running and benchmarking smoothers."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from smoothing.interface import SmootherInterface


@dataclass(frozen=True)
class SmootherRunResult:
    top_points: np.ndarray
    bottom_points: np.ndarray
    elapsed_ms: float
    ms_per_frame: float


def benchmark_smoothers(
    smoothers: dict[str, SmootherInterface],
    x_values: np.ndarray,
    y_values: np.ndarray | None = None,
) -> dict[str, SmootherRunResult]:
    """Run all smoothers on the same sequence and time them."""
    results: dict[str, SmootherRunResult] = {}
    x_data = np.asarray(x_values, dtype=float)
    if y_values is None:
        y_data = np.zeros_like(x_data, dtype=float)
    else:
        y_data = np.asarray(y_values, dtype=float)

    for name, smoother in smoothers.items():
        start = perf_counter()
        smooth_top, smooth_bottom = smoother.smooth_sequence(x_data, y_data)
        elapsed_ms = (perf_counter() - start) * 1000.0
        results[name] = SmootherRunResult(
            top_points=smooth_top,
            bottom_points=smooth_bottom,
            elapsed_ms=elapsed_ms,
            ms_per_frame=elapsed_ms / max(len(x_data), 1),
        )

    return results
