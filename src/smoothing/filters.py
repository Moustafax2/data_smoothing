"""Factory and compatibility helpers for class-based smoothers."""

from __future__ import annotations

import numpy as np

from smoothing.interface import SmootherInterface
from smoothing.runner import benchmark_smoothers
from smoothing.smoothers import (
    AdaptiveRealtimeSmoother,
    AlphaBetaSmoother,
    ConstantVelocityKalmanSmoother,
    FixedLagAdaptiveSmoother,
    SavitzkyGolaySmoother,
)


def available_smoothers(
    savgol_window_length: int = 51,
    savgol_polyorder: int = 2,
    alpha_beta_alpha: float = 0.18,
    alpha_beta_beta: float = 0.004,
    constant_velocity_process_variance: float = 1e-2,
    constant_velocity_measurement_variance: float = 20.0,
    fixed_lag_frames: int = 4,
) -> dict[str, SmootherInterface]:
    """Build the smoother registry."""
    adaptive_process_variance = max(7e-2, constant_velocity_process_variance)
    adaptive_measurement_variance = min(18.0, constant_velocity_measurement_variance)
    return {
        "savgol": SavitzkyGolaySmoother(window_length=savgol_window_length, polyorder=savgol_polyorder),
        "alpha_beta": AlphaBetaSmoother(alpha=alpha_beta_alpha, beta=alpha_beta_beta),
        "cv_kalman": ConstantVelocityKalmanSmoother(
            process_variance=constant_velocity_process_variance,
            measurement_variance=constant_velocity_measurement_variance,
        ),
        "hybrid_realtime": AdaptiveRealtimeSmoother(
            process_variance=adaptive_process_variance,
            measurement_variance=adaptive_measurement_variance,
            base_smoothing=0.68,
        ),
        "fixed_lag_adaptive": FixedLagAdaptiveSmoother(
            lag_frames=fixed_lag_frames,
            polyorder=2,
        ),
    }


def smooth_series(values: np.ndarray, method: str = "cv_kalman", **kwargs: float) -> np.ndarray:
    """Compatibility helper for smoothing one x-series."""
    x_values = np.asarray(values, dtype=float)
    registry = available_smoothers(**kwargs)
    try:
        smoother = registry[method]
    except KeyError as exc:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unknown smoothing method '{method}'. Available: {available}") from exc
    _, smooth_bottom = smoother.smooth_sequence(x_values, np.zeros_like(x_values, dtype=float))
    return smooth_bottom[:, 0]


def smooth_all_series(values: np.ndarray, **kwargs: float) -> dict[str, np.ndarray]:
    """Compatibility helper for running every smoother on one x-series."""
    results = benchmark_smoothers(available_smoothers(**kwargs), np.asarray(values, dtype=float))
    return {name: result.bottom_points[:, 0] for name, result in results.items()}
