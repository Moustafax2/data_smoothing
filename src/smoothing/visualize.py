"""Visualization entrypoint for smoothing experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from smoothing.data import generate_noisy_x_series
from smoothing.filters import available_smoothers
from smoothing.runner import benchmark_smoothers


def run_demo(
    num_points: int = 500,
    alpha: float = 0.2,
    noise_std: float = 4.5,
    seed: int = 7,
    savgol_window_length: int = 51,
    savgol_polyorder: int = 2,
    one_euro_min_cutoff: float = 0.35,
    one_euro_beta: float = 0.008,
    one_euro_derivative_cutoff: float = 0.6,
    one_euro_frame_rate: float = 30.0,
    alpha_beta_alpha: float = 0.18,
    alpha_beta_beta: float = 0.004,
    pid_kp: float = 0.18,
    pid_ki: float = 0.002,
    pid_kd: float = 0.08,
    pid_integral_limit: float = 25.0,
    constant_velocity_process_variance: float = 1e-2,
    constant_velocity_measurement_variance: float = 20.0,
    fixed_lag_frames: int = 4,
    adaptive_ema_alpha: float = 0.5,
    output_path: Path | None = None,
) -> Path:
    """Generate synthetic x-points and render a stacked comparison plot."""
    series = generate_noisy_x_series(
        num_points=num_points,
        noise_std=noise_std,
        seed=seed,
    )
    results = benchmark_smoothers(
        available_smoothers(
            alpha=alpha,
            savgol_window_length=savgol_window_length,
            savgol_polyorder=savgol_polyorder,
            one_euro_min_cutoff=one_euro_min_cutoff,
            one_euro_beta=one_euro_beta,
            one_euro_derivative_cutoff=one_euro_derivative_cutoff,
            one_euro_frame_rate=one_euro_frame_rate,
            alpha_beta_alpha=alpha_beta_alpha,
            alpha_beta_beta=alpha_beta_beta,
            pid_kp=pid_kp,
            pid_ki=pid_ki,
            pid_kd=pid_kd,
            pid_integral_limit=pid_integral_limit,
            constant_velocity_process_variance=constant_velocity_process_variance,
            constant_velocity_measurement_variance=constant_velocity_measurement_variance,
            fixed_lag_frames=fixed_lag_frames,
            adaptive_ema_alpha=adaptive_ema_alpha,
        ),
        series.raw_x,
    )

    if output_path is None:
        output_path = Path("outputs") / "smoothing_comparison.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(10, 1, figsize=(14, 34), sharex=True)
    panels = [
        ("ema", f"EMA Benchmark (alpha={alpha})", "#2563eb"),
        (
            "savgol",
            f"Savitzky-Golay Benchmark (window={savgol_window_length}, poly={savgol_polyorder})",
            "#059669",
        ),
        (
            "one_euro",
            (
                "One Euro "
                f"(min_cutoff={one_euro_min_cutoff}, beta={one_euro_beta}, "
                f"derivative_cutoff={one_euro_derivative_cutoff})"
            ),
            "#7c3aed",
        ),
        (
            "alpha_beta",
            f"Alpha-Beta (alpha={alpha_beta_alpha}, beta={alpha_beta_beta})",
            "#ea580c",
        ),
        (
            "pid",
            f"PID (kp={pid_kp}, ki={pid_ki}, kd={pid_kd})",
            "#be123c",
        ),
        (
            "cv_kalman",
            (
                "Constant-Velocity Kalman "
                f"(process_var={constant_velocity_process_variance}, "
                f"measurement_var={constant_velocity_measurement_variance})"
            ),
            "#0891b2",
        ),
        (
            "hybrid_realtime",
            "Adaptive Real-Time Filter",
            "#111827",
        ),
        (
            "adaptive_ema",
            f"Adaptive Filter + EMA Prepass (alpha={adaptive_ema_alpha})",
            "#1d4ed8",
        ),
        (
            "fixed_lag_adaptive",
            f"Fixed-Lag Adaptive Filter ({fixed_lag_frames} frame delay)",
            "#0f766e",
        ),
        (
            "fixed_lag_adaptive_ema",
            f"Fixed-Lag Adaptive + EMA Prepass ({fixed_lag_frames} frame delay, alpha={adaptive_ema_alpha})",
            "#155e75",
        ),
    ]

    for axis, (method_name, title, color) in zip(axes, panels):
        result = results[method_name]
        axis.plot(series.frames, series.raw_x, label="Raw x", color="#9ca3af", linewidth=1.1, alpha=0.9)
        axis.plot(series.frames, result.x_values, label=title, color=color, linewidth=1.35)
        axis.set_title(f"{title} | {result.elapsed_ms:.2f} ms total")
        axis.set_ylabel("X position")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper left")

    axes[-1].set_xlabel("Frame")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    return output_path
