"""Visualization entrypoint for smoothing experiments."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from smoothing.data import generate_noisy_x_series
from smoothing.filters import available_smoothers
from smoothing.runner import benchmark_smoothers


def run_demo(
    num_points: int = 500,
    noise_std: float = 4.5,
    seed: int = 7,
    savgol_window_length: int = 51,
    savgol_polyorder: int = 2,
    alpha_beta_alpha: float = 0.18,
    alpha_beta_beta: float = 0.004,
    constant_velocity_process_variance: float = 1e-2,
    constant_velocity_measurement_variance: float = 20.0,
    fixed_lag_frames: int = 4,
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
            savgol_window_length=savgol_window_length,
            savgol_polyorder=savgol_polyorder,
            alpha_beta_alpha=alpha_beta_alpha,
            alpha_beta_beta=alpha_beta_beta,
            constant_velocity_process_variance=constant_velocity_process_variance,
            constant_velocity_measurement_variance=constant_velocity_measurement_variance,
            fixed_lag_frames=fixed_lag_frames,
        ),
        series.raw_x,
    )

    if output_path is None:
        output_path = Path("outputs") / "smoothing_comparison.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
    panels = [
        (
            "savgol",
            f"Savitzky-Golay Benchmark (window={savgol_window_length}, poly={savgol_polyorder})",
            "#059669",
        ),
        (
            "alpha_beta",
            f"Alpha-Beta (alpha={alpha_beta_alpha}, beta={alpha_beta_beta})",
            "#ea580c",
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
            "fixed_lag_adaptive",
            f"Fixed-Lag Adaptive Filter ({fixed_lag_frames} frame delay)",
            "#0f766e",
        ),
    ]

    for axis, (method_name, title, color) in zip(axes, panels):
        result = results[method_name]
        axis.plot(series.frames, series.raw_x, label="Raw x", color="#9ca3af", linewidth=1.1, alpha=0.9)
        axis.plot(series.frames, result.bottom_points[:, 0], label=title, color=color, linewidth=1.35)
        axis.set_title(f"{title} | {result.elapsed_ms:.2f} ms total")
        axis.set_ylabel("X position")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper left")

    axes[-1].set_xlabel("Frame")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    return output_path


def _load_track_from_cache(cache_path: Path, track_id: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    frames: list[float] = []
    bottom_x: list[float] = []
    bottom_y: list[float] = []
    counts: Counter[int] = Counter()
    rows: list[dict] = []

    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            for player in row["players"]:
                counts[int(player["track_id"])] += 1

    if not rows:
        raise ValueError(f"Cache file is empty: {cache_path}")

    selected_track_id = track_id if track_id is not None else counts.most_common(1)[0][0]

    for row in rows:
        for player in row["players"]:
            if int(player["track_id"]) != selected_track_id:
                continue
            frames.append(float(row["frame_idx"]))
            bottom_x.append(float(player["bottom_point"][0]))
            bottom_y.append(float(player["bottom_point"][1]))
            break

    if not frames:
        raise ValueError(f"Track id {selected_track_id} was not found in {cache_path}")

    return np.asarray(frames), np.asarray(bottom_x), np.asarray(bottom_y), selected_track_id


def run_cache_demo(
    cache_path: Path,
    track_id: int | None = None,
    savgol_window_length: int = 51,
    savgol_polyorder: int = 2,
    alpha_beta_alpha: float = 0.18,
    alpha_beta_beta: float = 0.004,
    constant_velocity_process_variance: float = 1e-2,
    constant_velocity_measurement_variance: float = 20.0,
    fixed_lag_frames: int = 4,
    output_path: Path | None = None,
) -> Path:
    """Plot smoother comparisons on a real tracked player trajectory cache."""
    frames, bottom_x, bottom_y, selected_track_id = _load_track_from_cache(cache_path, track_id=track_id)
    results = benchmark_smoothers(
        available_smoothers(
            savgol_window_length=savgol_window_length,
            savgol_polyorder=savgol_polyorder,
            alpha_beta_alpha=alpha_beta_alpha,
            alpha_beta_beta=alpha_beta_beta,
            constant_velocity_process_variance=constant_velocity_process_variance,
            constant_velocity_measurement_variance=constant_velocity_measurement_variance,
            fixed_lag_frames=fixed_lag_frames,
        ),
        bottom_x,
        bottom_y,
    )

    if output_path is None:
        output_path = Path("outputs") / f"real_track_{selected_track_id}_comparison.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
    panels = [
        (
            "savgol",
            f"Savitzky-Golay Benchmark (window={savgol_window_length}, poly={savgol_polyorder})",
            "#059669",
        ),
        (
            "alpha_beta",
            f"Alpha-Beta (alpha={alpha_beta_alpha}, beta={alpha_beta_beta})",
            "#ea580c",
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
            "fixed_lag_adaptive",
            f"Fixed-Lag Adaptive Filter ({fixed_lag_frames} frame delay)",
            "#0f766e",
        ),
    ]

    for axis, (method_name, title, color) in zip(axes, panels):
        result = results[method_name]
        axis.plot(frames, bottom_x, label="Raw bottom x", color="#9ca3af", linewidth=1.1, alpha=0.9)
        axis.plot(frames, result.bottom_points[:, 0], label=title, color=color, linewidth=1.35)
        axis.set_title(f"Track {selected_track_id} | {title} | {result.elapsed_ms:.2f} ms total")
        axis.set_ylabel("Bottom X")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper left")

    axes[-1].set_xlabel("Frame")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    return output_path
