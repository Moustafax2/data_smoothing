"""Synthetic data generation for smoothing experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeneratedSeries:
    frames: np.ndarray
    clean_x: np.ndarray
    raw_x: np.ndarray


def generate_noisy_x_series(
    num_points: int = 500,
    noise_std: float = 4.5,
    seed: int = 7,
) -> GeneratedSeries:
    """Create a more realistic tracked x-trajectory with motion changes and jitter artifacts."""
    rng = np.random.default_rng(seed)
    frames = np.arange(num_points, dtype=float)
    dt = 1.0

    position = 320.0
    velocity = 0.0
    clean_x = np.empty(num_points, dtype=float)

    segment_lengths: list[int] = []
    while sum(segment_lengths) < num_points:
        segment_lengths.append(int(rng.integers(18, 55)))

    target_velocities = rng.normal(loc=0.0, scale=3.2, size=len(segment_lengths))
    target_velocities = np.clip(target_velocities, -7.0, 7.0)

    frame_index = 0
    for segment_length, target_velocity in zip(segment_lengths, target_velocities):
        remaining = min(segment_length, num_points - frame_index)
        for _ in range(remaining):
            # Ease velocity toward a new target, which creates realistic acceleration and cuts.
            velocity += 0.14 * (target_velocity - velocity)
            velocity += rng.normal(0.0, 0.08)
            velocity = float(np.clip(velocity, -8.5, 8.5))
            position += velocity * dt
            clean_x[frame_index] = position
            frame_index += 1
            if frame_index >= num_points:
                break
        if frame_index >= num_points:
            break

    # Add slow tactical drift so the path is not just piecewise linear.
    clean_x += 10.0 * np.sin(frames / 42.0) + 5.5 * np.sin(frames / 13.0)

    # Base detector jitter.
    measurement_noise = rng.normal(0.0, noise_std * 1.8, size=num_points)

    # Low-frequency detector bias/drift.
    drift = np.cumsum(rng.normal(0.0, noise_std * 0.08, size=num_points))

    # Short bursts of higher jitter, which are common around fast movement or occlusion.
    burst_noise = np.zeros(num_points, dtype=float)
    burst_count = max(6, num_points // 70)
    for _ in range(burst_count):
        burst_start = int(rng.integers(0, max(1, num_points - 8)))
        burst_length = int(rng.integers(5, 16))
        burst_scale = float(rng.uniform(noise_std * 3.0, noise_std * 5.5))
        burst_end = min(num_points, burst_start + burst_length)
        burst_noise[burst_start:burst_end] += rng.normal(0.0, burst_scale, size=burst_end - burst_start)

    # Occasional hard outliers to mimic momentary detector failures.
    outlier_noise = np.zeros(num_points, dtype=float)
    outlier_count = max(4, num_points // 90)
    outlier_indices = rng.choice(num_points, size=outlier_count, replace=False)
    outlier_noise[outlier_indices] = rng.normal(0.0, noise_std * 8.0, size=outlier_count)

    raw_x = clean_x + measurement_noise + drift + burst_noise + outlier_noise

    return GeneratedSeries(frames=frames, clean_x=clean_x, raw_x=raw_x)
