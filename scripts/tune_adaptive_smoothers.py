from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smoothing.smoothers import (  # noqa: E402
    AdaptiveRealtimeSmoother,
    ConstantVelocityKalmanSmoother,
    FixedLagAdaptiveSmoother,
    SavitzkyGolaySmoother,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune adaptive smoothers against real cached player tracks.")
    parser.add_argument("--input-cache", type=Path, required=True, help="Raw tracking JSONL cache.")
    parser.add_argument("--iterations", type=int, default=80, help="Random-search iterations.")
    parser.add_argument("--max-tracks", type=int, default=14, help="Tune on the longest N tracks.")
    parser.add_argument("--min-track-length", type=int, default=80, help="Minimum frames per track.")
    parser.add_argument("--fixed-lag-frames", type=int, default=4, help="Delay for fixed-lag adaptive.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "adaptive_tuning_results.json",
        help="Where to save the tuning summary.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible search.")
    return parser


def load_tracks(cache_path: Path, min_track_length: int, max_tracks: int) -> dict[int, np.ndarray]:
    tracks: dict[int, list[float]] = {}
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for player in row["players"]:
                track_id = int(player["track_id"])
                tracks.setdefault(track_id, []).append(float(player["bottom_point"][0]))

    eligible = {
        track_id: np.asarray(values, dtype=float)
        for track_id, values in tracks.items()
        if len(values) >= min_track_length
    }
    ranked = sorted(eligible.items(), key=lambda item: len(item[1]), reverse=True)
    return dict(ranked[:max_tracks])


def savgol_teacher(values: np.ndarray) -> np.ndarray:
    _, bottom = SavitzkyGolaySmoother(window_length=51, polyorder=2).smooth_sequence(values)
    return bottom[:, 0]


def estimate_turn_mask(target: np.ndarray) -> np.ndarray:
    if len(target) < 5:
        return np.zeros(len(target), dtype=bool)
    velocity = np.diff(target)
    sign = np.sign(velocity)
    changes = np.zeros(len(target), dtype=bool)
    for idx in range(1, len(sign)):
        if sign[idx - 1] == 0.0 or sign[idx] == 0.0:
            continue
        if sign[idx - 1] != sign[idx]:
            changes[max(0, idx - 1) : min(len(target), idx + 3)] = True
    return changes


def score_series(prediction: np.ndarray, raw: np.ndarray, teacher: np.ndarray) -> float:
    rmse = math.sqrt(float(np.mean((prediction - teacher) ** 2)))
    grad_rmse = math.sqrt(float(np.mean((np.diff(prediction) - np.diff(teacher)) ** 2))) if len(prediction) > 1 else 0.0
    turn_mask = estimate_turn_mask(teacher)
    if np.any(turn_mask):
        turn_rmse = math.sqrt(float(np.mean((prediction[turn_mask] - teacher[turn_mask]) ** 2)))
    else:
        turn_rmse = 0.0
    jitter = float(np.std(np.diff(prediction, n=2))) if len(prediction) > 2 else 0.0
    raw_deviation = float(np.mean(np.abs(prediction - raw)))
    return 0.44 * rmse + 0.24 * grad_rmse + 0.20 * turn_rmse + 0.08 * jitter + 0.04 * raw_deviation


def build_track_bundle(tracks: dict[int, np.ndarray]) -> dict[int, dict[str, np.ndarray]]:
    bundle: dict[int, dict[str, np.ndarray]] = {}
    cv_kalman = ConstantVelocityKalmanSmoother(process_variance=1e-2, measurement_variance=20.0)
    for track_id, values in tracks.items():
        teacher = savgol_teacher(values)
        _, kalman_bottom = cv_kalman.smooth_sequence(values)
        bundle[track_id] = {
            "raw": values,
            "teacher": teacher,
            "cv_kalman": kalman_bottom[:, 0],
        }
    return bundle


def evaluate_candidate(
    track_bundle: dict[int, dict[str, np.ndarray]],
    adaptive_params: dict[str, float | int],
    fixed_lag_frames: int,
) -> dict[str, float]:
    adaptive = AdaptiveRealtimeSmoother(**adaptive_params)
    fixed_lag = FixedLagAdaptiveSmoother(lag_frames=fixed_lag_frames, polyorder=2, **adaptive_params)

    adaptive_scores: list[float] = []
    fixed_lag_scores: list[float] = []
    kalman_scores: list[float] = []
    for bundle in track_bundle.values():
        values = bundle["raw"]
        teacher = bundle["teacher"]
        _, adaptive_bottom = adaptive.smooth_sequence(values)
        _, fixed_lag_bottom = fixed_lag.smooth_sequence(values)

        adaptive_scores.append(score_series(adaptive_bottom[:, 0], values, teacher))
        fixed_lag_scores.append(score_series(fixed_lag_bottom[:, 0], values, teacher))
        kalman_scores.append(score_series(bundle["cv_kalman"], values, teacher))

    return {
        "hybrid_realtime": float(np.mean(adaptive_scores)),
        "fixed_lag_adaptive": float(np.mean(fixed_lag_scores)),
        "cv_kalman_baseline": float(np.mean(kalman_scores)),
        "combined_score": float(0.48 * np.mean(adaptive_scores) + 0.52 * np.mean(fixed_lag_scores)),
    }


def sample_params(rng: random.Random) -> dict[str, float | int]:
    return {
        "outlier_window": rng.choice([5, 7, 9, 11]),
        "outlier_gate": rng.choice([1.9, 2.2, 2.5, 2.8, 3.1]),
        "process_variance": rng.choice([0.03, 0.05, 0.07, 0.09, 0.12, 0.16]),
        "measurement_variance": rng.choice([8.0, 10.0, 12.0, 14.0, 16.0, 18.0]),
        "base_smoothing": rng.choice([0.68, 0.72, 0.76, 0.80, 0.84]),
        "min_smoothing": rng.choice([0.54, 0.58, 0.62, 0.66]),
        "max_smoothing": rng.choice([0.88, 0.90, 0.92, 0.94, 0.96]),
        "turn_responsiveness": rng.choice([0.30, 0.45, 0.60, 0.75]),
        "turn_velocity_damping": rng.choice([0.35, 0.50, 0.65, 0.80]),
        "acceleration_responsiveness": rng.choice([0.10, 0.18, 0.26, 0.34]),
        "skip_update_gate": rng.choice([6.0, 7.0, 8.0, 9.0]),
    }


def normalize_params(params: dict[str, float | int]) -> dict[str, float | int]:
    params = dict(params)
    params["min_smoothing"] = min(float(params["min_smoothing"]), float(params["max_smoothing"]))
    return params


def main() -> int:
    args = build_parser().parse_args()
    tracks = load_tracks(args.input_cache, args.min_track_length, args.max_tracks)
    if not tracks:
        raise ValueError("No eligible tracks found in cache.")
    track_bundle = build_track_bundle(tracks)

    rng = random.Random(args.seed)
    trials: list[dict[str, object]] = []

    default_params = {
        "outlier_window": 7,
        "outlier_gate": 2.4,
        "process_variance": 0.09,
        "measurement_variance": 14.0,
        "base_smoothing": 0.74,
        "min_smoothing": 0.60,
        "max_smoothing": 0.94,
        "turn_responsiveness": 0.58,
        "turn_velocity_damping": 0.60,
        "acceleration_responsiveness": 0.22,
        "skip_update_gate": 7.5,
    }
    default_metrics = evaluate_candidate(track_bundle, default_params, args.fixed_lag_frames)
    trials.append({"label": "default", "params": default_params, "metrics": default_metrics})

    for iteration in range(args.iterations):
        params = normalize_params(sample_params(rng))
        metrics = evaluate_candidate(track_bundle, params, args.fixed_lag_frames)
        trials.append({"label": f"trial_{iteration + 1}", "params": params, "metrics": metrics})

    ranked = sorted(trials, key=lambda item: float(item["metrics"]["combined_score"]))
    best = ranked[0]
    summary = {
        "cache_path": str(args.input_cache),
        "track_count": len(tracks),
        "track_ids": list(tracks.keys()),
        "fixed_lag_frames": args.fixed_lag_frames,
        "baseline": default_metrics,
        "best": best,
        "top_trials": ranked[:10],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Tracks used: {len(tracks)}")
    print(
        "Default:"
        f" hybrid={default_metrics['hybrid_realtime']:.3f},"
        f" fixed_lag={default_metrics['fixed_lag_adaptive']:.3f},"
        f" combined={default_metrics['combined_score']:.3f},"
        f" cv_kalman={default_metrics['cv_kalman_baseline']:.3f}"
    )
    best_metrics = best["metrics"]
    print(
        "Best:"
        f" hybrid={best_metrics['hybrid_realtime']:.3f},"
        f" fixed_lag={best_metrics['fixed_lag_adaptive']:.3f},"
        f" combined={best_metrics['combined_score']:.3f},"
        f" cv_kalman={best_metrics['cv_kalman_baseline']:.3f}"
    )
    print(f"Saved tuning summary to {args.output}")
    print(json.dumps(best["params"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
