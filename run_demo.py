from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smoothing.visualize import run_cache_demo, run_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate noisy x-points and visualize EMA smoothing.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=500,
        help="Number of synthetic points to generate.",
    )
    parser.add_argument(
        "--input-cache",
        type=Path,
        default=None,
        help="Optional raw tracking JSONL cache to visualize instead of synthetic data.",
    )
    parser.add_argument(
        "--track-id",
        type=int,
        default=None,
        help="Specific track id from the cache. Defaults to the most frequent track.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=4.5,
        help="Standard deviation of the synthetic jitter.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible synthetic data.",
    )
    parser.add_argument(
        "--savgol-window-length",
        type=int,
        default=51,
        help="Savitzky-Golay odd window length.",
    )
    parser.add_argument(
        "--savgol-polyorder",
        type=int,
        default=2,
        help="Savitzky-Golay polynomial order.",
    )
    parser.add_argument(
        "--alpha-beta-alpha",
        type=float,
        default=0.18,
        help="Alpha-beta filter alpha gain.",
    )
    parser.add_argument(
        "--alpha-beta-beta",
        type=float,
        default=0.004,
        help="Alpha-beta filter beta gain.",
    )
    parser.add_argument(
        "--constant-velocity-process-variance",
        type=float,
        default=1e-2,
        help="Constant-velocity Kalman process variance.",
    )
    parser.add_argument(
        "--constant-velocity-measurement-variance",
        type=float,
        default=20.0,
        help="Constant-velocity Kalman measurement variance.",
    )
    parser.add_argument(
        "--fixed-lag-frames",
        type=int,
        default=4,
        help="Future frames available to the fixed-lag adaptive smoother.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "smoothing_comparison.png",
        help="Where to save the visualization image.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    kwargs = dict(
        savgol_window_length=args.savgol_window_length,
        savgol_polyorder=args.savgol_polyorder,
        alpha_beta_alpha=args.alpha_beta_alpha,
        alpha_beta_beta=args.alpha_beta_beta,
        constant_velocity_process_variance=args.constant_velocity_process_variance,
        constant_velocity_measurement_variance=args.constant_velocity_measurement_variance,
        fixed_lag_frames=args.fixed_lag_frames,
        output_path=args.output,
    )
    if args.input_cache is not None:
        output_path = run_cache_demo(
            cache_path=args.input_cache,
            track_id=args.track_id,
            **kwargs,
        )
    else:
        output_path = run_demo(
            num_points=args.num_points,
            noise_std=args.noise_std,
            seed=args.seed,
            **kwargs,
        )
    print(f"Saved plot to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
