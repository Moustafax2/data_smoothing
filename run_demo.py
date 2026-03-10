from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smoothing.visualize import run_demo


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
        "--alpha",
        type=float,
        default=0.2,
        help="EMA smoothing factor in the range (0, 1].",
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
        "--one-euro-min-cutoff",
        type=float,
        default=0.35,
        help="One Euro minimum cutoff.",
    )
    parser.add_argument(
        "--one-euro-beta",
        type=float,
        default=0.008,
        help="One Euro speed coefficient.",
    )
    parser.add_argument(
        "--one-euro-derivative-cutoff",
        type=float,
        default=0.6,
        help="One Euro derivative cutoff.",
    )
    parser.add_argument(
        "--one-euro-frame-rate",
        type=float,
        default=30.0,
        help="One Euro assumed frame rate.",
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
        "--pid-kp",
        type=float,
        default=0.18,
        help="PID proportional gain.",
    )
    parser.add_argument(
        "--pid-ki",
        type=float,
        default=0.002,
        help="PID integral gain.",
    )
    parser.add_argument(
        "--pid-kd",
        type=float,
        default=0.08,
        help="PID derivative gain.",
    )
    parser.add_argument(
        "--pid-integral-limit",
        type=float,
        default=25.0,
        help="PID integral clamp.",
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
    output_path = run_demo(
        num_points=args.num_points,
        alpha=args.alpha,
        noise_std=args.noise_std,
        seed=args.seed,
        savgol_window_length=args.savgol_window_length,
        savgol_polyorder=args.savgol_polyorder,
        one_euro_min_cutoff=args.one_euro_min_cutoff,
        one_euro_beta=args.one_euro_beta,
        one_euro_derivative_cutoff=args.one_euro_derivative_cutoff,
        one_euro_frame_rate=args.one_euro_frame_rate,
        alpha_beta_alpha=args.alpha_beta_alpha,
        alpha_beta_beta=args.alpha_beta_beta,
        pid_kp=args.pid_kp,
        pid_ki=args.pid_ki,
        pid_kd=args.pid_kd,
        pid_integral_limit=args.pid_integral_limit,
        constant_velocity_process_variance=args.constant_velocity_process_variance,
        constant_velocity_measurement_variance=args.constant_velocity_measurement_variance,
        fixed_lag_frames=args.fixed_lag_frames,
        output_path=args.output,
    )
    print(f"Saved plot to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
