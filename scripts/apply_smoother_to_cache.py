from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smoothing.filters import available_smoothers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a smoother over cached tracked points.")
    parser.add_argument("--input", type=Path, required=True, help="Input raw JSONL cache.")
    parser.add_argument("--output", type=Path, required=True, help="Output smoothed JSONL path.")
    parser.add_argument("--method", type=str, default="fixed_lag_adaptive", help="Smoother name from registry.")
    parser.add_argument("--fixed-lag-frames", type=int, default=4, help="Lag for fixed-lag smoothers.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    smoothers = available_smoothers(
        fixed_lag_frames=args.fixed_lag_frames,
    )
    try:
        smoother = smoothers[args.method]
    except KeyError as exc:
        available = ", ".join(sorted(smoothers))
        raise ValueError(f"Unknown smoother '{args.method}'. Available: {available}") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with (
        args.input.open("r", encoding="utf-8") as input_handle,
        args.output.open("w", encoding="utf-8") as output_handle,
    ):
        for line in input_handle:
            if not line.strip():
                continue
            frame = json.loads(line)
            current_ids: set[int] = set()
            smoothed_players: list[dict[str, object]] = []

            for player in frame["players"]:
                track_id = int(player["track_id"])
                current_ids.add(track_id)
                smoothed_top, smoothed_bottom = smoother.smooth_points(
                    track_id,
                    tuple(player["top_point"]),
                    tuple(player["bottom_point"]),
                )
                smoothed_players.append(
                    {
                        "track_id": track_id,
                        "bbox": player.get("bbox"),
                        "confidence": player.get("confidence"),
                        "top_point": [float(smoothed_top[0]), float(smoothed_top[1])],
                        "bottom_point": [float(smoothed_bottom[0]), float(smoothed_bottom[1])],
                        "raw_top_point": player["top_point"],
                        "raw_bottom_point": player["bottom_point"],
                    }
                )

            smoother.cleanup_old_players(current_ids)
            output_handle.write(
                json.dumps(
                    {
                        "frame_idx": frame["frame_idx"],
                        "timestamp_s": frame["timestamp_s"],
                        "players": smoothed_players,
                    }
                )
                + "\n"
            )

    print(f"Wrote smoothed tracking cache to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
