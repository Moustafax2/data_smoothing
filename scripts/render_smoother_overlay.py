from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smoothing.filters import available_smoothers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render smoothed bottom points onto a video.")
    parser.add_argument("--video", type=Path, required=True, help="Input video path.")
    parser.add_argument("--input-cache", type=Path, required=True, help="Raw tracking JSONL cache.")
    parser.add_argument("--output", type=Path, required=True, help="Output rendered video path.")
    parser.add_argument("--method", type=str, default="fixed_lag_adaptive", help="Smoother name from registry.")
    parser.add_argument("--track-id", type=int, default=None, help="Optional single track id to render.")
    parser.add_argument("--fixed-lag-frames", type=int, default=4, help="Lag for fixed-lag smoothers.")
    parser.add_argument("--marker-color", type=str, default="0,255,255", help="BGR marker color, e.g. 0,255,255.")
    return parser


def parse_color(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("Color must be three comma-separated ints in BGR order.")
    return parts[0], parts[1], parts[2]


def iter_cache_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def draw_marker(frame, point: tuple[float, float], label: str, color: tuple[int, int, int]) -> None:
    x = int(round(point[0]))
    y = int(round(point[1])) + 12
    tip = (x, y)
    center = (x, y + 16)

    cv2.line(frame, tip, center, color, 2, lineType=cv2.LINE_AA)
    cv2.circle(frame, center, 7, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(frame, center, 11, color, thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(
        frame,
        label,
        (center[0] + 10, center[1] + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        lineType=cv2.LINE_AA,
    )


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

    color = parse_color(args.marker_color)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {args.output}")

    frame_idx = 0
    for row in iter_cache_rows(args.input_cache):
        ok, frame = capture.read()
        if not ok:
            break

        current_ids: set[int] = set()
        for player in row["players"]:
            track_id = int(player["track_id"])
            if args.track_id is not None and track_id != args.track_id:
                continue

            current_ids.add(track_id)
            smoothed_top, smoothed_bottom = smoother.smooth_points(
                track_id,
                tuple(player["top_point"]),
                tuple(player["bottom_point"]),
            )
            draw_marker(
                frame,
                (float(smoothed_bottom[0]), float(smoothed_bottom[1])),
                label=f"ID {track_id}",
                color=color,
            )

        smoother.cleanup_old_players(current_ids)
        cv2.putText(
            frame,
            f"{args.method} | frame {frame_idx}",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        writer.write(frame)
        frame_idx += 1

    capture.release()
    writer.release()
    print(f"Wrote overlay video to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
