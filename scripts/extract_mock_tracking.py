from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import supervision as sv
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def get_top_center_of_bbox(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, _ = bbox
    return ((x1 + x2) / 2.0, y1)


def get_bottom_center_of_bbox(bbox: list[float]) -> tuple[float, float]:
    x1, _, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def resolve_class_id(model: YOLO, class_name: str | None, class_id: int | None) -> int | None:
    if class_id is not None:
        return class_id
    if class_name is None:
        return None

    names = model.names
    for idx, name in names.items():
        if str(name).lower() == class_name.lower():
            return int(idx)
    available = ", ".join(str(name) for name in names.values())
    raise ValueError(f"Unknown class name '{class_name}'. Available classes: {available}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract tracked player points into JSONL.")
    parser.add_argument("--video", type=Path, required=True, help="Input video path.")
    parser.add_argument("--model", type=Path, required=True, help="Ultralytics YOLO model path.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--class-name", type=str, default="basketball player", help="Target class name.")
    parser.add_argument("--class-id", type=int, default=None, help="Target class id override.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Ultralytics inference image size.")
    parser.add_argument("--device", type=str, default=None, help="Inference device, e.g. cpu or 0.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap for quick tests.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    target_class_id = resolve_class_id(model, args.class_name, args.class_id)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = sv.ByteTrack(
        track_activation_threshold=0.15,
        lost_track_buffer=60,
        frame_rate=round(fps),
        minimum_matching_threshold=0.85,
    )

    frame_idx = 0
    with args.output.open("w", encoding="utf-8") as handle:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break

            result = model.predict(
                source=frame_bgr,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]

            detections = sv.Detections.from_ultralytics(result)
            if target_class_id is not None and len(detections) > 0:
                detections = detections[detections.class_id == target_class_id]

            tracked = tracker.update_with_detections(detections)
            players: list[dict[str, object]] = []
            for bbox, _, confidence, class_id, track_id, _ in tracked:
                bbox_list = [float(value) for value in bbox]
                players.append(
                    {
                        "track_id": int(track_id),
                        "class_id": int(class_id) if class_id is not None else None,
                        "confidence": float(confidence) if confidence is not None else None,
                        "bbox": bbox_list,
                        "top_point": list(get_top_center_of_bbox(bbox_list)),
                        "bottom_point": list(get_bottom_center_of_bbox(bbox_list)),
                    }
                )

            row = {
                "frame_idx": frame_idx,
                "timestamp_s": frame_idx / float(fps),
                "players": players,
            }
            handle.write(json.dumps(row) + "\n")
            frame_idx += 1

    capture.release()
    print(f"Wrote raw tracking cache to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
