# data_smoothing

Small smoothing playground for noisy player tracking points.

Current smoothers:
- EMA
- Savitzky-Golay
- simple Kalman
- One Euro
- alpha-beta
- constant-velocity Kalman

## Mock Real Data Pipeline

Extract tracked player points from a video into JSONL:

```bash
python scripts/extract_mock_tracking.py \
  --video "C:/Users/moust/Downloads/WhatsApp Video 2026-03-11 at 6.50.27 PM.mp4" \
  --model "C:/Users/moust/Downloads/PITTvsFSU_yolov8m_fine_tuned.pt" \
  --output artifacts/raw_tracks.jsonl \
  --class-name "basketball player"
```

Replay any smoother over that cache:

```bash
python scripts/apply_smoother_to_cache.py \
  --input artifacts/raw_tracks.jsonl \
  --output artifacts/fixed_lag_adaptive.jsonl \
  --method fixed_lag_adaptive
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python run_demo.py
python main.py
```
