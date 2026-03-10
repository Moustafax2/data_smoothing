# data_smoothing

Small smoothing playground for noisy player tracking points.

Current smoothers:
- EMA
- Savitzky-Golay
- simple Kalman
- One Euro
- alpha-beta
- constant-velocity Kalman

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
