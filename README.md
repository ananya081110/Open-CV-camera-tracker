# AI-Based Camera Tracker — First POC

## Goal
A first working webcam prototype for the exact task described in the research:
person pose detection, basic tracking, fall detection, activity dwell time, and zone dwell time.

## Pipeline
Webcam → OpenCV → MediaPipe Pose Landmarker → body landmarks → centroid tracker → posture/fall rules → CSV alert log + fall snapshot

## Setup

Recommended Python: 3.11 or 3.12. MediaPipe 1.x currently triggers a native
Metal crash on macOS, so use the pinned 0.10.x dependency in a compatible venv.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_model.py
python main.py
```

On macOS, allow Terminal/VS Code camera access when prompted.

Press `Q` to stop.

## What the demo should show
- A skeleton over the person.
- `ID 1`, `ID 2`, etc. for tracked people.
- STANDING / SITTING / WALKING / FALLING labels.
- A blue monitoring-zone rectangle.
- Timers for state and zone dwell.
- A terminal alert when a threshold is crossed.
- `logs/events.csv` for event history.
- A snapshot in `alerts/` when a fall is confirmed.

## Tuning
Edit `config.py` to change:
- fall angle and fall confirmation time,
- sitting/standing thresholds,
- dwell thresholds,
- monitoring-zone coordinates,
- tracker distance.

## Limitations
This is a proof of concept, not a production safety system. The fall detector is rule-based and needs testing/calibration. The centroid tracker is intentionally simple. The research recommends upgrading to ByteTrack/DeepSORT and a stronger temporal fall classifier after the core logic is validated.
