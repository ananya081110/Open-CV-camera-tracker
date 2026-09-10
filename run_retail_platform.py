"""Start the multi-camera retail engine and FastAPI backend together.

Camera configuration is supplied through RETAIL_CAMERAS_JSON, for example:
[
  {"camera_id":"CAM_01","source":"0","zones":{"Entrance":[0,0,1,1]}},
  {"camera_id":"CAM_02","source":"sample_tv.mp4","zones":{"TV":[0.45,0.20,0.90,0.90]}}
]
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import uvicorn

from retail_event_pipeline import RetailEventPipeline
from retail_multicamera import CameraWorkerConfig, RetailMultiCameraEngine
from retail_api import create_app


def load_cameras():
    raw = os.getenv("RETAIL_CAMERAS_JSON", "")
    if not raw:
        # Safe development default. It does not claim a real store has a
        # camera until the user supplies actual sources.
        return [
            CameraWorkerConfig(
                camera_id="CAM_01",
                source="0",
                zones={"Entrance": (0.0, 0.0, 1.0, 1.0)},
            )
        ]

    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("RETAIL_CAMERAS_JSON must be a JSON list")

    cameras = []
    for item in payload:
        cameras.append(
            CameraWorkerConfig(
                camera_id=str(item["camera_id"]),
                source=item["source"],
                zones={
                    str(name): tuple(float(v) for v in box)
                    for name, box in item.get("zones", {}).items()
                    if len(box) == 4
                },
                enabled=bool(item.get("enabled", True)),
                frame_width=int(item.get("frame_width", 1280)),
                frame_height=int(item.get("frame_height", 720)),
            )
        )
    return cameras


def main():
    cameras = load_cameras()
    pipeline = RetailEventPipeline(
        event_file=str(Path("logs") / "retail_events.jsonl")
    )
    engine = RetailMultiCameraEngine(
        cameras=cameras,
        on_customer=pipeline.observe_customer,
    )
    app = create_app(engine=engine, pipeline=pipeline)

    engine.start()
    print(f"[INFO] Retail AI platform started with {len(cameras)} configured cameras")
    print("[INFO] FastAPI: http://127.0.0.1:8000")
    print("[INFO] API docs: http://127.0.0.1:8000/docs")

    try:
        uvicorn.run(app, host="127.0.0.1", port=8000)
    finally:
        engine.stop()
        print("[INFO] Retail AI platform stopped")


if __name__ == "__main__":
    main()
