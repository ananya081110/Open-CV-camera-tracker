from pathlib import Path
from shutil import copyfileobj
from ssl import create_default_context
from urllib.request import urlopen

import certifi

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "pose_landmarker_full.task"

if __name__ == "__main__":
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100_000:
        print(f"Model already exists: {MODEL_PATH}")
    else:
        print("Downloading MediaPipe Pose Landmarker model...")
        context = create_default_context(cafile=certifi.where())
        with urlopen(MODEL_URL, context=context) as response:
            with MODEL_PATH.open("wb") as output:
                copyfileobj(response, output)
        print(f"Saved model to: {MODEL_PATH}")
