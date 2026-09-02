
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTROL_PATH = Path(os.getenv(
    "CAMERA_CONTROL_FILE",
    str(ROOT / "logs" / "camera_control.json")
))

def _default_state():
    return {
        "camera_allowed": True,
        "auto_start": True,
        "camera_index": 0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "updated_by": "system",
    }

def load_camera_control():
    CONTROL_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        if CONTROL_PATH.exists():
            with CONTROL_PATH.open("r", encoding="utf-8") as f:
                state = _default_state()
                state.update(json.load(f))
                return state
    except Exception:
        pass
    state = _default_state()
    save_camera_control(state)
    return state

def save_camera_control(state, updated_by="system"):
    CONTROL_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_state = _default_state()
    new_state.update(state)
    new_state["updated_at"] = datetime.now(timezone.utc).isoformat()
    new_state["updated_by"] = updated_by

    fd, tmp = tempfile.mkstemp(
        prefix="camera_control_", suffix=".json",
        dir=str(CONTROL_PATH.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(new_state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, CONTROL_PATH)
    finally:
        Path(tmp).unlink(missing_ok=True)

    return new_state
