"""
DeepCamera integration bridge for AI Camera Tracker.

This file does NOT reimplement YOLO or DeepCamera detection.
It launches the ACTUAL SharpAI DeepCamera
skills/detection/yolo-detection-2026/scripts/detect.py and talks to it
using the JSONL protocol defined by DeepCamera.

The existing application API is preserved:
    DeepCameraDetector(...).detect(frame)
    DeepCameraDetector(...).status()
    DeepCameraDetector(...).close()

Expected project layout:
    ai_camera_tracker/
      main.py
      deepcamera_adapter.py
      DeepCamera/
        skills/detection/yolo-detection-2026/
          .venv/
          scripts/detect.py
          yolo26n.onnx

Environment overrides:
    DEEPCAMERA_ROOT
    DEEPCAMERA_SKILL
    DEEPCAMERA_PYTHON
    DEEPCAMERA_MODEL_SIZE   (nano/small/medium/large)
    DEEPCAMERA_CONFIDENCE
    DEEPCAMERA_FPS
"""

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
import tempfile

import cv2


def _find_skill():
    explicit = os.getenv("DEEPCAMERA_SKILL", "").strip()
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if p.exists():
            return p

    roots = []
    root = os.getenv("DEEPCAMERA_ROOT", "").strip()
    if root:
        roots.append(Path(root).expanduser())

    here = Path(__file__).resolve().parent
    roots.extend([
        here / "DeepCamera",
        Path.home() / "DeepCamera",
        Path.home() / "deepcamera",
    ])

    for r in roots:
        p = (
            r / "skills" / "detection" / "yolo-detection-2026"
            / "scripts" / "detect.py"
        )
        if p.exists():
            return p.resolve()

    return None


def _skill_python(skill_path: Path):
    explicit = os.getenv("DEEPCAMERA_PYTHON", "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return str(p)

    skill_root = skill_path.parent.parent
    candidates = [
        skill_root / ".venv" / "bin" / "python",
        skill_root / ".venv" / "bin" / "python3",
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    # Last resort. Normally the deployment-created .venv is used.
    return "python3"


class _DeepCameraProcess:
    """Persistent client for the actual DeepCamera detect.py."""

    def __init__(self, model_size, confidence, fps):
        self.skill_path = _find_skill()
        self.model_size = model_size
        self.confidence = float(confidence)
        self.fps = max(0.2, float(fps))

        self.process = None
        self.reader = None
        self.stderr_reader = None

        self.ready = False
        self.ready_event = None
        self.error = ""
        self.lock = threading.Lock()

        self.pending = {}
        self.pending_cv = threading.Condition(self.lock)
        self.frame_id = 0

        self.last_objects = []
        self.last_detection_time = 0.0
        self.min_interval = 1.0 / self.fps

        self._start()

    def _start(self):
        if self.skill_path is None:
            self.error = (
                "Actual DeepCamera skill not found. "
                "Set DEEPCAMERA_ROOT or DEEPCAMERA_SKILL."
            )
            print("[ERROR] " + self.error)
            return

        python_bin = _skill_python(self.skill_path)

        # Empty classes is intentional: the actual DeepCamera skill
        # interprets an empty class list as no filtering, exposing the
        # complete model vocabulary rather than only its default subset.
        params = {
            "model_size": self.model_size,
            "confidence": self.confidence,
            "classes": [],
            "device": "auto",
            "fps": self.fps,
            "use_optimized": True,
        }

        env = os.environ.copy()
        env["AEGIS_SKILL_PARAMS"] = json.dumps(params)
        env["PYTHONUNBUFFERED"] = "1"
        env["YOLO_AUTOINSTALL"] = "0"

        try:
            self.process = subprocess.Popen(
                [python_bin, str(self.skill_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )
        except Exception as exc:
            self.error = f"Could not start DeepCamera: {exc}"
            print("[ERROR] " + self.error)
            return

        self.reader = threading.Thread(
            target=self._read_stdout,
            daemon=True,
            name="DeepCamera-stdout",
        )
        self.reader.start()

        self.stderr_reader = threading.Thread(
            target=self._read_stderr,
            daemon=True,
            name="DeepCamera-stderr",
        )
        self.stderr_reader.start()

        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if self.ready:
                return
            if self.process.poll() is not None:
                self.error = (
                    f"DeepCamera exited during startup "
                    f"(code={self.process.returncode})."
                )
                return
            time.sleep(0.05)

        self.error = "Timed out waiting for DeepCamera ready event."

    def _read_stdout(self):
        try:
            for line in self.process.stdout:
                line = line.strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    # The actual skill reserves stdout for JSONL, so
                    # malformed/non-JSON lines are ignored defensively.
                    continue

                event_type = event.get("event")

                if event_type == "ready":
                    with self.lock:
                        self.ready = True
                        self.ready_event = event
                        self.error = ""
                    print(
                        "[DeepCamera] READY | "
                        f"model={event.get('model')} | "
                        f"device={event.get('device')} | "
                        f"backend={event.get('backend')} | "
                        f"format={event.get('format')} | "
                        f"classes={event.get('classes')}"
                    )

                elif event_type == "detections":
                    frame_id = event.get("frame_id")
                    with self.pending_cv:
                        self.pending[frame_id] = event
                        self.pending_cv.notify_all()

                elif event_type == "error":
                    frame_id = event.get("frame_id")
                    message = str(event.get("message", "DeepCamera error"))
                    with self.pending_cv:
                        self.error = message
                        if frame_id is not None:
                            self.pending[frame_id] = event
                        self.pending_cv.notify_all()

                elif event_type == "perf_stats":
                    timings = event.get("timings_ms", {})
                    total = timings.get("total", {})
                    if total:
                        print(
                            "[DeepCamera] PERF | "
                            f"frames={event.get('total_frames')} | "
                            f"inference={timings.get('inference', {}).get('avg', '-')}"
                            f"ms | total={total.get('avg', '-')}ms"
                        )

        except Exception as exc:
            with self.lock:
                self.error = str(exc)

    def _read_stderr(self):
        try:
            for line in self.process.stderr:
                line = line.strip()
                if line:
                    print("[DeepCamera] " + line)
        except Exception:
            pass

    @staticmethod
    def _write_frame(frame, frame_id):
        directory = (
            Path(tempfile.gettempdir())
            / "ai_camera_tracker_deepcamera"
        )
        directory.mkdir(parents=True, exist_ok=True)

        path = directory / f"frame_{frame_id}.jpg"
        ok = cv2.imwrite(
            str(path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )
        return path if ok else None

    def detect(self, frame, camera_id="ai_camera_tracker"):
        now = time.monotonic()

        if self.process is None or self.process.poll() is not None:
            return list(self.last_objects)

        if not self.ready:
            return list(self.last_objects)

        # Match DeepCamera's configured processing FPS instead of
        # running the skill once for every webcam frame.
        if (
            self.last_detection_time
            and now - self.last_detection_time < self.min_interval
        ):
            return list(self.last_objects)

        self.last_detection_time = now
        self.frame_id += 1
        frame_id = self.frame_id

        frame_path = self._write_frame(frame, frame_id)
        if frame_path is None:
            self.error = "Could not write frame for DeepCamera."
            return list(self.last_objects)

        message = {
            "event": "frame",
            "frame_id": frame_id,
            "camera_id": camera_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "frame_path": str(frame_path),
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
        }

        try:
            with self.lock:
                self.pending.pop(frame_id, None)

            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()
        except Exception as exc:
            self.error = str(exc)
            try:
                frame_path.unlink(missing_ok=True)
            except Exception:
                pass
            return list(self.last_objects)

        event = None
        deadline = time.monotonic() + 2.0

        with self.pending_cv:
            while time.monotonic() < deadline:
                event = self.pending.pop(frame_id, None)
                if event is not None:
                    break
                self.pending_cv.wait(timeout=0.01)

        # The skill reads the frame synchronously before returning its
        # detection event, so it is safe to remove the shared file now.
        try:
            frame_path.unlink(missing_ok=True)
        except Exception:
            pass

        if not event or event.get("event") != "detections":
            return list(self.last_objects)

        objects = []
        for obj in event.get("objects", []):
            try:
                bbox = obj["bbox"]
                if len(bbox) != 4:
                    continue

                objects.append({
                    "class_name": str(obj["class"]),
                    "confidence": float(obj["confidence"]),
                    "bbox": [int(v) for v in bbox],
                    "center": [
                        int((bbox[0] + bbox[2]) / 2),
                        int((bbox[1] + bbox[3]) / 2),
                    ],
                    "classification_source": "DeepCamera-YOLO2026",
                    "raw_class_name": str(obj["class"]),
                    "raw_confidence": float(obj["confidence"]),
                    "verified": False,
                })
            except (KeyError, TypeError, ValueError):
                continue

        self.last_objects = objects
        return list(objects)

    def status(self):
        if self.process is None:
            return "DeepCamera: OFF"
        if self.process.poll() is not None:
            return "DeepCamera: ERROR"
        if self.ready:
            if self.ready_event:
                return (
                    "DeepCamera: ACTIVE | "
                    f"{self.ready_event.get('model', 'YOLO2026')} | "
                    f"{self.ready_event.get('format', 'runtime')} | "
                    f"{self.ready_event.get('device', 'auto')}"
                )
            return "DeepCamera: ACTIVE"
        return "DeepCamera: STARTING"

    def close(self):
        if self.process is None:
            return

        try:
            self.process.stdin.write(
                json.dumps({"command": "stop"}) + "\n"
            )
            self.process.stdin.flush()
        except Exception:
            pass

        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass


class DeepCameraDetector:
    """
    Public compatibility wrapper.

    DeepCamera is the primary detector. The optional fallback is used
    only when the actual skill has no usable result, preserving the
    existing application if the external skill temporarily fails.
    """

    def __init__(
        self,
        model_path="yolo26n.pt",
        confidence=0.30,
        iou=0.45,
        classes=None,
        fallback=None,
        imgsz=640,
        max_det=300,
        vlm_url=None,
        vlm_model="qwen3-vl:8b",
    ):
        self.model_path = model_path
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.classes = classes
        self.fallback = fallback
        self.imgsz = int(imgsz)
        self.max_det = int(max_det)

        model_size = os.getenv(
            "DEEPCAMERA_MODEL_SIZE",
            "nano",
        )
        fps = float(os.getenv("DEEPCAMERA_FPS", "5"))

        self._deepcamera = _DeepCameraProcess(
            model_size=model_size,
            confidence=self.confidence,
            fps=fps,
        )

        # VLM parameters are retained for compatibility with the current
        # main.py. VLM is intentionally not mixed into the actual
        # DeepCamera YOLO skill.
        self.vlm_url = vlm_url
        self.vlm_model = vlm_model

        print(
            "[INFO] ACTUAL DeepCamera YOLO-2026 skill connected."
        )
        print(
            f"[INFO] Skill: {self._deepcamera.skill_path}"
        )
        print(
            f"[INFO] Model size: {model_size} | FPS: {fps}"
        )

    def detect(self, frame, timestamp_ms=None):
        objects = self._deepcamera.detect(frame)

        # Preserve existing functionality if DeepCamera is temporarily
        # unavailable, but never replace DeepCamera as the primary path.
        if not objects and self.fallback is not None:
            try:
                objects = self.fallback.detect(frame)
                for obj in objects:
                    obj.setdefault(
                        "classification_source",
                        "legacy-fallback",
                    )
            except Exception:
                objects = []

        return objects

    def status(self):
        return self._deepcamera.status()

    def close(self):
        self._deepcamera.close()


__all__ = ["DeepCameraDetector"]
