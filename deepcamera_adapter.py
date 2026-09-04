"""
Local DeepCamera-style object detection engine for AI Camera Tracker.

This module is self-contained inside the AI Camera Tracker project.
It does NOT import, clone, launch, or depend on the SharpAI/DeepCamera
repository.

It implements the useful DeepCamera detection-layer behavior directly:
- YOLO26 object detection
- configurable confidence / image size / max detections
- automatic Apple Silicon MPS selection when available
- optional ONNX model support
- frame-rate governor for camera streams
- lightweight IoU tracking with stable object IDs
- temporal confirmation to reduce one-frame false positives
- normalized application output used by main.py
- optional legacy fallback for backward compatibility

The model itself is downloaded by Ultralytics if it is not present.
The project therefore depends on the YOLO26 model/package, not on another
repository.

Public API preserved for the existing main.py:
    DeepCameraDetector(...).detect(frame)
    DeepCameraDetector(...).status()
    DeepCameraDetector(...).close()
"""

from __future__ import annotations

import os
import platform
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _iou(a, b) -> float:
    try:
        ax1, ay1, ax2, ay2 = map(float, a)
        bx1, by1, bx2, by2 = map(float, b)
    except Exception:
        return 0.0

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = (
        max(0.0, ix2 - ix1)
        * max(0.0, iy2 - iy1)
    )

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )
    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _clean_label(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


class _ObjectTrack:
    def __init__(self, track_id: int, obj: Dict[str, Any]):
        self.id = track_id
        self.bbox = list(obj["bbox"])
        self.class_name = obj["class_name"]
        self.confidence = float(obj["confidence"])
        self.last_seen = time.monotonic()
        self.missed = 0


class DeepCameraDetector:
    """
    Project-local DeepCamera-style detection engine.

    The class name is retained so the existing application does not
    need a broad refactor. Internally, this is now YOUR detector:
    YOLO26 + frame governor + temporal confirmation + IoU tracking.

    No SharpAI/DeepCamera repository is required at runtime.
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
        fps=5,
        temporal_hits=2,
        temporal_window=3,
        track_iou=0.25,
        max_missed=8,
    ):
        self.model_path = str(model_path or "yolo26n.pt")
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.classes = classes
        self.fallback = fallback
        self.imgsz = int(imgsz)
        self.max_det = int(max_det)

        self.fps = max(
            0.2,
            float(os.getenv("DEEPCAMERA_FPS", str(fps))),
        )
        self.min_interval = 1.0 / self.fps

        self.temporal_hits_required = max(1, int(temporal_hits))
        self.temporal_window = max(1, int(temporal_window))
        self.track_iou = float(track_iou)
        self.max_missed = int(max_missed)

        self.model = None
        self.model_kind = None
        self.device = "cpu"

        self.last_objects: List[Dict[str, Any]] = []
        self.last_detection_time = 0.0
        self.last_inference_ms = 0.0
        self.total_frames = 0
        self.total_detections = 0
        self.last_error = ""

        self.next_track_id = 1
        self.tracks: Dict[int, _ObjectTrack] = {}

        # Per-track temporal history: recent class/bbox confirmations.
        self.histories: Dict[int, deque] = {}

        self._load_model()

    # ============================================================
    # MODEL / HARDWARE
    # ============================================================

    def _select_device(self) -> str:
        forced = os.getenv("DEEPCAMERA_DEVICE", "").strip().lower()

        if forced:
            return forced

        try:
            import torch

            if platform.system() == "Darwin" and torch.backends.mps.is_available():
                return "mps"

            if torch.cuda.is_available():
                return "cuda"

        except Exception:
            pass

        return "cpu"

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            self.last_error = (
                "Ultralytics is not installed. "
                "Run: pip install -U ultralytics"
            )
            print(f"[ERROR] {self.last_error} ({exc})")
            return

        try:
            self.device = self._select_device()

            requested = Path(
                os.path.expanduser(self.model_path)
            )

            # Prefer a project-local model if the supplied path exists.
            # Otherwise allow Ultralytics to download the official YOLO26
            # checkpoint by name.
            model_source = (
                str(requested)
                if requested.exists()
                else self.model_path
            )

            self.model = YOLO(model_source)
            self.model_kind = "YOLO26"

            print(
                "[INFO] Local DeepCamera detection engine ready: "
                f"{self.model_path}"
            )
            print(
                "[INFO] Detection device: "
                f"{self.device}"
            )
            print(
                "[INFO] Inference: "
                f"{self.imgsz}px | confidence={self.confidence:.2f} | "
                f"FPS governor={self.fps:g}"
            )

            self.last_error = ""

        except Exception as exc:
            self.model = None
            self.model_kind = None
            self.last_error = f"YOLO26 model load failed: {exc}"
            print(f"[ERROR] {self.last_error}")

    # ============================================================
    # DETECTION
    # ============================================================

    def _predict(self, frame) -> List[Dict[str, Any]]:
        if self.model is None:
            return []

        kwargs = {
            "source": frame,
            "conf": self.confidence,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "max_det": self.max_det,
            "verbose": False,
        }

        # Explicit class filtering is optional. None/empty means all
        # classes supported by the selected YOLO26 checkpoint.
        if self.classes:
            names = getattr(self.model, "names", {}) or {}
            requested = {
                str(value).strip().lower()
                for value in self.classes
            }

            class_ids = [
                int(class_id)
                for class_id, name in names.items()
                if str(name).strip().lower() in requested
            ]

            if class_ids:
                kwargs["classes"] = class_ids

        if self.device in {"mps", "cuda"}:
            kwargs["device"] = self.device
        else:
            kwargs["device"] = "cpu"

        results = self.model.predict(**kwargs)

        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)

        if boxes is None:
            return []

        names = getattr(result, "names", None)
        if names is None:
            names = getattr(self.model, "names", {}) or {}

        objects = []

        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
        except Exception:
            return []

        for box, score, class_id in zip(
            xyxy,
            confs,
            class_ids,
        ):
            x1, y1, x2, y2 = map(int, box)

            if x2 <= x1 or y2 <= y1:
                continue

            if isinstance(names, dict):
                label = names.get(int(class_id), str(class_id))
            else:
                label = (
                    names[int(class_id)]
                    if 0 <= int(class_id) < len(names)
                    else str(class_id)
                )

            objects.append(
                {
                    "class_name": _clean_label(label),
                    "confidence": _clamp(float(score), 0.0, 1.0),
                    "bbox": [x1, y1, x2, y2],
                    "classification_source": "AI-Camera-Tracker-YOLO26",
                    "raw_class_name": _clean_label(label),
                    "raw_confidence": float(score),
                    "verified": False,
                }
            )

        return objects

    # ============================================================
    # TEMPORAL + IOU TRACKING
    # ============================================================

    def _match_tracks(self, objects: List[Dict[str, Any]]) -> None:
        now = time.monotonic()
        unmatched_tracks = set(self.tracks.keys())
        assignments = []

        # Greedy highest-IoU matching.
        candidates = []

        for obj_index, obj in enumerate(objects):
            for track_id, track in self.tracks.items():
                overlap = _iou(obj["bbox"], track.bbox)

                if overlap >= self.track_iou:
                    same_class = (
                        obj["class_name"].lower()
                        == track.class_name.lower()
                    )
                    score = overlap + (0.10 if same_class else 0.0)
                    candidates.append(
                        (score, obj_index, track_id)
                    )

        candidates.sort(reverse=True)

        used_objects = set()

        for _, obj_index, track_id in candidates:
            if obj_index in used_objects:
                continue
            if track_id not in unmatched_tracks:
                continue

            obj = objects[obj_index]
            track = self.tracks[track_id]

            obj["track_id"] = track_id
            track.bbox = list(obj["bbox"])
            track.class_name = obj["class_name"]
            track.confidence = float(obj["confidence"])
            track.last_seen = now
            track.missed = 0

            used_objects.add(obj_index)
            unmatched_tracks.discard(track_id)
            assignments.append((obj_index, track_id))

        # Create new tracks for unmatched detections.
        for obj_index, obj in enumerate(objects):
            if obj_index in used_objects:
                continue

            track_id = self.next_track_id
            self.next_track_id += 1

            self.tracks[track_id] = _ObjectTrack(
                track_id,
                obj,
            )

            self.histories[track_id] = deque(
                maxlen=self.temporal_window
            )

            obj["track_id"] = track_id

        # Age unmatched existing tracks.
        for track_id in list(unmatched_tracks):
            track = self.tracks.get(track_id)
            if track is None:
                continue

            track.missed += 1

            if track.missed > self.max_missed:
                self.tracks.pop(track_id, None)
                self.histories.pop(track_id, None)

        # Update temporal history and expose confirmation metadata.
        for obj in objects:
            track_id = obj.get("track_id")
            if track_id is None:
                continue

            history = self.histories.setdefault(
                track_id,
                deque(maxlen=self.temporal_window),
            )

            history.append(
                {
                    "class_name": obj["class_name"].lower(),
                    "bbox": list(obj["bbox"]),
                    "confidence": float(obj["confidence"]),
                    "timestamp": now,
                }
            )

            same_class_hits = sum(
                1
                for item in history
                if item["class_name"]
                == obj["class_name"].lower()
            )

            obj["temporal_hits"] = same_class_hits
            obj["temporal_confirmed"] = (
                same_class_hits >= self.temporal_hits_required
            )

        # A new object is shown immediately for responsive UI, but
        # temporal confirmation is available to the application for
        # event/security decisions.
        for obj in objects:
            if "track_id" not in obj:
                obj["track_id"] = None

            obj["center"] = [
                int((obj["bbox"][0] + obj["bbox"][2]) / 2),
                int((obj["bbox"][1] + obj["bbox"][3]) / 2),
            ]

    # ============================================================
    # PUBLIC API
    # ============================================================

    def detect(self, frame, timestamp_ms=None):
        if frame is None:
            return list(self.last_objects)

        now = time.monotonic()

        # Frame governor: keep the camera/UI loop responsive.
        if (
            self.last_detection_time
            and now - self.last_detection_time < self.min_interval
        ):
            return list(self.last_objects)

        self.last_detection_time = now
        self.total_frames += 1

        start = time.perf_counter()

        try:
            objects = self._predict(frame)
            self.last_inference_ms = (
                time.perf_counter() - start
            ) * 1000.0

            if objects:
                self._match_tracks(objects)
                self.last_objects = objects
                self.total_detections += len(objects)
                self.last_error = ""
                return list(objects)

            # If the YOLO26 engine is healthy but there are genuinely
            # no detections, return an empty list instead of invoking
            # the legacy detector unnecessarily.
            if self.model is not None:
                self.last_objects = []
                return []

        except Exception as exc:
            self.last_error = str(exc)
            print(
                "[WARNING] Local YOLO26 inference failed: "
                f"{exc}"
            )

        # Backward compatibility only: the existing application's
        # detector can still protect the rest of the monitoring
        # pipeline if this new engine cannot run.
        if self.fallback is not None:
            try:
                objects = self.fallback.detect(frame)

                for obj in objects:
                    obj.setdefault(
                        "classification_source",
                        "legacy-fallback",
                    )

                self.last_objects = list(objects)
                return list(objects)

            except Exception as exc:
                self.last_error = (
                    f"{self.last_error}; fallback failed: {exc}"
                )

        return list(self.last_objects)

    def status(self) -> str:
        if self.model is None:
            return "YOLO26: OFFLINE"

        if self.last_error:
            return "YOLO26: ERROR"

        return (
            "YOLO26: ACTIVE | "
            f"{self.device.upper()} | "
            f"{self.fps:g} FPS"
        )

    def stats(self) -> Dict[str, Any]:
        return {
            "engine": "YOLO26",
            "device": self.device,
            "model": self.model_path,
            "confidence": self.confidence,
            "imgsz": self.imgsz,
            "max_det": self.max_det,
            "fps": self.fps,
            "inference_ms": round(self.last_inference_ms, 2),
            "frames_processed": self.total_frames,
            "detections": self.total_detections,
            "tracks": len(self.tracks),
            "temporal_hits_required": self.temporal_hits_required,
        }

    def close(self):
        self.model = None
        self.tracks.clear()
        self.histories.clear()
        self.last_objects = []


__all__ = ["DeepCameraDetector"]
