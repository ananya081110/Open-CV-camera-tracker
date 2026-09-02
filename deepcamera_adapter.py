"""
DeepCamera integration layer for AI Camera Tracker.

This adapter uses the current DeepCamera detection design:
- YOLO 2026 / YOLO26 object detection
- 80+ COCO classes
- normalized output dictionaries compatible with this project
- optional external DeepCamera/Aegis JSONL bridge via DEEPCAMERA_SKILL_CMD

Local mode is the practical default for this repository: it runs a
DeepCamera-compatible YOLO26 model directly in the same Python process.
If YOLO26 cannot be loaded, it can fall back to the existing detector.
"""

import json
import os
import shlex
import subprocess
from pathlib import Path

import cv2


class DeepCameraDetector:
    """DeepCamera-compatible object detector.

    Output format:
        {
            "class_name": str,
            "confidence": float,
            "bbox": [x1, y1, x2, y2],
            "center": [cx, cy],
        }
    """

    def __init__(
        self,
        model_path="yolo26n.pt",
        confidence=0.50,
        iou=0.45,
        classes=None,
        fallback=None,
    ):
        self.model_path = model_path
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.classes = classes
        self.fallback = fallback

        self.model = None
        self.external_process = None
        self.external_mode = False
        self.frame_id = 0

        # -----------------------------------------------------
        # Optional external DeepCamera skill process.
        #
        # Example:
        # DEEPCAMERA_SKILL_CMD="python /path/to/detect.py"
        #
        # The current DeepCamera skill protocol is JSONL:
        # frame -> detections.
        # -----------------------------------------------------
        command = os.getenv("DEEPCAMERA_SKILL_CMD", "").strip()

        if command:
            self._start_external_skill(command)

        # -----------------------------------------------------
        # Default: local DeepCamera-compatible YOLO26
        # -----------------------------------------------------
        if not self.external_mode:
            self._load_local_model()

    def _start_external_skill(self, command):
        try:
            self.external_process = subprocess.Popen(
                shlex.split(command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self.external_mode = True
            print(
                "[INFO] DeepCamera external JSONL skill connected."
            )
        except Exception as exc:
            self.external_process = None
            self.external_mode = False
            print(
                "[WARNING] Could not start DeepCamera skill: "
                f"{exc}"
            )

    def _load_local_model(self):
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)

            print(
                "[INFO] DeepCamera YOLO26 detector ready: "
                f"{self.model_path}"
            )
        except Exception as exc:
            self.model = None
            print(
                "[WARNING] DeepCamera YOLO26 model unavailable: "
                f"{exc}"
            )

            if self.fallback is None:
                raise

            print(
                "[INFO] Falling back to the existing object detector."
            )

    @staticmethod
    def _normalize_detection(box, class_name, confidence):
        x1, y1, x2, y2 = [float(v) for v in box]

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        return {
            "class_name": str(class_name),
            "confidence": float(confidence),
            "bbox": [
                int(round(x1)),
                int(round(y1)),
                int(round(x2)),
                int(round(y2)),
            ],
            "center": [
                int(round(cx)),
                int(round(cy)),
            ],
        }

    def _detect_local(self, frame):
        if self.model is None:
            if self.fallback is not None:
                return self.fallback.detect(frame)
            return []

        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                iou=self.iou,
                classes=self.classes,
                verbose=False,
            )
        except Exception as exc:
            print(
                "[WARNING] DeepCamera YOLO26 inference failed: "
                f"{exc}"
            )
            if self.fallback is not None:
                return self.fallback.detect(frame)
            return []

        detections = []

        if not results:
            return detections

        result = results[0]
        names = result.names

        if result.boxes is None:
            return detections

        for box in result.boxes:
            try:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = names[class_id]

                detections.append(
                    self._normalize_detection(
                        box.xyxy[0].tolist(),
                        class_name,
                        confidence,
                    )
                )
            except Exception:
                continue

        return detections

    def _detect_external(self, frame):
        """Send a frame to an external DeepCamera JSONL skill.

        DeepCamera's current skill protocol accepts a frame_path rather
        than raw image bytes. We therefore write one temporary JPEG.
        """
        if (
            self.external_process is None
            or self.external_process.stdin is None
            or self.external_process.stdout is None
        ):
            return []

        temp_dir = Path(
            os.getenv(
                "DEEPCAMERA_FRAME_DIR",
                "/tmp/deepcamera_detection",
            )
        )
        temp_dir.mkdir(parents=True, exist_ok=True)

        self.frame_id += 1
        frame_path = temp_dir / f"frame_{self.frame_id}.jpg"

        if not cv2.imwrite(str(frame_path), frame):
            return []

        message = {
            "event": "frame",
            "frame_id": self.frame_id,
            "camera_id": "local_webcam",
            "timestamp": str(self.frame_id),
            "frame_path": str(frame_path),
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
        }

        try:
            self.external_process.stdin.write(
                json.dumps(message) + "\n"
            )
            self.external_process.stdin.flush()

            line = self.external_process.stdout.readline().strip()

            if not line:
                return []

            payload = json.loads(line)

            if payload.get("event") != "detections":
                return []

            detections = []

            for obj in payload.get("objects", []):
                label = obj.get(
                    "class",
                    obj.get("class_name", "object"),
                )
                confidence = obj.get("confidence", 0.0)
                bbox = obj.get("bbox")

                if bbox and len(bbox) == 4:
                    detections.append(
                        self._normalize_detection(
                            bbox,
                            label,
                            confidence,
                        )
                    )

            return detections

        except Exception as exc:
            print(
                "[WARNING] DeepCamera JSONL detection failed: "
                f"{exc}"
            )
            return []

    def detect(self, frame):
        if self.external_mode:
            return self._detect_external(frame)

        return self._detect_local(frame)

    def close(self):
        if self.external_process is not None:
            try:
                self.external_process.stdin.write(
                    json.dumps({"command": "stop"}) + "\n"
                )
                self.external_process.stdin.flush()
            except Exception:
                pass

            try:
                self.external_process.terminate()
            except Exception:
                pass

            self.external_process = None
