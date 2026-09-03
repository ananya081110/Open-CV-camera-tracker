
"""
DeepCamera Object Recognition - high accuracy mode.

Design:
  1. YOLOE-26 prompt-free provides broad candidate detection
     (Ultralytics documents a built-in 4,585-name vocabulary).
  2. Qwen3-VL 8B through Ollama is the authoritative classifier.
  3. Qwen3-VL verifies YOLO crops asynchronously, so the camera
     loop stays responsive.
  4. A periodic whole-frame Qwen3-VL inventory catches obvious
     objects that YOLOE missed.
  5. Temporal tracking keeps IDs stable while labels are corrected.

The final displayed label is the VLM label whenever a VLM result
exists. YOLO is never allowed to overwrite a verified VLM label.

Recommended setup:
    ollama pull qwen3-vl:8b
"""

import base64
import json
import os
import queue
import threading
import time
import urllib.request
from collections import deque

import cv2


class TemporalObjectTracker:
    def __init__(self, min_hits=2, max_missed=12, iou_threshold=0.18):
        self.min_hits = min_hits
        self.max_missed = max_missed
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.tracks = {}

    @staticmethod
    def iou(a, b):
        ax1, ay1, ax2, ay2 = map(float, a)
        bx1, by1, bx2, by2 = map(float, b)
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        ab = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = aa + ab - inter
        return inter / union if union else 0.0

    def update(self, detections):
        detections = list(detections or [])
        pairs = []

        for tid, track in self.tracks.items():
            for i, det in enumerate(detections):
                score = self.iou(track["bbox"], det["bbox"])
                if score >= self.iou_threshold:
                    pairs.append((score, tid, i))

        pairs.sort(reverse=True)
        used_tracks = set()
        used_dets = set()

        for _, tid, i in pairs:
            if tid in used_tracks or i in used_dets:
                continue
            det = detections[i]
            track = self.tracks[tid]
            track["bbox"] = det["bbox"]
            track["last"] = det
            track["hits"] += 1
            track["missed"] = 0
            used_tracks.add(tid)
            used_dets.add(i)

        for i, det in enumerate(detections):
            if i in used_dets:
                continue
            self.tracks[self.next_id] = {
                "bbox": det["bbox"],
                "last": det,
                "hits": 1,
                "missed": 0,
            }
            self.next_id += 1

        for tid in list(self.tracks):
            if tid not in used_tracks:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]

        out = []
        for tid, track in self.tracks.items():
            if track["missed"] == 0 and track["hits"] >= self.min_hits:
                item = dict(track["last"])
                item["track_id"] = tid
                item["temporal_hits"] = track["hits"]
                out.append(item)
        return out


class Qwen3VLVerifier:
    """
    Native Ollama /api/chat client.

    Two modes:
      - crop classification: authoritative label for a detected box
      - whole-frame inventory: catches missed objects

    Only one request runs at a time to avoid saturating a MacBook Air.
    """

    def __init__(self):
        self.url = os.getenv(
            "DEEPCAMERA_VLM_URL",
            "http://localhost:11434/api/chat",
        ).strip()

        self.model = os.getenv(
            "DEEPCAMERA_VLM_MODEL",
            "qwen3-vl:8b",
        ).strip()

        self.crop_interval = float(
            os.getenv("DEEPCAMERA_VLM_CROP_INTERVAL", "0.35")
        )
        self.inventory_interval = float(
            os.getenv("DEEPCAMERA_VLM_INVENTORY_INTERVAL", "2.5")
        )
        self.timeout = float(
            os.getenv("DEEPCAMERA_VLM_TIMEOUT", "20")
        )

        self.enabled = False
        self.busy = False
        self.jobs = queue.Queue(maxsize=3)
        self.results = {}
        self.inventory = []
        self.last_inventory = 0.0
        self.last_crop = 0.0
        self.last_error = ""
        self.lock = threading.Lock()

        self._probe()
        if self.enabled:
            threading.Thread(
                target=self._worker,
                daemon=True,
            ).start()

    def _probe(self):
        try:
            url = self.url.replace("/api/chat", "/api/tags")
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as response:
                data = json.loads(response.read().decode("utf-8"))

            installed = {
                str(x.get("name", ""))
                for x in data.get("models", [])
            }

            if self.model in installed:
                self.enabled = True
                print(
                    f"[INFO] Qwen3-VL authoritative classification enabled: "
                    f"{self.model}"
                )
            else:
                print(
                    f"[WARNING] {self.model} is not installed in Ollama."
                )

        except Exception as exc:
            print(
                "[WARNING] Cannot connect to Ollama at "
                f"{self.url}: {exc}"
            )

    @staticmethod
    def _encode(image):
        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )
        if not ok:
            return None
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    @staticmethod
    def _parse_json(text):
        text = str(text or "")
        # Qwen may emit thinking text before the JSON.
        for opener, closer in (("[", "]"), ("{", "}")):
            start = text.find(opener)
            end = text.rfind(closer)
            if start >= 0 and end > start:
                try:
                    value = json.loads(text[start:end + 1])
                    if isinstance(value, dict):
                        return [value]
                    if isinstance(value, list):
                        return value
                except Exception:
                    continue
        return []

    def _call(self, image_b64, prompt):
        payload = {
            "model": self.model,
            "stream": False,
            "options": {
                "temperature": 0,
            },
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
        }

        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(
            request,
            timeout=self.timeout,
        ) as response:
            data = json.loads(response.read().decode("utf-8"))

        return data.get("message", {}).get("content", "")

    def _crop_job(self, frame, track_id, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        # Add a little context around the object.
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_x = int(bw * 0.12)
        pad_y = int(bh * 0.12)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        crop = frame[y1:y2, x1:x2]
        encoded = self._encode(crop)
        if encoded is None:
            return

        prompt = """
Identify the single physical object in this crop.

Return ONLY JSON:
{"label":"...", "confidence":0.00}

Rules:
- The label must be the actual object, not a visual guess from
  another category.
- Use a specific everyday object name.
- A spoon must be "spoon"; a toothbrush must have a brush head
  and bristles; do not confuse similar-looking objects.
- Ignore the person, skin, hand, hair, clothes and background.
- If the object is clearly a spoon, fork, knife, cup, phone,
  bottle, pen, glasses, remote, keyboard, mouse, laptop, book,
  chair, bag, etc., name it exactly.
- Do not return "object", "item", "thing", or "unknown" if a
  concrete object can be recognized.
"""

        raw = self._call(encoded, prompt)
        parsed = self._parse_json(raw)

        if not parsed:
            return

        item = parsed[0]
        label = str(item.get("label", "")).strip()
        try:
            confidence = float(item.get("confidence", 0))
        except Exception:
            confidence = 0.0

        if not label:
            return

        with self.lock:
            self.results[track_id] = {
                "label": label,
                "confidence": max(0.0, min(1.0, confidence)),
                "updated": time.time(),
            }
            self.last_error = ""

    def _inventory_job(self, frame):
        encoded = self._encode(frame)
        if encoded is None:
            self.busy = False
            return

        prompt = """
Find all clearly visible physical objects in this camera frame.

Return ONLY a JSON array:
[
  {
    "label":"specific object name",
    "confidence":0.00,
    "bbox":[x1,y1,x2,y2]
  }
]

Coordinates must be normalized 0.0-1.0 relative to the complete image.

Rules:
- Include every clearly visible object, including small handheld
  objects.
- Do NOT include people, body parts, hair, skin, clothing or
  background surfaces.
- Use specific everyday names.
- Do not invent objects.
- Do not confuse a spoon with a toothbrush, fork, knife, pen,
  or other object.
- If an object is visible but uncertain, give lower confidence.
"""

        raw = self._call(encoded, prompt)
        parsed = self._parse_json(raw)
        h, w = frame.shape[:2]
        found = []

        for item in parsed:
            try:
                label = str(item.get("label", "")).strip()
                conf = float(item.get("confidence", 0))
                box = item.get("bbox")

                if (
                    not label
                    or not isinstance(box, list)
                    or len(box) != 4
                    or conf < 0.45
                ):
                    continue

                if label.lower() in {
                    "person", "human", "face", "hand",
                    "arm", "leg", "body",
                }:
                    continue

                x1 = max(0, min(w - 1, int(float(box[0]) * w)))
                y1 = max(0, min(h - 1, int(float(box[1]) * h)))
                x2 = max(x1 + 1, min(w, int(float(box[2]) * w)))
                y2 = max(y1 + 1, min(h, int(float(box[3]) * h)))

                found.append({
                    "class_name": label,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "center": [
                        (x1 + x2) // 2,
                        (y1 + y2) // 2,
                    ],
                    "classification_source": "Qwen3-VL",
                    "raw_class_name": label,
                    "raw_confidence": conf,
                    "verified": True,
                    "inventory": True,
                })
            except Exception:
                continue

        with self.lock:
            self.inventory = found
            self.last_inventory = time.time()
            self.last_error = ""

    def _worker(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return

            self.busy = True
            try:
                kind, frame, ident, bbox = job

                if kind == "crop":
                    self._crop_job(
                        frame,
                        ident,
                        bbox,
                    )
                    self.last_crop = time.time()
                else:
                    self._inventory_job(frame)
            except Exception as exc:
                with self.lock:
                    self.last_error = str(exc)
            finally:
                self.busy = False
                self.jobs.task_done()

    def submit(self, frame, tracks):
        if not self.enabled:
            return

        now = time.time()

        # Prioritize crop verification for newly detected/low-confidence
        # tracked objects. Only queue one at a time.
        with self.lock:
            verified_ids = set(self.results)

        candidates = [
            t for t in tracks
            if t.get("track_id") not in verified_ids
        ]

        if (
            not self.busy
            and candidates
            and now - self.last_crop >= self.crop_interval
        ):
            candidate = candidates[0]
            try:
                self.jobs.put_nowait((
                    "crop",
                    frame.copy(),
                    candidate["track_id"],
                    candidate["bbox"],
                ))
            except queue.Full:
                pass

        # Periodic whole-frame inventory catches objects YOLOE missed.
        if (
            not self.busy
            and now - self.last_inventory >= self.inventory_interval
        ):
            try:
                self.jobs.put_nowait((
                    "inventory",
                    frame.copy(),
                    None,
                    None,
                ))
            except queue.Full:
                pass

    def apply(self, tracks):
        output = []

        with self.lock:
            verified = dict(self.results)
            inventory = list(self.inventory)

        for item in tracks:
            item = dict(item)
            tid = item.get("track_id")
            result = verified.get(tid)

            if result:
                item["raw_class_name"] = item.get(
                    "class_name",
                    "",
                )
                item["raw_confidence"] = item.get(
                    "confidence",
                    0.0,
                )
                item["class_name"] = result["label"]
                item["confidence"] = result["confidence"]
                item["classification_source"] = (
                    "YOLOE+Qwen3-VL"
                )
                item["verified"] = True

            output.append(item)

        # Add VLM-only objects that don't overlap YOLO boxes.
        for item in inventory:
            duplicate = False
            for existing in output:
                if TemporalObjectTracker.iou(
                    existing["bbox"],
                    item["bbox"],
                ) >= 0.25:
                    duplicate = True
                    break

            if not duplicate:
                output.append(item)

        return output

    def status(self):
        with self.lock:
            if not self.enabled:
                return "Qwen3-VL: OFF"
            if self.busy:
                return "Qwen3-VL: VERIFYING"
            if self.last_error:
                return "Qwen3-VL: ERROR"
            return "Qwen3-VL: ACTIVE"


class DeepCameraDetector:
    def __init__(
        self,
        model_path="yoloe-26s-seg-pf.pt",
        confidence=0.12,
        iou=0.45,
        classes=None,
        fallback=None,
        imgsz=960,
        max_det=300,
        vlm_url=None,
        vlm_model=None,
    ):
        self.model_path = model_path
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.classes = classes
        self.fallback = fallback
        self.imgsz = int(imgsz)
        self.max_det = int(max_det)

        if vlm_url:
            os.environ["DEEPCAMERA_VLM_URL"] = str(vlm_url)
        if vlm_model:
            os.environ["DEEPCAMERA_VLM_MODEL"] = str(vlm_model)

        self.model = None
        self.model_type = "none"
        self.tracker = TemporalObjectTracker()
        self.vlm = Qwen3VLVerifier()

        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLOE

            self.model = YOLOE(self.model_path)
            self.model_type = "yoloe"

            print(
                "[INFO] YOLOE-26 prompt-free ready "
                f"(built-in broad vocabulary): {self.model_path}"
            )
            return
        except Exception as exc:
            print(f"[WARNING] YOLOE unavailable: {exc}")

        try:
            from ultralytics import YOLO
            fallback_model = os.getenv(
                "DEEPCAMERA_FALLBACK_MODEL",
                "yolo26s.pt",
            )
            self.model = YOLO(fallback_model)
            self.model_type = "yolo26"
            print(
                f"[INFO] YOLO26 fallback ready: {fallback_model}"
            )
            return
        except Exception as exc:
            print(f"[WARNING] YOLO26 unavailable: {exc}")

        if self.fallback is not None:
            print("[WARNING] Using existing ObjectDetector fallback.")

    @staticmethod
    def _normalize(box, name, conf, source):
        x1, y1, x2, y2 = map(float, box)
        return {
            "class_name": str(name).strip(),
            "confidence": float(conf),
            "bbox": [
                int(x1), int(y1), int(x2), int(y2)
            ],
            "center": [
                int((x1 + x2) / 2),
                int((y1 + y2) / 2),
            ],
            "classification_source": source,
            "raw_class_name": str(name).strip(),
            "raw_confidence": float(conf),
            "verified": False,
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
                imgsz=self.imgsz,
                max_det=self.max_det,
                verbose=False,
            )
        except Exception as exc:
            print(f"[WARNING] YOLO inference failed: {exc}")
            if self.fallback is not None:
                return self.fallback.detect(frame)
            return []

        if not results or results[0].boxes is None:
            return []

        result = results[0]
        names = result.names
        detections = []

        for box in result.boxes:
            try:
                cid = int(box.cls[0])
                conf = float(box.conf[0])
                name = names[cid]

                detections.append(
                    self._normalize(
                        box.xyxy[0].tolist(),
                        name,
                        conf,
                        "YOLOE"
                        if self.model_type == "yoloe"
                        else "YOLO26",
                    )
                )
            except Exception:
                continue

        return detections

    def detect(self, frame, timestamp_ms=None):
        raw = self._detect_local(frame)

        # Track first, then send stable boxes to the authoritative VLM.
        tracks = self.tracker.update(raw)

        self.vlm.submit(
            frame,
            tracks,
        )

        final = self.vlm.apply(
            tracks
        )

        return final

    def status(self):
        return self.vlm.status()

    def close(self):
        pass
