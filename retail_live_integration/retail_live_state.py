"""Shared live state between the AI camera loop and FastAPI."""
from __future__ import annotations
import threading, time, json
from collections import deque
import cv2

class RetailLiveState:
    def __init__(self):
        self.lock = threading.RLock()
        self.frame_jpeg = None
        self.last_update = 0.0
        self.camera_id = "CAM_01"
        self.camera_status = "starting"
        self.fps = 0.0
        self.customers = {}
        self.alerts = deque(maxlen=200)
        self.events = deque(maxlen=500)
        self.metrics = {"customers": 0, "high_intent": 0, "alerts": 0}

    def publish(self, frame, camera_id, tracks, insights, alerts, fps=0.0):
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        with self.lock:
            if ok:
                self.frame_jpeg = encoded.tobytes()
            self.last_update = time.time()
            self.camera_id = camera_id
            self.camera_status = "online"
            self.fps = round(float(fps), 1)
            active = {int(t.id): t for t in tracks if not getattr(t, "missed", False)}
            rows = {}
            for ins in insights:
                t = active.get(int(ins.person_id))
                if not t:
                    continue
                rows[str(ins.person_id)] = {
                    "customer_id": f"CUST-{int(ins.person_id):04d}",
                    "person_id": int(ins.person_id),
                    "camera_id": camera_id,
                    "zone": getattr(ins, "zone", None) or "TV",
                    "dwell_seconds": round(float(getattr(ins, "dwell_seconds", 0)), 1),
                    "intent_score": int(getattr(ins, "intent_score", 0)),
                    "intent_level": str(getattr(ins, "intent_level", "LOW")),
                    "product_interaction": bool(getattr(ins, "product_interaction", False)),
                    "phone_comparison": bool(getattr(ins, "phone_comparison", False)),
                    "staff_nearby": getattr(ins, "staff_nearby", None),
                    "activity": str(getattr(t, "state", "unknown")),
                    "bbox": [int(x) for x in getattr(t, "bbox", (0,0,0,0))],
                }
            self.customers = rows
            for alert in alerts:
                item = {
                    "id": f"{time.time_ns()}",
                    "timestamp": time.time(),
                    "camera_id": camera_id,
                    "customer_id": f"CUST-{int(getattr(alert, 'person_id', 0)):04d}",
                    "type": "HIGH_INTENT_CUSTOMER",
                    "severity": "high",
                    "message": getattr(alert, "alert", None) or "High-intent customer detected.",
                }
                self.alerts.appendleft(item)
                self.events.appendleft(item)
            self.metrics = {
                "customers": len(rows),
                "high_intent": sum(1 for x in rows.values() if x["intent_level"].upper() == "HIGH"),
                "alerts": len(self.alerts),
            }

    def snapshot(self):
        with self.lock:
            return {
                "camera_id": self.camera_id,
                "camera_status": self.camera_status,
                "fps": self.fps,
                "last_update": self.last_update,
                "customers": list(self.customers.values()),
                "alerts": list(self.alerts)[:50],
                "events": list(self.events)[:100],
                "metrics": dict(self.metrics),
            }

LIVE_STATE = RetailLiveState()
