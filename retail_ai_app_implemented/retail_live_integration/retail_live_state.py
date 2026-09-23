"""Shared live state and multi-camera fleet aggregation for the Retail AI platform."""
from __future__ import annotations

import os
import statistics
import threading
import time
from collections import deque

import cv2


def _camera_ids():
    raw = os.getenv("RETAIL_CAMERA_IDS", "").strip()
    if raw:
        return [x.strip() for x in raw.split(",") if x.strip()]
    try:
        count = max(1, int(os.getenv("RETAIL_CAMERA_COUNT", "32")))
    except ValueError:
        count = 32
    return [f"CAM_{i:02d}" for i in range(1, count + 1)]


def _camera_zone_map():
    # Optional format: CAM_01:Entrance,CAM_02:Entrance,CAM_03:Mobile
    result = {}
    raw = os.getenv("RETAIL_CAMERA_ZONE_MAP", "").strip()
    for item in raw.split(",") if raw else []:
        if ":" in item:
            camera_id, zone = item.split(":", 1)
            result[camera_id.strip()] = zone.strip()
    return result


class RetailLiveState:
    """Thread-safe state shared by camera workers and FastAPI.

    The current POC still has one active AI worker, but state is now keyed by
    camera_id. Any number of camera workers can publish independently and the
    dashboard receives a store-wide aggregate plus per-camera telemetry.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.last_update = 0.0
        self.camera_id = "CAM_01"
        self.camera_status = "starting"
        self.fps = 0.0
        self.frame_jpeg = None
        self.customers = {}
        self.alerts = deque(maxlen=200)
        self.events = deque(maxlen=500)
        self.metrics = {"customers": 0, "high_intent": 0, "alerts": 0}
        self.zone_stats = []
        self.staff_coverage = []
        self.staff_tracking_configured = False
        self.acknowledged = set()
        self.notification_status = {}
        self.recommendations = []
        self.zone_history = deque(maxlen=120)
        self.response_times = deque(maxlen=100)
        self.anomalies = []

        self.camera_timeout_seconds = float(os.getenv("RETAIL_CAMERA_TIMEOUT_SECONDS", "8"))
        self.camera_zone_map = _camera_zone_map()
        self.camera_states = {}
        for cid in _camera_ids():
            self._ensure_camera(cid)

    def _ensure_camera(self, camera_id):
        camera_id = str(camera_id)
        if camera_id not in self.camera_states:
            self.camera_states[camera_id] = {
                "camera_id": camera_id,
                "status": "offline",
                "fps": 0.0,
                "last_update": 0.0,
                "latency_ms": None,
                "customer_count": 0,
                "high_intent": 0,
                "alert_count": 0,
                "zone": self.camera_zone_map.get(camera_id, "Unassigned"),
                "source": "unconfigured",
                "frame_jpeg": None,
                "customers": {},
                "zone_stats": [],
                "staff_coverage": [],
            }
        return self.camera_states[camera_id]

    def register_camera(self, camera_id, zone=None, source=None):
        with self.lock:
            item = self._ensure_camera(camera_id)
            if zone:
                item["zone"] = zone
            if source:
                item["source"] = source
            return dict(item)

    def publish(self, frame, camera_id, tracks, insights, alerts, fps=0.0,
                operational_alerts=None, zone_stats=None, staff_coverage=None,
                staff_tracking_configured=False, customer_journeys=None):
        camera_id = str(camera_id)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        now = time.time()

        with self.lock:
            camera = self._ensure_camera(camera_id)
            if ok:
                camera["frame_jpeg"] = encoded.tobytes()
                self.frame_jpeg = camera["frame_jpeg"]
            camera["status"] = "online"
            camera["fps"] = round(float(fps), 1)
            camera["last_update"] = now
            camera["source"] = "ai_worker"

            active = {int(t.id): t for t in tracks if not getattr(t, "missed", False)}
            journey_map = {
                int(j.get("customer_id")): j
                for j in (customer_journeys or [])
                if j.get("customer_id") is not None
            }
            rows = {}
            for ins in insights:
                t = active.get(int(ins.person_id))
                if not t:
                    continue
                key = f"{camera_id}:{int(ins.person_id)}"
                journey = journey_map.get(int(ins.person_id), {})
                rows[key] = {
                    "customer_id": f"{camera_id}-CUST-{int(ins.person_id):04d}",
                    "legacy_customer_id": f"CUST-{int(ins.person_id):04d}",
                    "person_id": int(ins.person_id),
                    "camera_id": camera_id,
                    "zone": getattr(ins, "zone", None) or journey.get("zone") or "Unknown",
                    "zone_history": list(journey.get("zone_history", [])),
                    "zone_dwell": dict(journey.get("zone_dwell", {})),
                    "total_store_dwell": round(float(journey.get("total_store_dwell", getattr(ins, "dwell_seconds", 0))), 1),
                    "dwell_seconds": round(float(getattr(ins, "dwell_seconds", journey.get("total_store_dwell", 0))), 1),
                    "intent_score": int(getattr(ins, "intent_score", journey.get("intent_score", 0))),
                    "intent_level": str(getattr(ins, "intent_level", journey.get("intent_level", "LOW"))),
                    "product_interaction": bool(getattr(ins, "product_interaction", journey.get("product_interaction", False))),
                    "phone_comparison": bool(getattr(ins, "phone_comparison", journey.get("phone_comparison", False))),
                    "repeat_visit": bool(journey.get("repeat_visit", False)),
                    "staff_nearby": getattr(ins, "staff_nearby", journey.get("staff_nearby")),
                    "activity": str(getattr(t, "state", "unknown")),
                    "bbox": [int(x) for x in getattr(t, "bbox", (0, 0, 0, 0))],
                }
            camera["customers"] = rows
            camera["customer_count"] = len(rows)
            camera["high_intent"] = sum(1 for x in rows.values() if x["intent_level"].upper() == "HIGH")
            if zone_stats is not None:
                camera["zone_stats"] = list(zone_stats)
            if staff_coverage is not None:
                camera["staff_coverage"] = list(staff_coverage)
            camera["alert_count"] = len(alerts or []) + len(operational_alerts or [])

            # Preserve the existing event model, but tag every event with its source camera.
            for alert in alerts:
                item = {
                    "id": f"{time.time_ns()}",
                    "timestamp": now,
                    "camera_id": camera_id,
                    "customer_id": f"{camera_id}-CUST-{int(getattr(alert, 'person_id', 0)):04d}",
                    "type": "HIGH_INTENT_CUSTOMER",
                    "severity": "high",
                    "message": getattr(alert, "alert", None) or "High-intent customer detected.",
                    "acknowledged": False,
                }
                self.alerts.appendleft(item)
                self.events.appendleft(dict(item))

            for item in (operational_alerts or []):
                item = dict(item)
                item.setdefault("id", f"{time.time_ns()}")
                item.setdefault("timestamp", now)
                item.setdefault("camera_id", camera_id)
                item.setdefault("severity", "high")
                item["acknowledged"] = item["id"] in self.acknowledged
                self.alerts.appendleft(item)
                self.events.appendleft(dict(item))

            # Store-wide customer state is the union of all currently live camera workers.
            self._refresh_camera_statuses(now)
            self._aggregate_store_state()

            if zone_stats is not None:
                # Keep historical zone observations at the store level.
                sample = {
                    "timestamp": now,
                    "zones": {str(z.get("zone")): int(z.get("customer_count", 0) or 0) for z in self.zone_stats},
                }
                self.zone_history.append(sample)
                anomalies = []
                history = list(self.zone_history)
                for z in self.zone_stats:
                    name = str(z.get("zone"))
                    current = int(z.get("customer_count", 0) or 0)
                    prior = [int(h.get("zones", {}).get(name, 0)) for h in history[:-1]][-30:]
                    if len(prior) >= 6:
                        mean = statistics.mean(prior)
                        stdev = statistics.pstdev(prior) or 1.0
                        zscore = (current - mean) / stdev
                        if abs(zscore) >= 2.0 and current > 0:
                            anomalies.append({
                                "zone": name,
                                "type": "TRAFFIC_SPIKE" if zscore > 0 else "TRAFFIC_DROP",
                                "severity": "high" if abs(zscore) >= 3 else "medium",
                                "current": current,
                                "baseline": round(mean, 1),
                                "z_score": round(zscore, 2),
                                "message": f"{name} has unusual traffic: {current} customer(s) vs baseline {mean:.1f}.",
                            })
                self.anomalies = anomalies[:10]

            self.staff_tracking_configured = bool(staff_tracking_configured)
            self.last_update = now
            self.camera_id = camera_id
            self.camera_status = "online"
            self.fps = round(float(fps), 1)

    def _refresh_camera_statuses(self, now=None):
        now = now or time.time()
        for camera in self.camera_states.values():
            if camera["last_update"] <= 0:
                camera["status"] = "offline"
            elif now - camera["last_update"] > self.camera_timeout_seconds:
                camera["status"] = "offline"

    def _aggregate_store_state(self):
        live = [c for c in self.camera_states.values() if c["status"] == "online"]
        customers = {}
        for camera in live:
            customers.update(camera["customers"])

        zone_map = {}
        staff_map = {}
        for camera in live:
            for z in camera.get("zone_stats", []):
                name = str(z.get("zone", "Unknown"))
                item = zone_map.setdefault(name, {"zone": name, "customer_count": 0, "staff_present": False})
                item["customer_count"] += int(z.get("customer_count", 0) or 0)
                item["staff_present"] = item["staff_present"] or bool(z.get("staff_present", False))
            for z in camera.get("staff_coverage", []):
                name = str(z.get("zone", "Unknown"))
                item = staff_map.setdefault(name, {"zone": name, "customer_count": 0, "staff_present": False})
                item["customer_count"] += int(z.get("customer_count", 0) or 0)
                item["staff_present"] = item["staff_present"] or bool(z.get("staff_present", False))

        self.customers = customers
        self.zone_stats = list(zone_map.values())
        self.staff_coverage = list(staff_map.values())
        self.metrics = {
            "customers": len(customers),
            "high_intent": sum(1 for x in customers.values() if x["intent_level"].upper() == "HIGH"),
            "alerts": sum(1 for x in self.alerts if not x.get("acknowledged", False)),
        }

    def set_recommendations(self, recommendations):
        with self.lock:
            self.recommendations = list(recommendations or [])[:20]

    def set_notification_status(self, status):
        with self.lock:
            self.notification_status = dict(status or {})

    def acknowledge(self, alert_id):
        with self.lock:
            self.acknowledged.add(str(alert_id))
            response_seconds = None
            now = time.time()
            for item in self.alerts:
                if str(item.get("id")) == str(alert_id):
                    item["acknowledged"] = True
                    item["acknowledged_at"] = now
                    response_seconds = max(0.0, now - float(item.get("timestamp", now)))
                    item["response_seconds"] = round(response_seconds, 1)
                    break
            if response_seconds is not None:
                self.response_times.append(response_seconds)
            self.metrics["alerts"] = sum(1 for x in self.alerts if not x.get("acknowledged", False))
            return any(str(x.get("id")) == str(alert_id) for x in self.alerts)

    def camera_snapshot(self, camera_id):
        with self.lock:
            self._refresh_camera_statuses()
            camera = self.camera_states.get(str(camera_id))
            if not camera:
                return None
            result = dict(camera)
            result.pop("frame_jpeg", None)
            return result

    def camera_frame(self, camera_id):
        with self.lock:
            camera = self.camera_states.get(str(camera_id))
            return camera.get("frame_jpeg") if camera else None

    def snapshot(self):
        with self.lock:
            self._refresh_camera_statuses()
            self._aggregate_store_state()
            cameras = []
            for camera in self.camera_states.values():
                cameras.append({
                    "camera_id": camera["camera_id"],
                    "status": camera["status"],
                    "fps": camera["fps"],
                    "last_update": camera["last_update"],
                    "latency_ms": camera["latency_ms"],
                    "customer_count": camera["customer_count"],
                    "high_intent": camera["high_intent"],
                    "alert_count": camera["alert_count"],
                    "zone": camera["zone"],
                    "source": camera["source"],
                })
            online = [c for c in cameras if c["status"] == "online"]
            configured = len(cameras)
            total_customers = sum(c["customer_count"] for c in online)
            return {
                "camera_id": self.camera_id,
                "camera_status": self.camera_status,
                "fps": self.fps,
                "last_update": self.last_update,
                "customers": list(self.customers.values()),
                "alerts": list(self.alerts)[:50],
                "events": list(self.events)[:100],
                "metrics": dict(self.metrics),
                "zone_stats": list(self.zone_stats),
                "staff_coverage": list(self.staff_coverage),
                "staff_tracking_configured": self.staff_tracking_configured,
                "cameras": cameras,
                "fleet_metrics": {
                    "configured_cameras": configured,
                    "online_cameras": len(online),
                    "offline_cameras": configured - len(online),
                    "online_ratio": round((len(online) / configured) * 100, 1) if configured else 0.0,
                    "active_customers": total_customers,
                    "camera_fps_avg": round(statistics.mean([c["fps"] for c in online]), 1) if online else 0.0,
                },
                "notification_status": dict(self.notification_status),
                "recommendations": list(self.recommendations),
                "anomalies": list(self.anomalies),
                "zone_history": list(self.zone_history),
                "operational_metrics": {
                    "avg_response_seconds": round(statistics.mean(self.response_times), 1) if self.response_times else 0.0,
                    "resolved_alerts": len(self.response_times),
                    "peak_zone": max(self.zone_stats, key=lambda z: int(z.get("customer_count", 0) or 0), default={}).get("zone") if self.zone_stats else None,
                    "peak_zone_customers": max([int(z.get("customer_count", 0) or 0) for z in self.zone_stats], default=0),
                    "anomaly_count": len(self.anomalies),
                },
            }


LIVE_STATE = RetailLiveState()
