"""Shared live state between the AI camera loop and FastAPI."""
from __future__ import annotations
import threading, time
from collections import deque
import statistics
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
        self.zone_stats = []
        self.staff_coverage = []
        self.staff_tracking_configured = False
        self.acknowledged = set()
        self.notification_status = {}
        self.recommendations = []
        self.zone_history = deque(maxlen=120)
        self.response_times = deque(maxlen=100)
        self.anomalies = []

    def publish(self, frame, camera_id, tracks, insights, alerts, fps=0.0, operational_alerts=None, zone_stats=None, staff_coverage=None, staff_tracking_configured=False):
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
                    "zone": getattr(ins, "zone", None) or "Unknown",
                    "dwell_seconds": round(float(getattr(ins, "dwell_seconds", 0)), 1),
                    "intent_score": int(getattr(ins, "intent_score", 0)),
                    "intent_level": str(getattr(ins, "intent_level", "LOW")),
                    "product_interaction": bool(getattr(ins, "product_interaction", False)),
                    "phone_comparison": bool(getattr(ins, "phone_comparison", False)),
                    "staff_nearby": getattr(ins, "staff_nearby", None),
                    "activity": str(getattr(t, "state", "unknown")),
                    "bbox": [int(x) for x in getattr(t, "bbox", (0, 0, 0, 0))],
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
                    "acknowledged": False,
                }
                self.alerts.appendleft(item)
                self.events.appendleft(dict(item))

            for item in (operational_alerts or []):
                item = dict(item)
                item.setdefault("id", f"{time.time_ns()}")
                item.setdefault("timestamp", time.time())
                item.setdefault("camera_id", camera_id)
                item.setdefault("severity", "high")
                item["acknowledged"] = item["id"] in self.acknowledged
                self.alerts.appendleft(item)
                self.events.appendleft(dict(item))

            if zone_stats is not None:
                self.zone_stats = list(zone_stats)
                sample = {
                    "timestamp": time.time(),
                    "zones": {str(z.get("zone")): int(z.get("customer_count", 0) or 0) for z in zone_stats},
                }
                self.zone_history.append(sample)
                anomalies = []
                history = list(self.zone_history)
                for z in zone_stats:
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
            if staff_coverage is not None:
                self.staff_coverage = list(staff_coverage)
            self.staff_tracking_configured = bool(staff_tracking_configured)

            self.metrics = {
                "customers": len(rows),
                "high_intent": sum(1 for x in rows.values() if x["intent_level"].upper() == "HIGH"),
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
                "zone_stats": list(self.zone_stats),
                "staff_coverage": list(self.staff_coverage),
                "staff_tracking_configured": self.staff_tracking_configured,
                "cameras": [{"camera_id": self.camera_id, "status": self.camera_status, "fps": self.fps}],
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
