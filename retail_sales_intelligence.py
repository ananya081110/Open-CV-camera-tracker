
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RetailVisit:
    zone: str
    entered_at: float
    last_seen: float
    total_dwell: float = 0.0
    visit_count: int = 1
    alerted: bool = False
    product_interaction: bool = False
    phone_comparison: bool = False


@dataclass
class CustomerInsight:
    person_id: int
    zone: str
    dwell_seconds: float
    repeat_visit: bool
    product_interaction: bool
    phone_comparison: bool
    staff_nearby: bool
    intent_score: int
    intent_level: str
    reasons: List[str] = field(default_factory=list)
    alert: Optional[str] = None


class RetailSalesIntelligence:
    """
    First retail POC:
    tracked customer -> retail zone -> dwell/repeat behaviour
    -> intent score -> actionable alert.

    Scoring:
      dwell >= threshold +2
      product interaction +3
      phone comparison +2
      repeat visit +2
      no staff nearby +3

    Only signals explicitly supplied to update() are treated as observed.
    """

    DEFAULT_ZONES = {
        "TV": (0.45, 0.20, 0.90, 0.90),
    }

    def __init__(
        self,
        zones: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
        dwell_threshold: float = 60.0,
        high_intent_threshold: int = 7,
        alert_cooldown: float = 120.0,
        stale_after: float = 15.0,
    ):
        self.dwell_threshold = max(
            1.0,
            float(os.getenv("RETAIL_DWELL_THRESHOLD", dwell_threshold)),
        )
        self.high_intent_threshold = max(
            1,
            int(os.getenv(
                "RETAIL_HIGH_INTENT_THRESHOLD",
                high_intent_threshold,
            )),
        )
        self.alert_cooldown = max(
            0.0,
            float(os.getenv("RETAIL_ALERT_COOLDOWN", alert_cooldown)),
        )
        self.stale_after = max(
            1.0,
            float(os.getenv("RETAIL_STALE_AFTER", stale_after)),
        )

        self.zones = self._load_zones(zones)
        self.visits: Dict[Tuple[int, str], RetailVisit] = {}
        self.completed_visits: Dict[Tuple[int, str], int] = {}
        self.last_alert_at: Dict[Tuple[int, str], float] = {}
        self.latest: Dict[int, CustomerInsight] = {}
        self.latest_alerts: List[CustomerInsight] = []

    def _load_zones(self, zones):
        raw = os.getenv("RETAIL_ZONES", "").strip()

        if raw:
            try:
                data = json.loads(raw)
                loaded = {
                    str(name): tuple(float(v) for v in box)
                    for name, box in data.items()
                    if isinstance(box, (list, tuple)) and len(box) == 4
                }
                if loaded:
                    return loaded
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                print(f"[WARNING] Invalid RETAIL_ZONES: {exc}")

        if zones:
            return {
                str(name): tuple(float(v) for v in box)
                for name, box in zones.items()
                if isinstance(box, (list, tuple)) and len(box) == 4
            }

        return dict(self.DEFAULT_ZONES)

    @staticmethod
    def _inside(center, zone, width, height):
        try:
            x, y = float(center[0]), float(center[1])
            x1, y1, x2, y2 = map(float, zone)
        except (TypeError, ValueError, IndexError):
            return False

        return (
            x1 * width <= x <= x2 * width
            and y1 * height <= y <= y2 * height
        )

    def zone_for(self, center, width, height):
        for name, box in self.zones.items():
            if self._inside(center, box, width, height):
                return name
        return None

    def _score(
        self,
        visit,
        product_interaction,
        phone_comparison,
        staff_nearby,
    ):
        score = 0
        reasons = []

        if visit.total_dwell >= self.dwell_threshold:
            score += 2
            reasons.append(
                f"dwell > {int(self.dwell_threshold)}s (+2)"
            )

        if product_interaction:
            score += 3
            reasons.append("product interaction (+3)")

        if phone_comparison:
            score += 2
            reasons.append("phone comparison (+2)")

        if visit.visit_count > 1:
            score += 2
            reasons.append("repeat visit (+2)")

        if staff_nearby is False:
            score += 3
            reasons.append("no staff nearby (+3)")

        return score, reasons

    @staticmethod
    def _level(score):
        if score >= 7:
            return "HIGH"
        if score >= 4:
            return "MEDIUM"
        return "LOW"

    def update(
        self,
        tracks,
        width,
        height,
        now=None,
        interactions=None,
        phone_comparison=None,
        staff_nearby=None,
    ):
        now = time.monotonic() if now is None else float(now)
        interactions = {} if interactions is None else interactions
        phone_comparison = (
            {} if phone_comparison is None else phone_comparison
        )
        staff_nearby = {} if staff_nearby is None else staff_nearby

        active_ids = set()
        insights = []
        alerts = []

        for track in tracks or []:
            if getattr(track, "missed", 0):
                continue

            try:
                person_id = int(track.id)
            except (TypeError, ValueError, AttributeError):
                continue

            center = getattr(track, "center", None)
            if center is None:
                continue

            active_ids.add(person_id)
            zone = self.zone_for(center, width, height)

            if zone is None:
                self.latest.pop(person_id, None)
                continue

            key = (person_id, zone)
            visit = self.visits.get(key)

            if visit is None:
                visit = RetailVisit(
                    zone=zone,
                    entered_at=now,
                    last_seen=now,
                    visit_count=self.completed_visits.get(key, 0) + 1,
                )
                self.visits[key] = visit
                print(
                    f"[RETAIL] Customer {person_id} entered "
                    f"{zone} (visit {visit.visit_count})"
                )
            else:
                elapsed = max(0.0, now - visit.last_seen)
                visit.total_dwell += min(elapsed, 2.0)
                visit.last_seen = now

            product = bool(interactions.get(person_id, False))
            phone = bool(phone_comparison.get(person_id, False))
            staff_raw = staff_nearby.get(person_id, None)
            staff = None if staff_raw is None else bool(staff_raw)

            visit.product_interaction |= product
            visit.phone_comparison |= phone

            score, reasons = self._score(
                visit,
                product_interaction=visit.product_interaction,
                phone_comparison=visit.phone_comparison,
                staff_nearby=staff,
            )

            level = self._level(score)
            alert = None

            if (
                level == "HIGH"
                and visit.total_dwell >= self.dwell_threshold
                and not visit.alerted
                and now - self.last_alert_at.get(key, -1e18)
                >= self.alert_cooldown
            ):
                alert = self._build_alert(
                    person_id,
                    zone,
                    visit.total_dwell,
                    score,
                    reasons,
                )
                visit.alerted = True
                self.last_alert_at[key] = now

            insight = CustomerInsight(
                person_id=person_id,
                zone=zone,
                dwell_seconds=visit.total_dwell,
                repeat_visit=visit.visit_count > 1,
                product_interaction=visit.product_interaction,
                phone_comparison=visit.phone_comparison,
                staff_nearby=staff,
                intent_score=score,
                intent_level=level,
                reasons=reasons,
                alert=alert,
            )

            self.latest[person_id] = insight
            insights.append(insight)

            if alert:
                alerts.append(insight)
                print(
                    f"[RETAIL ALERT] Customer #{person_id} | "
                    f"{zone} | HIGH | score={score}"
                )

        for key, visit in list(self.visits.items()):
            person_id, _ = key
            if person_id not in active_ids and now - visit.last_seen >= self.stale_after:
                self.completed_visits[key] = visit.visit_count
                del self.visits[key]

        self.latest_alerts = alerts
        return insights, alerts

    @staticmethod
    def _build_alert(person_id, zone, dwell_seconds, score, reasons):
        minutes = int(dwell_seconds // 60)
        seconds = int(dwell_seconds % 60)
        dwell = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

        return (
            "HIGH-INTENT CUSTOMER DETECTED\n"
            f"Customer: #{person_id}\n"
            f"Section: {zone}\n"
            f"Dwell time: {dwell}\n"
            f"Intent score: {score}\n"
            f"Signals: {', '.join(reasons)}\n"
            "Suggested action: Send sales associate."
        )

    def zone_overlay(self, frame):
        import cv2

        h, w = frame.shape[:2]
        for name, box in self.zones.items():
            x1, y1, x2, y2 = box
            p1 = (int(x1 * w), int(y1 * h))
            p2 = (int(x2 * w), int(y2 * h))

            cv2.rectangle(frame, p1, p2, (255, 180, 0), 2)
            cv2.putText(
                frame,
                f"RETAIL: {name}",
                (p1[0], max(25, p1[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 180, 0),
                2,
                cv2.LINE_AA,
            )

    def status(self):
           return (
               f"Retail Intelligence: ACTIVE | "
               f"Zones: {len(self.zones)} | "
               f"Dwell: {self.dwell_threshold:.0f}s | "
               f"High: {self.high_intent_threshold}"
           )