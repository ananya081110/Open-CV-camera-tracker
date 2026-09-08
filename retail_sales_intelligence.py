"""
Retail Sales Intelligence POC.

First end-to-end retail insight:
    tracked customer -> retail zone -> dwell time -> intent score
    -> actionable sales insight

This module intentionally starts small. It does not try to infer
unobservable customer traits from appearance. It scores observable
shopping behaviour only.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class RetailVisit:
    zone: str
    entered_at: float
    last_seen: float
    total_dwell: float = 0.0
    visit_count: int = 1
    alerted: bool = False


@dataclass
class CustomerInsight:
    person_id: int
    zone: str
    dwell_seconds: float
    repeat_visit: bool
    intent_score: int
    intent_level: str
    reasons: List[str] = field(default_factory=list)
    alert: Optional[str] = None


class RetailSalesIntelligence:
    """
    Maintains per-customer/per-zone state and converts behaviour
    into a simple POC intent score.

    Current scoring:
      dwell >= 60s       +2
      repeat visit       +2

    Reserved signals for later:
      product interaction +3
      phone comparison   +2
      no staff nearby    +3
    """

    DEFAULT_ZONES = {
        # x1, y1, x2, y2 in normalized frame coordinates.
        # Change these for the sample CCTV camera.
        "TV": (0.45, 0.20, 0.90, 0.90),
    }

    def __init__(
        self,
        zones: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
        dwell_threshold: float = 60.0,
        high_intent_threshold: int = 4,
        alert_cooldown: float = 120.0,
        stale_after: float = 15.0,
    ):
        self.dwell_threshold = float(
            os.getenv("RETAIL_DWELL_THRESHOLD", dwell_threshold)
        )
        self.high_intent_threshold = int(
            os.getenv("RETAIL_HIGH_INTENT_THRESHOLD", high_intent_threshold)
        )
        self.alert_cooldown = float(
            os.getenv("RETAIL_ALERT_COOLDOWN", alert_cooldown)
        )
        self.stale_after = float(
            os.getenv("RETAIL_STALE_AFTER", stale_after)
        )

        self.zones = self._load_zones(zones)
        self.visits: Dict[Tuple[int, str], RetailVisit] = {}
        self.completed_visits: Dict[Tuple[int, str], int] = {}
        self.last_alert_at: Dict[Tuple[int, str], float] = {}

        self.latest: Dict[int, CustomerInsight] = {}
        self.latest_alerts: List[CustomerInsight] = []

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _load_zones(self, zones):
        env = os.getenv("RETAIL_ZONES", "").strip()

        if env:
            try:
                raw = json.loads(env)
                loaded = {}

                for name, box in raw.items():
                    if isinstance(box, (list, tuple)) and len(box) == 4:
                        loaded[str(name)] = tuple(float(v) for v in box)

                if loaded:
                    return loaded
            except Exception as exc:
                print(f"[WARNING] Invalid RETAIL_ZONES: {exc}")

        if zones:
            return {
                str(name): tuple(float(v) for v in box)
                for name, box in zones.items()
            }

        return dict(self.DEFAULT_ZONES)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _inside(center, zone, width, height):
        x, y = float(center[0]), float(center[1])
        x1, y1, x2, y2 = zone

        return (
            x1 * width <= x <= x2 * width
            and y1 * height <= y <= y2 * height
        )

    def zone_for(self, center, width, height):
        # If zones overlap, the first configured zone wins.
        for name, box in self.zones.items():
            if self._inside(center, box, width, height):
                return name
        return None

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(self, visit, key):
        score = 0
        reasons = []

        if visit.total_dwell >= self.dwell_threshold:
            score += 2
            reasons.append(
                f"dwell > {int(self.dwell_threshold)}s (+2)"
            )

        if visit.visit_count > 1:
            score += 2
            reasons.append("repeat visit (+2)")

        # These are deliberately not awarded yet. They are the next
        # implementation steps once the base dwell flow is validated.
        return score, reasons

    @staticmethod
    def _level(score):
        if score >= 4:
            return "HIGH"
        if score >= 2:
            return "MEDIUM"
        return "LOW"

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, tracks, width, height, now=None):
        now = time.monotonic() if now is None else float(now)

        active_ids = set()
        insights = []
        alerts = []

        for track in tracks or []:
            if getattr(track, "missed", False):
                continue

            person_id = int(track.id)
            active_ids.add(person_id)

            center = getattr(track, "center", None)
            if not center:
                continue

            zone = self.zone_for(center, width, height)

            # Not in a configured retail zone.
            if zone is None:
                self.latest.pop(person_id, None)
                continue

            key = (person_id, zone)
            visit = self.visits.get(key)

            if visit is None:
                previous_visits = self.completed_visits.get(key, 0)

                visit = RetailVisit(
                    zone=zone,
                    entered_at=now,
                    last_seen=now,
                    visit_count=previous_visits + 1,
                )
                self.visits[key] = visit
                print(
                    f"[RETAIL] Customer {person_id} entered {zone} zone "
                    f"(visit {visit.visit_count})"
                )

            else:
                elapsed = max(0.0, now - visit.last_seen)
                # Cap a single update so camera stalls do not create
                # unrealistic dwell times.
                visit.total_dwell += min(elapsed, 2.0)
                visit.last_seen = now

            # First observation should have zero dwell, but a customer
            # who remains in the zone accumulates time on later frames.
            if visit.last_seen == now and visit.total_dwell == 0.0:
                dwell = 0.0
            else:
                dwell = visit.total_dwell

            score, reasons = self._score(visit, key)
            level = self._level(score)

            alert = None

            if (
                score >= self.high_intent_threshold
                and visit.total_dwell >= self.dwell_threshold
                and not visit.alerted
            ):
                last_alert = self.last_alert_at.get(key, 0.0)

                if now - last_alert >= self.alert_cooldown:
                    alert = self._build_alert(
                        person_id=person_id,
                        zone=zone,
                        dwell_seconds=visit.total_dwell,
                        score=score,
                    )
                    visit.alerted = True
                    self.last_alert_at[key] = now

            insight = CustomerInsight(
                person_id=person_id,
                zone=zone,
                dwell_seconds=visit.total_dwell,
                repeat_visit=visit.visit_count > 1,
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
                    f"[RETAIL ALERT] Customer {person_id} | "
                    f"{zone} | score={score} | "
                    f"dwell={visit.total_dwell:.0f}s"
                )

        # Move visits to completed memory when the customer is no
        # longer active. This lets a later return count as a repeat visit.
        for key, visit in list(self.visits.items()):
            person_id, zone = key

            if person_id in active_ids:
                continue

            if now - visit.last_seen >= self.stale_after:
                self.completed_visits[key] = visit.visit_count
                del self.visits[key]

        self.latest_alerts = alerts
        return insights, alerts

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    @staticmethod
    def _build_alert(person_id, zone, dwell_seconds, score):
        minutes = int(dwell_seconds // 60)
        seconds = int(dwell_seconds % 60)

        if minutes:
            dwell_text = f"{minutes}m {seconds}s"
        else:
            dwell_text = f"{seconds}s"

        return (
            "HIGH-INTENT CUSTOMER DETECTED\n"
            f"Customer: #{person_id}\n"
            f"Section: {zone}\n"
            f"Dwell time: {dwell_text}\n"
            f"Intent score: {score}\n"
            "Suggested action: Send sales associate."
        )

    def draw(self, frame, insights):
        """Draw retail intelligence without changing existing person boxes."""
        import cv2

        for insight in insights:
            # Find current person track is intentionally not required here;
            # main.py can draw the exact track box separately.
            pass

        # Compact summary for the POC.
        if insights:
            high = sum(
                item.intent_level == "HIGH"
                for item in insights
            )
            medium = sum(
                item.intent_level == "MEDIUM"
                for item in insights
            )

            cv2.putText(
                frame,
                f"Retail Intelligence | High: {high} | Medium: {medium}",
                (20, 290),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 200, 0),
                2,
                cv2.LINE_AA,
            )

    def zone_overlay(self, frame):
        import cv2

        h, w = frame.shape[:2]

        for name, box in self.zones.items():
            x1, y1, x2, y2 = box
            p1 = (int(x1 * w), int(y1 * h))
            p2 = (int(x2 * w), int(y2 * h))

            cv2.rectangle(
                frame,
                p1,
                p2,
                (255, 180, 0),
                2,
            )

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

    def get_customer(self, person_id):
        return self.latest.get(int(person_id))

    def status(self):
        return (
            f"Retail Intelligence: ACTIVE | "
            f"Zones: {len(self.zones)} | "
            f"Dwell threshold: {self.dwell_threshold:.0f}s"
        )
