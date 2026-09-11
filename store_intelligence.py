"""
Store-wide Retail Sales Intelligence.

This module sits above individual camera pipelines.

Responsibilities:
- Maintain camera -> zone mappings
- Maintain store-wide customer observations
- Track staff presence
- Detect unattended high-intent customers
- Prevent duplicate alerts
- Produce WhatsApp/Telegram-ready messages

No external DeepCamera dependency.
"""

from __future__ import annotations

import json
import os
import time

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class CameraConfig:
    camera_id: str
    source: str
    zones: Dict[str, Tuple[float, float, float, float]]
    enabled: bool = True


@dataclass
class CustomerObservation:
    customer_id: int
    camera_id: str
    zone: str

    first_seen: float
    last_seen: float

    dwell_seconds: float = 0.0

    product_interaction: bool = False
    phone_comparison: bool = False
    repeat_visit: bool = False

    staff_nearby: Optional[bool] = None

    intent_score: int = 0
    intent_level: str = "LOW"

    unattended_since: Optional[float] = None
    assistance_alert_sent: bool = False

    # Retail journey analytics
    zone_history: List[str] = field(default_factory=list)
    zone_dwell: Dict[str, float] = field(default_factory=dict)
    zone_entered_at: Optional[float] = None
    total_store_dwell: float = 0.0
    zones_visited: int = 0


@dataclass
class StaffObservation:
    staff_id: str
    camera_id: str
    zone: str
    last_seen: float


@dataclass
class ZoneVisit:
    zone: str
    entered_at: float
    exited_at: Optional[float] = None
    dwell_seconds: float = 0.0


@dataclass
class StaffAlert:
    customer_id: int
    camera_id: str
    zone: str
    dwell_seconds: float
    intent_score: int
    created_at: float

    message: str


# ============================================================
# STORE INTELLIGENCE
# ============================================================

class StoreIntelligence:

    DEFAULT_DWELL_THRESHOLD = 60.0
    DEFAULT_HIGH_INTENT_SCORE = 7

    DEFAULT_STAFF_DISTANCE = 150.0

    DEFAULT_UNATTENDED_CONFIRMATION = 10.0

    DEFAULT_ALERT_COOLDOWN = 120.0

    def __init__(
        self,
        dwell_threshold: float = DEFAULT_DWELL_THRESHOLD,
        high_intent_score: int = DEFAULT_HIGH_INTENT_SCORE,
        staff_distance: float = DEFAULT_STAFF_DISTANCE,
        unattended_confirmation: float = (
            DEFAULT_UNATTENDED_CONFIRMATION
        ),
        alert_cooldown: float = DEFAULT_ALERT_COOLDOWN,
    ):

        self.dwell_threshold = float(
            os.getenv(
                "STORE_DWELL_THRESHOLD",
                dwell_threshold,
            )
        )

        self.high_intent_score = int(
            os.getenv(
                "STORE_HIGH_INTENT_SCORE",
                high_intent_score,
            )
        )

        self.staff_distance = float(
            os.getenv(
                "STORE_STAFF_DISTANCE",
                staff_distance,
            )
        )

        self.unattended_confirmation = float(
            os.getenv(
                "STORE_UNATTENDED_CONFIRMATION",
                unattended_confirmation,
            )
        )

        self.alert_cooldown = float(
            os.getenv(
                "STORE_ALERT_COOLDOWN",
                alert_cooldown,
            )
        )

        self.cameras: Dict[
            str,
            CameraConfig
        ] = {}

        self.customers: Dict[
            int,
            CustomerObservation
        ] = {}

        self.staff: Dict[
            str,
            StaffObservation
        ] = {}

        self.last_alert_at: Dict[
            Tuple[int, str],
            float
        ] = {}

        self.latest_alerts: List[
            StaffAlert
        ] = []

        # Store-wide retail journey state.
        self.customer_journeys: Dict[int, List[ZoneVisit]] = {}
        self.completed_visits: Dict[Tuple[int, str], int] = {}
        self.zone_customer_counts: Dict[str, int] = {}

    # ========================================================
    # CAMERA CONFIGURATION
    # ========================================================

    def add_camera(
        self,
        camera_id: str,
        source: str,
        zones: Dict[
            str,
            Tuple[
                float,
                float,
                float,
                float
            ]
        ],
    ):

        self.cameras[camera_id] = CameraConfig(
            camera_id=camera_id,
            source=str(source),
            zones=zones,
        )

    def remove_camera(
        self,
        camera_id: str,
    ):

        self.cameras.pop(
            camera_id,
            None,
        )

    # ========================================================
    # ZONE LOOKUP
    # ========================================================

    @staticmethod
    def point_in_zone(
        point,
        zone,
        width,
        height,
    ):

        try:

            x = float(point[0])
            y = float(point[1])

            x1, y1, x2, y2 = (
                float(v)
                for v in zone
            )

        except (
            TypeError,
            ValueError,
            IndexError,
        ):

            return False

        return (
            x1 * width
            <= x
            <= x2 * width
            and
            y1 * height
            <= y
            <= y2 * height
        )

    def find_zone(
        self,
        camera_id,
        center,
        width,
        height,
    ):

        camera = self.cameras.get(
            camera_id
        )

        if camera is None:
            return None

        for (
            zone_name,
            zone
        ) in camera.zones.items():

            if self.point_in_zone(
                center,
                zone,
                width,
                height,
            ):

                return zone_name

        return None

    # ========================================================
    # CUSTOMER UPDATE
    # ========================================================

    def update_customer(
        self,
        customer_id: int,
        camera_id: str,
        zone: str,
        now: Optional[float] = None,
        product_interaction: bool = False,
        phone_comparison: bool = False,
        repeat_visit: bool = False,
        staff_nearby: Optional[bool] = None,
    ):
        """Update a customer while maintaining store journey analytics."""
        now = time.monotonic() if now is None else float(now)
        zone = str(zone or "Unassigned")
        customer = self.customers.get(customer_id)

        if customer is None:
            customer = CustomerObservation(
                customer_id=customer_id, camera_id=camera_id, zone=zone,
                first_seen=now, last_seen=now, zone_entered_at=now,
            )
            self.customers[customer_id] = customer
            self.customer_journeys[customer_id] = [ZoneVisit(zone, now)]
            customer.zone_history = [zone]
            customer.zone_dwell[zone] = 0.0
            customer.zones_visited = 1
        else:
            elapsed = min(max(0.0, now - customer.last_seen), 2.0)
            customer.total_store_dwell += elapsed
            customer.dwell_seconds += elapsed
            customer.zone_dwell[customer.zone] = customer.zone_dwell.get(customer.zone, 0.0) + elapsed

            journey = self.customer_journeys.setdefault(customer_id, [])
            if journey:
                journey[-1].dwell_seconds += elapsed
            if not journey:
                journey.append(ZoneVisit(zone, now))
            if zone != customer.zone:
                journey[-1].exited_at = now
                key = (customer_id, customer.zone)
                self.completed_visits[key] = self.completed_visits.get(key, 0) + 1
                already_visited = zone in customer.zone_history
                repeat_visit = repeat_visit or already_visited or self.completed_visits[key] > 1
                journey.append(ZoneVisit(zone, now))
                if not already_visited:
                    customer.zone_history.append(zone)
                customer.zone_dwell.setdefault(zone, 0.0)
                customer.zone_entered_at = now
                customer.zones_visited = len(customer.zone_history)
            customer.last_seen = now
            customer.camera_id = camera_id
            customer.zone = zone

        customer.product_interaction |= bool(product_interaction)
        customer.phone_comparison |= bool(phone_comparison)
        customer.repeat_visit |= bool(repeat_visit)
        if staff_nearby is not None:
            customer.staff_nearby = bool(staff_nearby)

        self._calculate_intent(customer)
        self._update_unattended_state(customer, now)
        alert = self._maybe_create_staff_alert(customer, now)
        self._refresh_zone_counts()
        return customer, alert

    # ========================================================
    # RETAIL JOURNEY / STORE ANALYTICS
    # ========================================================

    def get_customer_journey(self, customer_id: int) -> List[str]:
        """Return the customer's ordered zone journey."""
        customer = self.customers.get(customer_id)
        return list(customer.zone_history) if customer else []

    def get_customer_journey_detail(self, customer_id: int) -> List[dict]:
        """Return zone-by-zone journey data for dashboard/API use."""
        return [{
            "zone": visit.zone,
            "entered_at": visit.entered_at,
            "exited_at": visit.exited_at,
            "dwell_seconds": round(max(0.0, visit.dwell_seconds), 1),
        } for visit in self.customer_journeys.get(customer_id, [])]

    def _refresh_zone_counts(self):
        counts: Dict[str, int] = {}
        for customer in self.customers.values():
            counts[customer.zone] = counts.get(customer.zone, 0) + 1
        self.zone_customer_counts = counts

    def store_analytics(self) -> dict:
        """Return store-oriented KPIs for API/dashboard consumption."""
        customers = list(self.customers.values())
        high = sum(c.intent_level == "HIGH" for c in customers)
        medium = sum(c.intent_level == "MEDIUM" for c in customers)
        unattended = sum(c.intent_level == "HIGH" and c.staff_nearby is False for c in customers)
        avg_dwell = sum(c.total_store_dwell for c in customers) / len(customers) if customers else 0.0
        zone_stats = {}
        for zone, count in self.zone_customer_counts.items():
            values = [c.zone_dwell.get(zone, 0.0) for c in customers if zone in c.zone_dwell]
            zone_stats[zone] = {
                "active_customers": count,
                "average_dwell_seconds": round(sum(values) / len(values), 1) if values else 0.0,
            }
        return {
            "active_customers": len(customers),
            "high_intent_customers": high,
            "medium_intent_customers": medium,
            "unattended_opportunities": unattended,
            "average_store_dwell_seconds": round(avg_dwell, 1),
            "zones": zone_stats,
        }

    # ========================================================
    # INTENT
    # ========================================================

    def _calculate_intent(
        self,
        customer: CustomerObservation,
    ):

        score = 0

        if (
            customer.dwell_seconds
            >= self.dwell_threshold
        ):

            score += 2

        if customer.product_interaction:

            score += 3

        if customer.phone_comparison:

            score += 2

        if customer.repeat_visit:

            score += 2

        # Explicit staff absence is a sales-opportunity signal.
        # Unknown staff status is deliberately neutral.
        if customer.staff_nearby is False:

            score += 3

        customer.intent_score = score

        if (
            score
            >= self.high_intent_score
        ):

            customer.intent_level = "HIGH"

        elif score >= 4:

            customer.intent_level = "MEDIUM"

        else:

            customer.intent_level = "LOW"

    # ========================================================
    # UNATTENDED STATE
    # ========================================================

    def _update_unattended_state(
        self,
        customer: CustomerObservation,
        now: float,
    ):

        # Unknown staff status is NOT treated as absence.
        if customer.staff_nearby is not False:

            customer.unattended_since = None
            customer.assistance_alert_sent = False

            return

        # Customer isn't sufficiently high intent yet.
        if (
            customer.intent_level
            != "HIGH"
        ):

            customer.unattended_since = None
            customer.assistance_alert_sent = False

            return

        if customer.unattended_since is None:

            customer.unattended_since = now

    # ========================================================
    # STAFF ALERT
    # ========================================================

    def _maybe_create_staff_alert(
        self,
        customer: CustomerObservation,
        now: float,
    ):

        if (
            customer.unattended_since
            is None
        ):

            return None

        if (
            now
            - customer.unattended_since
            <
            self.unattended_confirmation
        ):

            return None

        if customer.assistance_alert_sent:

            return None

        key = (
            customer.customer_id,
            customer.zone,
        )

        last_alert = self.last_alert_at.get(
            key,
            -1e18,
        )

        if (
            now - last_alert
            <
            self.alert_cooldown
        ):

            return None

        message = self.build_staff_message(
            customer
        )

        alert = StaffAlert(
            customer_id=customer.customer_id,
            camera_id=customer.camera_id,
            zone=customer.zone,
            dwell_seconds=customer.dwell_seconds,
            intent_score=customer.intent_score,
            created_at=now,
            message=message,
        )

        customer.assistance_alert_sent = True

        self.last_alert_at[key] = now

        self.latest_alerts.append(
            alert
        )

        print(
            "[STAFF ALERT] "
            f"Customer #{customer.customer_id} | "
            f"{customer.zone} | "
            f"score={customer.intent_score}"
        )

        return alert

    # ========================================================
    # WHATSAPP / STAFF MESSAGE
    # ========================================================

    @staticmethod
    def build_staff_message(
        customer: CustomerObservation,
    ):

        minutes = int(
            customer.dwell_seconds // 60
        )

        seconds = int(
            customer.dwell_seconds % 60
        )

        if minutes:

            dwell = (
                f"{minutes}m "
                f"{seconds}s"
            )

        else:

            dwell = (
                f"{seconds}s"
            )

        return (
            "🚨 CUSTOMER ASSISTANCE REQUIRED\n\n"
            f"📍 Section: {customer.zone}\n"
            f"👤 Customer: #{customer.customer_id}\n"
            f"⏱️ Dwell time: {dwell}\n"
            f"🎯 Intent score: "
            f"{customer.intent_score}\n"
            "⚠️ Customer appears to be "
            "high-intent and unattended.\n\n"
            "👉 Please send a sales associate "
            "to assist the customer."
        )

    # ========================================================
    # STORE STATUS
    # ========================================================

    def status(self):

        return (
            "Store Intelligence: ACTIVE | "
            f"Cameras: {len(self.cameras)} | "
            f"Customers: {len(self.customers)} | "
            f"Staff: {len(self.staff)}"
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    def cleanup(
        self,
        now: Optional[float] = None,
        stale_after: float = 20.0,
    ):

        now = (
            time.monotonic()
            if now is None
            else float(now)
        )

        stale_customers = []

        for customer_id, customer in (
            self.customers.items()
        ):

            if (
                now - customer.last_seen
                >= stale_after
            ):

                stale_customers.append(
                    customer_id
                )

        for customer_id in stale_customers:

            del self.customers[
                customer_id
            ]