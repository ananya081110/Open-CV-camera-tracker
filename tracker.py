from __future__ import annotations

import json
import math
import time
from collections import deque
from pathlib import Path


# ============================================================
# PERSISTENT PERSON RE-ID CONFIGURATION
# ============================================================

# Identity profiles are kept indefinitely unless explicitly removed.
# This is different from ordinary active-track memory.
REID_PROFILE_PATH = (
    Path(__file__).resolve().parent
    / "logs"
    / "person_identities.json"
)

# Minimum score required to restore an existing person ID.
REID_MIN_SCORE = 0.72

# Appearance is the primary cue. Body geometry is secondary.
REID_APPEARANCE_WEIGHT = 0.70
REID_BODY_WEIGHT = 0.30

# Require a little more evidence for a weak appearance match.
REID_MIN_APPEARANCE_SCORE = 0.60

# Active tracks can be temporarily lost before entering identity memory.
# This only controls short-term track continuity, not permanent identity.
REID_ACTIVE_MEMORY_SECONDS = 300.0


# ============================================================
# HELPERS
# ============================================================

def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_dimensions(bbox):
    x1, y1, x2, y2 = bbox
    return max(1.0, x2 - x1), max(1.0, y2 - y1)


def midpoint(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def point_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def cosine_similarity(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))

    if na <= 1e-12 or nb <= 1e-12:
        return 0.0

    return max(0.0, min(1.0, dot / (na * nb)))


# ============================================================
# BODY SIGNATURE
# ============================================================

def build_body_signature(landmarks):
    """
    Normalized body geometry used as a secondary Re-ID cue.

    This is not facial recognition.
    """

    if not landmarks or len(landmarks) <= 28:
        return None

    indices = [11, 12, 23, 24, 25, 26, 27, 28]
    points = []

    for index in indices:
        point = landmarks[index]
        if point is None:
            return None

        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError, IndexError):
            return None

        if not (math.isfinite(x) and math.isfinite(y)):
            return None

        points.append((x, y))

    shoulder_center = midpoint(points[0], points[1])
    hip_center = midpoint(points[2], points[3])
    torso_length = point_distance(shoulder_center, hip_center)

    if torso_length < 0.001:
        return None

    signature = []

    for point in points:
        signature.extend([
            (point[0] - shoulder_center[0]) / torso_length,
            (point[1] - shoulder_center[1]) / torso_length,
        ])

    signature.extend([
        point_distance(points[0], points[1]) / torso_length,
        point_distance(points[2], points[3]) / torso_length,
        (
            point_distance(points[2], points[4])
            + point_distance(points[4], points[6])
        ) / torso_length,
        (
            point_distance(points[3], points[5])
            + point_distance(points[5], points[7])
        ) / torso_length,
    ])

    return tuple(signature)


def compare_body_signatures(signature_a, signature_b):
    if signature_a is None or signature_b is None:
        return 0.0

    if len(signature_a) != len(signature_b):
        return 0.0

    average_difference = sum(
        abs(a - b)
        for a, b in zip(signature_a, signature_b)
    ) / len(signature_a)

    return max(
        0.0,
        min(1.0, math.exp(-2.5 * average_difference))
    )


# ============================================================
# APPEARANCE SIGNATURE
# ============================================================

def build_appearance_signature(frame, bbox):
    """
    Build a lightweight appearance descriptor from the person's
    bounding-box crop.

    It intentionally avoids storing the person's image. Only the
    numerical descriptor is persisted.
    """

    if frame is None:
        return None

    try:
        import cv2
        import numpy as np

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        # Remove a small amount of the outer border so the descriptor
        # focuses more on the person than background pixels.
        ch, cw = crop.shape[:2]
        pad_x = int(cw * 0.08)
        pad_y = int(ch * 0.05)

        if cw > 2 * pad_x + 2 and ch > 2 * pad_y + 2:
            crop = crop[
                pad_y:ch - pad_y,
                pad_x:cw - pad_x
            ]

        crop = cv2.resize(
            crop,
            (64, 128),
            interpolation=cv2.INTER_AREA
        )

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Color/appearance histogram.
        hist = cv2.calcHist(
            [hsv],
            [0, 1],
            None,
            [16, 8],
            [0, 180, 0, 256],
        )

        hist = cv2.normalize(
            hist,
            hist,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_L2,
        ).flatten()

        # Coarse grayscale spatial appearance keeps some texture/layout
        # information while remaining lightweight.
        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY
        )

        small = cv2.resize(
            gray,
            (8, 16),
            interpolation=cv2.INTER_AREA
        ).astype("float32") / 255.0

        spatial = small.flatten()

        # Combine and normalize.
        vector = np.concatenate([
            hist.astype("float32"),
            spatial.astype("float32"),
        ])

        norm = float(np.linalg.norm(vector))

        if norm <= 1e-12:
            return None

        vector = vector / norm

        return tuple(float(x) for x in vector)

    except Exception:
        return None


def compare_appearance_signatures(signature_a, signature_b):
    return cosine_similarity(signature_a, signature_b)


# ============================================================
# TRACK
# ============================================================

class Track:

    def __init__(self, track_id, detection, now, appearance_signature=None):
        self.id = track_id
        self.center = detection["center"]
        self.landmarks = detection["landmarks"]
        self.bbox = detection["bbox"]

        self.state = "unknown"
        self.previous_state = "unknown"
        self.state_started = now
        self.candidate_state = None
        self.candidate_started = None
        self.posture_history = deque(maxlen=8)

        self.last_seen = now
        self.missed = 0
        self.previous_center = self.center

        self.previous_hip_y = None
        self.previous_hip_x = None
        self.body_movement = 0.0
        self.movement_history = deque(maxlen=12)

        self.movement_state = "unknown"
        self.previous_movement_state = "unknown"
        self.movement_state_started = now
        self.movement_candidate = None
        self.movement_candidate_started = None
        self.movement_speed = 0.0
        self.movement_direction = "stationary"
        self.speed_history = deque(maxlen=8)
        self.movement_state_history = deque(maxlen=3)

        self.STATIONARY_THRESHOLD = 2.0
        self.WALKING_THRESHOLD = 8.0
        self.RUNNING_THRESHOLD = 18.0
        self.SUDDEN_MOVEMENT_THRESHOLD = 35.0
        self.MOVEMENT_CONFIRM_FRAMES = 3

        self.hip_history = deque(maxlen=12)
        self.torso_history = deque(maxlen=12)
        self.fall_candidate_started = None
        self.fall_alerted_at = None
        self.fall_cooldown_until = 0.0
        self.fall_motion_detected = False
        self.fall_horizontal_detected = False
        self.fall_low_position_detected = False

        self.sitting_alerted = False
        self.standing_alerted = False

        self.zone_started = None
        self.zone_alerted = False
        self.was_in_zone = False
        self.zone_initialized = False
        self.last_alerts = {}

        # Persistent identity descriptors.
        self.body_signature = build_body_signature(self.landmarks)
        self.appearance_signature = appearance_signature

        self.reidentified = False
        self.reidentification_count = 0
        self.last_reidentified_at = None

    def update(self, detection, now, appearance_signature=None):
        old_center = self.center

        self.center = detection["center"]
        self.landmarks = detection["landmarks"]
        self.bbox = detection["bbox"]
        self.previous_center = old_center
        self.last_seen = now
        self.missed = 0

        self.body_movement = math.hypot(
            self.center[0] - old_center[0],
            self.center[1] - old_center[1],
        )

        self.movement_history.append(
            (now, self.body_movement)
        )

        self._update_movement_analysis(now)

        new_body_signature = build_body_signature(self.landmarks)

        if new_body_signature is not None:
            if self.body_signature is None:
                self.body_signature = new_body_signature
            else:
                self.body_signature = tuple(
                    0.80 * old + 0.20 * new
                    for old, new in zip(
                        self.body_signature,
                        new_body_signature,
                    )
                )

        if appearance_signature is not None:
            if self.appearance_signature is None:
                self.appearance_signature = appearance_signature
            else:
                # Slow adaptation prevents one bad frame from changing
                # the person's stored identity.
                self.appearance_signature = tuple(
                    0.90 * old + 0.10 * new
                    for old, new in zip(
                        self.appearance_signature,
                        appearance_signature,
                    )
                )

    def _update_movement_analysis(self, now):
        self.speed_history.append(self.body_movement)

        if self.speed_history:
            self.movement_speed = (
                sum(self.speed_history) / len(self.speed_history)
            )

        if self.body_movement >= self.SUDDEN_MOVEMENT_THRESHOLD:
            raw_state = "sudden_movement"
        elif self.movement_speed >= self.RUNNING_THRESHOLD:
            raw_state = "running"
        elif self.movement_speed >= self.WALKING_THRESHOLD:
            raw_state = "walking"
        else:
            raw_state = "stationary"

        if raw_state == self.movement_state:
            self.movement_candidate = None
            self.movement_candidate_started = None
            self.movement_state_history.clear()
        else:
            if self.movement_candidate != raw_state:
                self.movement_candidate = raw_state
                self.movement_candidate_started = now
                self.movement_state_history.clear()

            self.movement_state_history.append(raw_state)

            if len(self.movement_state_history) >= self.MOVEMENT_CONFIRM_FRAMES:
                self.previous_movement_state = self.movement_state
                self.movement_state = raw_state
                self.movement_state_started = now
                self.movement_candidate = None
                self.movement_candidate_started = None
                self.movement_state_history.clear()

        self._update_movement_direction()

    def _update_movement_direction(self):
        dx = self.center[0] - self.previous_center[0]
        dy = self.center[1] - self.previous_center[1]

        if abs(dx) < 2 and abs(dy) < 2:
            self.movement_direction = "stationary"
            return

        if abs(dx) >= abs(dy):
            self.movement_direction = "right" if dx > 0 else "left"
        else:
            self.movement_direction = "down" if dy > 0 else "up"


# ============================================================
# CENTROID TRACKER + PERSISTENT RE-ID
# ============================================================

class CentroidTracker:

    def __init__(self, max_distance=140, max_missed=20):
        self.max_distance = max_distance
        self.max_missed = max_missed

        self.next_id = 1
        self.tracks = {}
        self.lost_tracks = {}

        # Persistent identity profiles:
        # person ID -> numerical appearance/body descriptors.
        self.identity_profiles = {}
        self.identity_last_seen = {}

        self.reid_memory_seconds = REID_ACTIVE_MEMORY_SECONDS

        self._load_identity_profiles()

    # =========================================================
    # PERSISTENCE
    # =========================================================

    def _load_identity_profiles(self):
        try:
            REID_PROFILE_PATH.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            if not REID_PROFILE_PATH.exists():
                return

            data = json.loads(
                REID_PROFILE_PATH.read_text()
            )

            self.next_id = max(
                1,
                int(data.get("next_id", 1))
            )

            profiles = data.get("profiles", {})

            for key, profile in profiles.items():
                try:
                    person_id = int(key)

                    self.identity_profiles[person_id] = {
                        "appearance": tuple(
                            profile.get("appearance", [])
                        ) or None,
                        "body": tuple(
                            profile.get("body", [])
                        ) or None,
                    }

                    self.identity_last_seen[person_id] = float(
                        profile.get("last_seen", 0.0)
                    )

                except Exception:
                    continue

            if self.identity_profiles:
                self.next_id = max(
                    self.next_id,
                    max(self.identity_profiles) + 1
                )

            print(
                f"[RE-ID] Loaded "
                f"{len(self.identity_profiles)} persistent identities."
            )

        except Exception as exc:
            print(
                f"[RE-ID] Could not load identity profiles: {exc}"
            )

    def _save_identity_profiles(self):
        try:
            REID_PROFILE_PATH.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            payload = {
                "next_id": self.next_id,
                "profiles": {},
            }

            for person_id, profile in self.identity_profiles.items():
                payload["profiles"][str(person_id)] = {
                    "appearance": list(
                        profile.get("appearance")
                        or []
                    ),
                    "body": list(
                        profile.get("body")
                        or []
                    ),
                    "last_seen": self.identity_last_seen.get(
                        person_id,
                        0.0
                    ),
                }

            temp_path = REID_PROFILE_PATH.with_suffix(".tmp")
            temp_path.write_text(
                json.dumps(payload)
            )
            temp_path.replace(REID_PROFILE_PATH)

        except Exception as exc:
            print(
                f"[RE-ID] Could not save identity profiles: {exc}"
            )

    def _register_or_update_profile(
        self,
        track,
        appearance_signature,
        now
    ):
        appearance = (
            appearance_signature
            if appearance_signature is not None
            else track.appearance_signature
        )

        body = track.body_signature

        existing = self.identity_profiles.get(track.id)

        if existing is None:
            self.identity_profiles[track.id] = {
                "appearance": appearance,
                "body": body,
            }
        else:
            if appearance is not None:
                old = existing.get("appearance")
                if old is None or len(old) != len(appearance):
                    existing["appearance"] = appearance
                else:
                    existing["appearance"] = tuple(
                        0.95 * a + 0.05 * b
                        for a, b in zip(old, appearance)
                    )

            if body is not None:
                old_body = existing.get("body")
                if old_body is None or len(old_body) != len(body):
                    existing["body"] = body
                else:
                    existing["body"] = tuple(
                        0.95 * a + 0.05 * b
                        for a, b in zip(old_body, body)
                    )

        self.identity_last_seen[track.id] = now

    # =========================================================
    # DISTANCE
    # =========================================================

    @staticmethod
    def distance(a, b):
        return math.hypot(
            a[0] - b[0],
            a[1] - b[1]
        )

    # =========================================================
    # UPDATE
    # =========================================================

    def update(self, detections, now, frame=None):
        # Clean only short-term lost track objects.
        self._cleanup_lost_tracks(now)

        if not self.tracks:
            unmatched = set(range(len(detections)))

            for i in list(unmatched):
                restored = self._try_reidentify(
                    detections[i],
                    now,
                    frame,
                )

                if restored is not None:
                    unmatched.remove(i)

            for i in unmatched:
                self._add(
                    detections[i],
                    now,
                    frame,
                )

            self._save_identity_profiles()
            return list(self.tracks.values())

        unmatched_tracks = set(self.tracks.keys())
        unmatched_detections = set(range(len(detections)))

        pairs = []

        for tid, track in self.tracks.items():
            for i, detection in enumerate(detections):
                d = self.distance(
                    track.center,
                    detection["center"]
                )
                pairs.append((d, tid, i))

        # Normal frame-to-frame tracking first.
        for d, tid, i in sorted(pairs):
            if d > self.max_distance:
                break

            if (
                tid not in unmatched_tracks
                or i not in unmatched_detections
            ):
                continue

            appearance = build_appearance_signature(
                frame,
                detections[i]["bbox"],
            )

            self.tracks[tid].update(
                detections[i],
                now,
                appearance,
            )

            self._register_or_update_profile(
                self.tracks[tid],
                appearance,
                now,
            )

            unmatched_tracks.remove(tid)
            unmatched_detections.remove(i)

        # Re-identify detections that could not be matched normally.
        for i in list(unmatched_detections):
            restored = self._try_reidentify(
                detections[i],
                now,
                frame,
            )

            if restored is not None:
                unmatched_detections.remove(i)

        # Anything remaining is a genuinely new person.
        for i in unmatched_detections:
            self._add(
                detections[i],
                now,
                frame,
            )

        # Move disappeared active tracks into short-term memory.
        for tid in list(unmatched_tracks):
            track = self.tracks[tid]
            track.missed += 1

            if track.missed > self.max_missed:
                self.lost_tracks[tid] = track
                del self.tracks[tid]

        self._save_identity_profiles()
        return list(self.tracks.values())

    # =========================================================
    # PERSISTENT RE-ID
    # =========================================================

    def _try_reidentify(self, detection, now, frame):
        detection_appearance = build_appearance_signature(
            frame,
            detection["bbox"],
        )

        detection_body = build_body_signature(
            detection["landmarks"]
        )

        if detection_appearance is None and detection_body is None:
            return None

        best_id = None
        best_score = 0.0
        best_appearance = 0.0
        best_body = 0.0

        # Compare against ALL known identities, not only people who
        # disappeared seconds ago. This is what makes the ID persistent.
        for person_id, profile in self.identity_profiles.items():
            if person_id in self.tracks:
                continue

            profile_appearance = profile.get("appearance")
            profile_body = profile.get("body")

            appearance_score = compare_appearance_signatures(
                profile_appearance,
                detection_appearance,
            )

            body_score = compare_body_signatures(
                profile_body,
                detection_body,
            )

            if (
                appearance_score < REID_MIN_APPEARANCE_SCORE
                and body_score < 0.72
            ):
                continue

            if profile_appearance is not None and detection_appearance is not None:
                score = (
                    REID_APPEARANCE_WEIGHT * appearance_score
                    + REID_BODY_WEIGHT * body_score
                )
            else:
                score = body_score

            if score > best_score:
                best_score = score
                best_id = person_id
                best_appearance = appearance_score
                best_body = body_score

        if best_id is None or best_score < REID_MIN_SCORE:
            return None

        # Prefer a currently lost Track object so posture/zone history
        # survives the disappearance as well.
        track = self.lost_tracks.pop(best_id, None)

        appearance = detection_appearance

        if track is None:
            track = Track(
                best_id,
                detection,
                now,
                appearance,
            )
        else:
            track.update(
                detection,
                now,
                appearance,
            )

        track.reidentified = True
        track.reidentification_count += 1
        track.last_reidentified_at = now

        self.tracks[best_id] = track

        self._register_or_update_profile(
            track,
            appearance,
            now,
        )

        print(
            "[RE-ID] Restored "
            f"Person ID {best_id} "
            f"(score={best_score:.2f}, "
            f"appearance={best_appearance:.2f}, "
            f"body={best_body:.2f})"
        )

        return track

    # =========================================================
    # LOST TRACK CLEANUP
    # =========================================================

    def _cleanup_lost_tracks(self, now):
        expired = []

        for track_id, track in self.lost_tracks.items():
            if (
                now - track.last_seen
                > self.reid_memory_seconds
            ):
                expired.append(track_id)

        for track_id in expired:
            del self.lost_tracks[track_id]

    # =========================================================
    # ADD
    # =========================================================

    def _add(self, detection, now, frame=None):
        person_id = self.next_id
        self.next_id += 1

        appearance = build_appearance_signature(
            frame,
            detection["bbox"],
        )

        track = Track(
            person_id,
            detection,
            now,
            appearance,
        )

        self.tracks[person_id] = track

        self._register_or_update_profile(
            track,
            appearance,
            now,
        )

        print(
            f"[TRACK] New Person ID {person_id}"
        )
