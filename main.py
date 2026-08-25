import csv
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config
from database import init_db
from detector import PoseDetector
from download_model import MODEL_PATH
from event_manager import capture_event
from tracker import CentroidTracker


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

LOG_DIR = ROOT / "logs"
ALERT_DIR = ROOT / "alerts"
CAPTURE_DIR = ROOT / "captured"

EVENT_LOG = LOG_DIR / "events.csv"


# ============================================================
# CONFIG HELPERS
# ============================================================

def cfg(name, default):
    """
    Read a configuration value if it exists.
    Otherwise use the supplied default.
    """
    return getattr(config, name, default)


CAMERA_INDEX = cfg("CAMERA_INDEX", 0)
FRAME_WIDTH = cfg("FRAME_WIDTH", 1280)
FRAME_HEIGHT = cfg("FRAME_HEIGHT", 720)

MAX_TRACK_DISTANCE = cfg(
    "MAX_TRACK_DISTANCE_PX",
    140
)

MAX_MISSED_FRAMES = cfg(
    "MAX_MISSED_FRAMES",
    20
)

FALL_DROP_RATIO = cfg(
    "FALL_DROP_RATIO",
    0.10
)

FALL_CONFIRM_SECONDS = cfg(
    "FALL_CONFIRM_SECONDS",
    0.7
)

ALERT_COOLDOWN_SECONDS = cfg(
    "ALERT_COOLDOWN_SECONDS",
    10
)

SITTING_DWELL_SECONDS = cfg(
    "SITTING_DWELL_SECONDS",
    30
)

STANDING_DWELL_SECONDS = cfg(
    "STANDING_DWELL_SECONDS",
    30
)

ZONE_DWELL_SECONDS = cfg(
    "ZONE_DWELL_SECONDS",
    30
)


# ============================================================
# LANDMARK INDICES
# MediaPipe Pose landmark indices
# ============================================================

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_HIP = 23
RIGHT_HIP = 24

LEFT_KNEE = 25
RIGHT_KNEE = 26

LEFT_ANKLE = 27
RIGHT_ANKLE = 28


# ============================================================
# INITIALIZATION
# ============================================================

def ensure_dirs():
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    ALERT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CAPTURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not EVENT_LOG.exists():
        with EVENT_LOG.open(
            "w",
            newline=""
        ) as f:

            csv.writer(f).writerow([
                "timestamp",
                "track_id",
                "event",
                "details"
            ])


def ensure_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Pose model not found.\n"
            "Run:\n"
            "python download_model.py"
        )


# ============================================================
# GENERAL HELPERS
# ============================================================

def fmt(seconds):
    seconds = max(
        0,
        int(seconds)
    )

    return (
        f"{seconds // 60:02d}:"
        f"{seconds % 60:02d}"
    )


def distance(a, b):
    return math.hypot(
        a[0] - b[0],
        a[1] - b[1]
    )


def midpoint(a, b):
    return (
        (a[0] + b[0]) / 2,
        (a[1] + b[1]) / 2
    )


def angle(a, b, c):
    """
    Angle ABC.
    """

    a = np.array(
        a,
        dtype=float
    )

    b = np.array(
        b,
        dtype=float
    )

    c = np.array(
        c,
        dtype=float
    )

    ba = a - b
    bc = c - b

    denom = (
        np.linalg.norm(ba)
        *
        np.linalg.norm(bc)
    )

    if denom == 0:
        return 180.0

    cosine = (
        np.dot(ba, bc)
        /
        denom
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return math.degrees(
        math.acos(cosine)
    )


# ============================================================
# LANDMARK VALIDATION
# ============================================================

def valid_point(point):
    """
    Check whether a landmark contains a usable x/y pair.
    """

    if point is None:
        return False

    if len(point) < 2:
        return False

    x = point[0]
    y = point[1]

    if not np.isfinite(x):
        return False

    if not np.isfinite(y):
        return False

    return True


def point_near_frame_bottom(point, frame_height):
    """
    Detect landmarks that are effectively outside
    the useful lower part of the camera frame.
    """

    if not valid_point(point):
        return True

    y = point[1]

    return y >= frame_height * 0.94


def lower_body_visibility(pts, frame_height):
    """
    Determine whether enough lower-body information
    exists to classify sitting/standing reliably.

    This is deliberately conservative.

    If both knees/ankles are missing or effectively
    outside the frame, we return LIMITED VIEW instead
    of guessing STANDING.
    """

    knee_left = pts[LEFT_KNEE]
    knee_right = pts[RIGHT_KNEE]

    ankle_left = pts[LEFT_ANKLE]
    ankle_right = pts[RIGHT_ANKLE]

    knees_valid = (
        valid_point(knee_left)
        and
        valid_point(knee_right)
    )

    ankles_valid = (
        valid_point(ankle_left)
        and
        valid_point(ankle_right)
    )

    knees_at_bottom = (
        point_near_frame_bottom(
            knee_left,
            frame_height
        )
        and
        point_near_frame_bottom(
            knee_right,
            frame_height
        )
    )

    ankles_at_bottom = (
        point_near_frame_bottom(
            ankle_left,
            frame_height
        )
        and
        point_near_frame_bottom(
            ankle_right,
            frame_height
        )
    )

    # Strong indication that lower body is outside frame.
    if knees_at_bottom and not ankles_valid:
        return False

    if ankles_at_bottom:
        return False

    # If knees are completely unavailable, do not guess.
    if not knees_valid:
        return False

    return True


# ============================================================
# POSTURE FEATURES
# ============================================================

def get_body_features(pts, frame_height):
    """
    Calculate body geometry.

    Returns:
        dict with:
        - torso_angle
        - knee_angle
        - hip_angle
        - lower_body_visible
    """

    shoulders_valid = (
        valid_point(pts[LEFT_SHOULDER])
        and
        valid_point(pts[RIGHT_SHOULDER])
    )

    hips_valid = (
        valid_point(pts[LEFT_HIP])
        and
        valid_point(pts[RIGHT_HIP])
    )

    if not shoulders_valid or not hips_valid:
        return {
            "valid": False,
            "lower_body_visible": False,
            "torso_angle": None,
            "knee_angle": None,
            "hip_angle": None
        }

    shoulder = midpoint(
        pts[LEFT_SHOULDER],
        pts[RIGHT_SHOULDER]
    )

    hip = midpoint(
        pts[LEFT_HIP],
        pts[RIGHT_HIP]
    )

    # --------------------------------------------------------
    # Torso angle
    #
    # Vertical torso ≈ 0
    # Horizontal torso ≈ 90
    # --------------------------------------------------------

    dx = hip[0] - shoulder[0]
    dy = hip[1] - shoulder[1]

    torso_angle = abs(
        math.degrees(
            math.atan2(
                dx,
                dy
            )
        )
    )

    # --------------------------------------------------------
    # Lower body visibility
    # --------------------------------------------------------

    lower_visible = lower_body_visibility(
        pts,
        frame_height
    )

    # --------------------------------------------------------
    # Knee angles
    # --------------------------------------------------------

    knee_angles = []

    if (
        valid_point(pts[LEFT_HIP])
        and
        valid_point(pts[LEFT_KNEE])
        and
        valid_point(pts[LEFT_ANKLE])
    ):
        knee_angles.append(
            angle(
                pts[LEFT_HIP],
                pts[LEFT_KNEE],
                pts[LEFT_ANKLE]
            )
        )

    if (
        valid_point(pts[RIGHT_HIP])
        and
        valid_point(pts[RIGHT_KNEE])
        and
        valid_point(pts[RIGHT_ANKLE])
    ):
        knee_angles.append(
            angle(
                pts[RIGHT_HIP],
                pts[RIGHT_KNEE],
                pts[RIGHT_ANKLE]
            )
        )

    knee_angle = (
        sum(knee_angles) / len(knee_angles)
        if knee_angles
        else None
    )

    # --------------------------------------------------------
    # Hip angles
    # --------------------------------------------------------

    hip_angles = []

    if (
        valid_point(pts[LEFT_SHOULDER])
        and
        valid_point(pts[LEFT_HIP])
        and
        valid_point(pts[LEFT_KNEE])
    ):
        hip_angles.append(
            angle(
                pts[LEFT_SHOULDER],
                pts[LEFT_HIP],
                pts[LEFT_KNEE]
            )
        )

    if (
        valid_point(pts[RIGHT_SHOULDER])
        and
        valid_point(pts[RIGHT_HIP])
        and
        valid_point(pts[RIGHT_KNEE])
    ):
        hip_angles.append(
            angle(
                pts[RIGHT_SHOULDER],
                pts[RIGHT_HIP],
                pts[RIGHT_KNEE]
            )
        )

    hip_angle = (
        sum(hip_angles) / len(hip_angles)
        if hip_angles
        else None
    )

    return {
        "valid": True,
        "lower_body_visible": lower_visible,
        "torso_angle": torso_angle,
        "knee_angle": knee_angle,
        "hip_angle": hip_angle
    }


# ============================================================
# POSTURE CLASSIFICATION
# ============================================================

def classify_posture(features):
    """
    Conservative posture classification.

    IMPORTANT:
    When lower body is not visible we return LIMITED_VIEW.
    We never guess STANDING from only the upper body.
    """

    if not features["valid"]:
        return (
            "limited_view",
            "upper body landmarks insufficient"
        )

    lower_visible = features[
        "lower_body_visible"
    ]

    torso_angle = features[
        "torso_angle"
    ]

    knee_angle = features[
        "knee_angle"
    ]

    hip_angle = features[
        "hip_angle"
    ]

    # --------------------------------------------------------
    # LIMITED VIEW
    # --------------------------------------------------------

    if not lower_visible:
        return (
            "limited_view",
            "lower body not reliably visible"
        )

    # We need actual leg information.
    if knee_angle is None:
        return (
            "limited_view",
            "knee landmarks unavailable"
        )

    # --------------------------------------------------------
    # FALLING
    # --------------------------------------------------------

    if torso_angle >= 55:
        return (
            "falling",
            "torso angle indicates horizontal posture"
        )

    # --------------------------------------------------------
    # SITTING
    #
    # Sitting normally produces a substantially
    # reduced knee angle.
    # --------------------------------------------------------

    if knee_angle < 145:
        return (
            "sitting",
            f"knee angle {knee_angle:.1f}"
        )

    # More tolerant sitting condition.
    if (
        knee_angle < 165
        and
        hip_angle is not None
        and
        hip_angle < 125
    ):
        return (
            "sitting",
            (
                f"knee={knee_angle:.1f}, "
                f"hip={hip_angle:.1f}"
            )
        )

    # --------------------------------------------------------
    # STANDING
    # --------------------------------------------------------

    if (
        torso_angle < 35
        and
        knee_angle >= 165
    ):
        return (
            "standing",
            (
                f"torso={torso_angle:.1f}, "
                f"knee={knee_angle:.1f}"
            )
        )

    # --------------------------------------------------------
    # WALKING / MOVING
    # --------------------------------------------------------

    if knee_angle < 175:
        return (
            "walking",
            f"knee angle {knee_angle:.1f}"
        )

    return (
        "unknown",
        "posture not confidently classified"
    )


# ============================================================
# ZONE
# ============================================================

def inside_zone(center, width, height):

    zone = cfg(
        "ZONE",
        (
            0.45,
            0.20,
            0.90,
            0.90
        )
    )

    x1, y1, x2, y2 = zone

    return (
        x1 * width
        <= center[0]
        <= x2 * width
        and
        y1 * height
        <= center[1]
        <= y2 * height
    )


# ============================================================
# DRAWING
# ============================================================

def draw_skeleton(
    frame,
    points,
    color=(0, 255, 0)
):

    connections = [
        (11, 12),

        (11, 13),
        (13, 15),

        (12, 14),
        (14, 16),

        (11, 23),
        (12, 24),

        (23, 24),

        (23, 25),
        (25, 27),

        (24, 26),
        (26, 28),

        (27, 29),
        (29, 31),

        (28, 30),
        (30, 32)
    ]

    for a, b in connections:

        if (
            a >= len(points)
            or
            b >= len(points)
        ):
            continue

        if (
            not valid_point(points[a])
            or
            not valid_point(points[b])
        ):
            continue

        cv2.line(
            frame,
            (
                int(points[a][0]),
                int(points[a][1])
            ),
            (
                int(points[b][0]),
                int(points[b][1])
            ),
            color,
            2
        )

    for point in points:

        if not valid_point(point):
            continue

        cv2.circle(
            frame,
            (
                int(point[0]),
                int(point[1])
            ),
            3,
            (255, 255, 255),
            -1
        )


def draw_zone(frame):

    h, w = frame.shape[:2]

    zone = cfg(
        "ZONE",
        (
            0.45,
            0.20,
            0.90,
            0.90
        )
    )

    x1, y1, x2, y2 = zone

    cv2.rectangle(
        frame,
        (
            int(x1 * w),
            int(y1 * h)
        ),
        (
            int(x2 * w),
            int(y2 * h)
        ),
        (255, 180, 0),
        2
    )

    cv2.putText(
        frame,
        "MONITORING ZONE",
        (
            int(x1 * w),
            max(
                25,
                int(y1 * h) - 8
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 180, 0),
        2
    )


# ============================================================
# EVENT HELPERS
# ============================================================

def capture_and_log(
    frame,
    track_id,
    event_type,
    activity,
    details,
    category="events"
):
    """
    Single pipeline for:
        1. saving image
        2. logging SQLite event
        3. printing event
    """

    try:

        event_id, path = capture_event(
            frame=frame,
            person_id=track_id,
            event_type=event_type,
            activity=activity,
            details=details,
            category=category
        )

        return event_id, path

    except Exception as exc:

        print(
            f"[ERROR] Could not capture "
            f"{event_type} for Person {track_id}: "
            f"{exc}"
        )

        return None, None


# ============================================================
# EVENT COOLDOWN
# ============================================================

def can_alert(
    track,
    event_name,
    now,
    cooldown=ALERT_COOLDOWN_SECONDS
):

    if not hasattr(
        track,
        "event_alert_times"
    ):
        track.event_alert_times = {}

    previous = track.event_alert_times.get(
        event_name
    )

    if previous is None:
        return True

    return (
        now - previous
        >= cooldown
    )


def mark_alert(
    track,
    event_name,
    now
):

    if not hasattr(
        track,
        "event_alert_times"
    ):
        track.event_alert_times = {}

    track.event_alert_times[
        event_name
    ] = now


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("AI CAMERA TRACKER")
    print("=" * 60)

    ensure_model()
    ensure_dirs()

    # Initialize SQLite database.
    init_db()

    print("[INFO] SQLite database initialized.")
    print(
        f"[INFO] Database: "
        f"{ROOT / 'logs' / 'camera_events.db'}"
    )

    # --------------------------------------------------------
    # Detector
    # --------------------------------------------------------

    detector = PoseDetector(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Tracker
    # --------------------------------------------------------

    tracker = CentroidTracker(
        MAX_TRACK_DISTANCE,
        MAX_MISSED_FRAMES
    )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Camera could not be opened. "
            "Check macOS camera permission."
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    print("[INFO] Camera started.")
    print("[INFO] Press C to capture manually.")
    print("[INFO] Press Q to quit.")

    timestamp_ms = 0

    previous_time = time.monotonic()

    # ========================================================
    # MAIN LOOP
    # ========================================================

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                print(
                    "[ERROR] Camera frame could not be read."
                )
                break

            now = time.monotonic()

            elapsed_frame = (
                now - previous_time
            )

            previous_time = now

            timestamp_ms += max(
                1,
                int(
                    elapsed_frame * 1000
                )
            )

            h, w = frame.shape[:2]

            # ------------------------------------------------
            # Pose detection
            # ------------------------------------------------

            detections = detector.detect(
                frame,
                timestamp_ms
            )

            # ------------------------------------------------
            # Tracking
            # ------------------------------------------------

            tracks = tracker.update(
                detections,
                now
            )

            # ------------------------------------------------
            # Draw monitoring zone
            # ------------------------------------------------

            draw_zone(frame)

            # =================================================
            # PROCESS EACH PERSON
            # =================================================

            for track in tracks:

                if track.missed:
                    continue

                pts = track.landmarks

                # ------------------------------------------------
                # Safety check
                # ------------------------------------------------

                if len(pts) < 33:
                    continue

                # ------------------------------------------------
                # Calculate posture features
                # ------------------------------------------------

                features = get_body_features(
                    pts,
                    h
                )

                state, reason = classify_posture(
                    features
                )

                torso_angle = features[
                    "torso_angle"
                ]

                knee_angle = features[
                    "knee_angle"
                ]

                hip_angle = features[
                    "hip_angle"
                ]

                lower_visible = features[
                    "lower_body_visible"
                ]

                # ------------------------------------------------
                # State smoothing
                #
                # Prevents one-frame fluctuations.
                # ------------------------------------------------

                if not hasattr(
                    track,
                    "state_history"
                ):
                    track.state_history = deque(
                        maxlen=7
                    )

                track.state_history.append(
                    state
                )

                counts = {}

                for s in track.state_history:
                    counts[s] = (
                        counts.get(s, 0)
                        + 1
                    )

                stable_state = max(
                    counts,
                    key=counts.get
                )

                # Only accept a new state when
                # it appears repeatedly.
                if (
                    stable_state != track.state
                    and
                    counts[stable_state] >= 3
                ):

                    previous_state = (
                        track.state
                    )

                    track.previous_state = (
                        previous_state
                    )

                    track.state = (
                        stable_state
                    )

                    track.state_started = now

                    # Reset activity alerts.
                    track.sitting_alerted = False
                    track.standing_alerted = False

                    # ------------------------------------------------
                    # STATE CHANGE EVENT
                    # ------------------------------------------------

                    if (
                        stable_state
                        != "unknown"
                    ):

                        if can_alert(
                            track,
                            f"STATE_{stable_state}",
                            now,
                            3
                        ):

                            details = (
                                f"previous={previous_state}; "
                                f"reason={reason}; "
                                f"torso_angle="
                                f"{torso_angle}; "
                                f"knee_angle="
                                f"{knee_angle}; "
                                f"hip_angle="
                                f"{hip_angle}; "
                                f"lower_body_visible="
                                f"{lower_visible}"
                            )

                            capture_and_log(
                                frame,
                                track.id,
                                "POSTURE_CHANGE",
                                stable_state,
                                details,
                                category="posture"
                            )

                            mark_alert(
                                track,
                                f"STATE_{stable_state}",
                                now
                            )

                state = track.state

                # ------------------------------------------------
                # If state is still unknown, use current
                # classification for display.
                # ------------------------------------------------

                display_state = (
                    state
                    if state != "unknown"
                    else stable_state
                )

                # ------------------------------------------------
                # Hip position / fall tracking
                # ------------------------------------------------

                if (
                    valid_point(
                        pts[LEFT_HIP]
                    )
                    and
                    valid_point(
                        pts[RIGHT_HIP]
                    )
                ):

                    hip = midpoint(
                        pts[LEFT_HIP],
                        pts[RIGHT_HIP]
                    )

                    hip_y = (
                        hip[1] / h
                    )

                else:

                    hip = track.center

                    hip_y = (
                        hip[1] / h
                    )

                hip_drop = 0

                if (
                    hasattr(
                        track,
                        "previous_hip_y"
                    )
                    and
                    track.previous_hip_y
                    is not None
                ):

                    hip_drop = (
                        hip_y
                        -
                        track.previous_hip_y
                    )

                track.previous_hip_y = hip_y

                # =================================================
                # FALL DETECTION
                # =================================================

                if (
                    track.previous_state
                    in (
                        "standing",
                        "walking"
                    )
                    and
                    display_state
                    == "falling"
                    and
                    hip_drop
                    >= FALL_DROP_RATIO
                ):

                    if (
                        track.fall_candidate_started
                        is None
                    ):

                        track.fall_candidate_started = now

                if (
                    display_state
                    != "falling"
                ):

                    track.fall_candidate_started = None

                if (
                    track.fall_candidate_started
                    is not None
                    and
                    now
                    -
                    track.fall_candidate_started
                    >= FALL_CONFIRM_SECONDS
                    and
                    can_alert(
                        track,
                        "FALL",
                        now
                    )
                ):

                    details = (
                        f"torso_angle="
                        f"{torso_angle}; "
                        f"hip_drop="
                        f"{hip_drop:.3f}; "
                        f"lower_body_visible="
                        f"{lower_visible}"
                    )

                    capture_and_log(
                        frame,
                        track.id,
                        "FALL_DETECTED",
                        "falling",
                        details,
                        category="falls"
                    )

                    mark_alert(
                        track,
                        "FALL",
                        now
                    )

                    track.fall_candidate_started = None

                # =================================================
                # STATE DWELL
                # =================================================

                state_elapsed = (
                    now
                    -
                    track.state_started
                )

                # -------------------------------------------------
                # SITTING
                # -------------------------------------------------

                if (
                    display_state
                    == "sitting"
                    and
                    state_elapsed
                    >= SITTING_DWELL_SECONDS
                ):

                    if (
                        not track.sitting_alerted
                    ):

                        details = (
                            f"{fmt(state_elapsed)} "
                            f"sitting; "
                            f"knee_angle="
                            f"{knee_angle}"
                        )

                        capture_and_log(
                            frame,
                            track.id,
                            "SITTING_DWELL_THRESHOLD",
                            "sitting",
                            details,
                            category="sitting"
                        )

                        track.sitting_alerted = True

                # -------------------------------------------------
                # STANDING
                # -------------------------------------------------

                if (
                    display_state
                    == "standing"
                    and
                    state_elapsed
                    >= STANDING_DWELL_SECONDS
                ):

                    if (
                        not track.standing_alerted
                    ):

                        details = (
                            f"{fmt(state_elapsed)} "
                            f"standing; "
                            f"knee_angle="
                            f"{knee_angle}"
                        )

                        capture_and_log(
                            frame,
                            track.id,
                            "STANDING_DWELL_THRESHOLD",
                            "standing",
                            details,
                            category="standing"
                        )

                        track.standing_alerted = True

                # =================================================
                # LIMITED VIEW EVENT
                # =================================================

                if (
                    display_state
                    == "limited_view"
                ):

                    if can_alert(
                        track,
                        "LIMITED_VIEW",
                        now,
                        15
                    ):

                        details = (
                            "Lower body is not "
                            "reliably visible; "
                            "posture classification "
                            "suppressed to avoid "
                            "false standing/sitting "
                            "classification."
                        )

                        capture_and_log(
                            frame,
                            track.id,
                            "LIMITED_VIEW",
                            "limited_view",
                            details,
                            category="limited_view"
                        )

                        mark_alert(
                            track,
                            "LIMITED_VIEW",
                            now
                        )

                # =================================================
                # ZONE DETECTION
                # =================================================

                in_zone = inside_zone(
                    track.center,
                    w,
                    h
                )

                # -------------------------------------------------
                # First zone observation
                # -------------------------------------------------

                if not hasattr(
                    track,
                    "zone_initialized"
                ):
                    track.zone_initialized = False

                if not hasattr(
                    track,
                    "was_in_zone"
                ):
                    track.was_in_zone = False

                # -------------------------------------------------
                # ENTER
                # -------------------------------------------------

                if (
                    in_zone
                    and
                    not track.was_in_zone
                ):

                    track.zone_started = now
                    track.zone_alerted = False

                    details = (
                        "Person entered "
                        "monitoring zone."
                    )

                    capture_and_log(
                        frame,
                        track.id,
                        "ZONE_ENTRY",
                        display_state,
                        details,
                        category="zone"
                    )

                    track.was_in_zone = True
                    track.zone_initialized = True

                # -------------------------------------------------
                # EXIT
                # -------------------------------------------------

                elif (
                    not in_zone
                    and
                    track.was_in_zone
                ):

                    zone_duration = 0

                    if (
                        track.zone_started
                        is not None
                    ):

                        zone_duration = (
                            now
                            -
                            track.zone_started
                        )

                    details = (
                        "Person exited "
                        "monitoring zone; "
                        f"duration="
                        f"{fmt(zone_duration)}"
                    )

                    capture_and_log(
                        frame,
                        track.id,
                        "ZONE_EXIT",
                        display_state,
                        details,
                        category="zone"
                    )

                    track.zone_started = None
                    track.zone_alerted = False
                    track.was_in_zone = False

                # -------------------------------------------------
                # DWELL
                # -------------------------------------------------

                zone_elapsed = 0

                if in_zone:

                    if (
                        track.zone_started
                        is None
                    ):
                        track.zone_started = now

                    zone_elapsed = (
                        now
                        -
                        track.zone_started
                    )

                    if (
                        zone_elapsed
                        >= ZONE_DWELL_SECONDS
                        and
                        not track.zone_alerted
                    ):

                        details = (
                            f"{fmt(zone_elapsed)} "
                            "inside monitoring zone."
                        )

                        capture_and_log(
                            frame,
                            track.id,
                            "ZONE_DWELL_THRESHOLD",
                            display_state,
                            details,
                            category="zone"
                        )

                        track.zone_alerted = True

                # =================================================
                # DRAW TRACK
                # =================================================

                x1, y1, x2, y2 = map(
                    int,
                    track.bbox
                )

                # -------------------------------------------------
                # Color based on state
                # -------------------------------------------------

                if display_state == "falling":

                    box_color = (
                        0,
                        0,
                        255
                    )

                elif display_state == "limited_view":

                    box_color = (
                        0,
                        165,
                        255
                    )

                elif display_state == "sitting":

                    box_color = (
                        255,
                        0,
                        255
                    )

                elif display_state == "standing":

                    box_color = (
                        0,
                        255,
                        0
                    )

                elif display_state == "walking":

                    box_color = (
                        255,
                        255,
                        0
                    )

                else:

                    box_color = (
                        255,
                        255,
                        255
                    )

                # -------------------------------------------------
                # Main label
                # -------------------------------------------------

                label = (
                    f"ID {track.id} | "
                    f"{display_state.upper()} | "
                    f"{fmt(state_elapsed)}"
                )

                if in_zone:

                    label += (
                        f" | Zone "
                        f"{fmt(zone_elapsed)}"
                    )

                cv2.rectangle(
                    frame,
                    (
                        x1,
                        y1
                    ),
                    (
                        x2,
                        y2
                    ),
                    box_color,
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (
                        x1,
                        max(
                            25,
                            y1 - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    box_color,
                    2
                )

                # -------------------------------------------------
                # Skeleton
                # -------------------------------------------------

                draw_skeleton(
                    frame,
                    pts,
                    box_color
                )

                # =================================================
                # LIMITED VIEW WARNING
                # =================================================

                if display_state == "limited_view":

                    cv2.putText(
                        frame,
                        "LIMITED VIEW - FULL BODY REQUIRED",
                        (
                            25,
                            40
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Lower body not visible - posture suppressed",
                        (
                            25,
                            70
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 165, 255),
                        2
                    )

                # =================================================
                # FALL WARNING
                # =================================================

                if display_state == "falling":

                    cv2.putText(
                        frame,
                        "POSSIBLE FALL - CONFIRMING",
                        (
                            25,
                            40
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        3
                    )

            # ====================================================
            # CAMERA CONTROLS
            # ====================================================

            cv2.putText(
                frame,
                "C = capture | Q = quit",
                (
                    20,
                    h - 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            # ====================================================
            # DISPLAY
            # ====================================================

            cv2.imshow(
                "AI Camera Tracker - POC",
                frame
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            # ----------------------------------------------------
            # MANUAL CAPTURE
            # ----------------------------------------------------

            if key == ord("c"):

                # If a person is tracked,
                # associate manual capture with them.
                if tracks:

                    active_tracks = [
                        t
                        for t in tracks
                        if not t.missed
                    ]

                    if active_tracks:

                        selected = (
                            active_tracks[0]
                        )

                        capture_and_log(
                            frame,
                            selected.id,
                            "MANUAL_CAPTURE",
                            selected.state,
                            "Manual image capture requested.",
                            category="manual"
                        )

                        print(
                            "[INFO] Manual capture saved."
                        )

                    else:

                        print(
                            "[INFO] No active person "
                            "to associate capture with."
                        )

                else:

                    # Person ID 0 means camera-level
                    # manual capture.
                    capture_and_log(
                        frame,
                        0,
                        "MANUAL_CAPTURE",
                        "unknown",
                        "Manual image capture; no person detected.",
                        category="manual"
                    )

                    print(
                        "[INFO] Manual capture saved."
                    )

            # ----------------------------------------------------
            # QUIT
            # ----------------------------------------------------

            elif key == ord("q"):

                print(
                    "[INFO] Stopping camera..."
                )

                break

    finally:

        cap.release()

        detector.close()

        cv2.destroyAllWindows()

        print(
            "[INFO] Camera tracker stopped."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()