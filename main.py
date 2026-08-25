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
# CONFIG
# ============================================================

def cfg(name, default):
    return getattr(config, name, default)


CAMERA_INDEX = cfg("CAMERA_INDEX", 0)

FRAME_WIDTH = cfg(
    "FRAME_WIDTH",
    1280
)

FRAME_HEIGHT = cfg(
    "FRAME_HEIGHT",
    720
)

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
# POSTURE SETTINGS
# ============================================================

SITTING_KNEE_ANGLE = 155
STRONG_SITTING_KNEE_ANGLE = 140

STANDING_KNEE_ANGLE = 168

UPRIGHT_TORSO_ANGLE = 32

FALL_TORSO_ANGLE = 55

WALKING_MOVEMENT_THRESHOLD = 8

POSTURE_CONFIRM_FRAMES = 5

POSTURE_CONFIRM_SECONDS = 0.7

MIN_POSTURE_CONFIDENCE = 0.62

UNCERTAIN_CONFIDENCE = 0.45

POSTURE_HISTORY_SIZE = 9


# ============================================================
# MEDIAPIPE LANDMARK INDICES
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

def ensure_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Pose model not found.\n"
            "Run:\n"
            "python download_model.py"
        )


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

    categories = [
        "events",
        "fall",
        "falls",
        "manual",
        "zone",
        "posture",
        "limited_view",
        "dwell"
    ]

    for category in categories:

        (
            CAPTURE_DIR / category
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    if not EVENT_LOG.exists():

        with EVENT_LOG.open(
            "w",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "timestamp",
                "track_id",
                "event",
                "activity",
                "details",
                "image_path"
            ])


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


def timestamp():

    return datetime.now().isoformat(
        timespec="seconds"
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


def clamp(
    value,
    low=0.0,
    high=1.0
):

    return max(
        low,
        min(
            high,
            value
        )
    )


# ============================================================
# LANDMARK VALIDATION
# ============================================================

def valid_point(point):

    if point is None:
        return False

    try:

        if len(point) < 2:
            return False

        x = float(point[0])
        y = float(point[1])

        return (
            math.isfinite(x)
            and
            math.isfinite(y)
            and
            x >= 0
            and
            y >= 0
        )

    except (
        TypeError,
        ValueError
    ):

        return False


# ============================================================
# ANGLE
# ============================================================

def angle(a, b, c):

    if (
        not valid_point(a)
        or
        not valid_point(b)
        or
        not valid_point(c)
    ):

        return None

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

    denominator = (
        np.linalg.norm(ba)
        *
        np.linalg.norm(bc)
    )

    if denominator == 0:

        return None

    cosine = (
        np.dot(ba, bc)
        /
        denominator
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
# BODY VISIBILITY
# ============================================================

def lower_body_visibility(
    points,
    frame_height,
    bbox
):
    """
    Determine whether enough lower-body information
    is actually visible.

    Important:
    MediaPipe can sometimes estimate lower-body
    landmarks even when the camera cannot see them.

    Therefore we combine:

        1. Landmark validity
        2. Landmark position
        3. Bounding-box bottom
        4. Frame boundaries

    If the camera mainly sees the upper body,
    return False instead of guessing STANDING.
    """

    if len(points) < 33:

        return False

    knees = [
        points[LEFT_KNEE],
        points[RIGHT_KNEE]
    ]

    ankles = [
        points[LEFT_ANKLE],
        points[RIGHT_ANKLE]
    ]

    valid_knees = [
        p for p in knees
        if valid_point(p)
    ]

    valid_ankles = [
        p for p in ankles
        if valid_point(p)
    ]

    # Need at least one knee.
    if len(valid_knees) == 0:

        return False

    # --------------------------------------------------------
    # Frame-bottom test
    # --------------------------------------------------------

    bottom_limit = (
        frame_height * 0.94
    )

    knees_at_bottom = all(
        p[1] >= bottom_limit
        for p in valid_knees
    )

    ankles_at_bottom = (
        len(valid_ankles) > 0
        and
        all(
            p[1] >= bottom_limit
            for p in valid_ankles
        )
    )

    # Both knees are effectively cut off.
    if knees_at_bottom:

        return False

    # Ankle estimates are all outside the useful frame.
    if ankles_at_bottom:

        return False

    # --------------------------------------------------------
    # Bounding-box test
    # --------------------------------------------------------

    if bbox is not None:

        try:

            x1, y1, x2, y2 = map(
                float,
                bbox
            )

            bbox_bottom = y2

            # If the person's detected body extends
            # to the camera bottom and ankle information
            # is unavailable, treat as limited.
            if (
                bbox_bottom
                >=
                frame_height * 0.97
                and
                len(valid_ankles) == 0
            ):

                return False

        except Exception:
            pass

    # Require at least one ankle when possible.
    # This prevents upper-body-only frames from
    # becoming STANDING.
    if len(valid_ankles) == 0:

        return False

    return True


# ============================================================
# BODY FEATURES
# ============================================================

def get_body_features(
    points,
    frame_height,
    bbox
):

    if len(points) < 33:

        return {
            "valid": False,
            "upper_visible": False,
            "lower_body_visible": False,
            "torso_angle": None,
            "knee_angle": None,
            "hip_angle": None,
            "body_quality": 0.0,
            "leg_torso_ratio": 0.0
        }

    shoulders_valid = (
        valid_point(
            points[LEFT_SHOULDER]
        )
        and
        valid_point(
            points[RIGHT_SHOULDER]
        )
    )

    hips_valid = (
        valid_point(
            points[LEFT_HIP]
        )
        and
        valid_point(
            points[RIGHT_HIP]
        )
    )

    upper_visible = (
        shoulders_valid
        and
        hips_valid
    )

    if not upper_visible:

        return {
            "valid": False,
            "upper_visible": False,
            "lower_body_visible": False,
            "torso_angle": None,
            "knee_angle": None,
            "hip_angle": None,
            "body_quality": 0.0,
            "leg_torso_ratio": 0.0
        }

    shoulder = midpoint(
        points[LEFT_SHOULDER],
        points[RIGHT_SHOULDER]
    )

    hip = midpoint(
        points[LEFT_HIP],
        points[RIGHT_HIP]
    )

    # --------------------------------------------------------
    # Torso angle
    # --------------------------------------------------------

    torso_angle = math.degrees(
        math.atan2(
            abs(
                hip[0] - shoulder[0]
            ),
            abs(
                hip[1] - shoulder[1]
            ) + 1e-6
        )
    )

    # --------------------------------------------------------
    # Knee angles
    # --------------------------------------------------------

    left_knee_angle = angle(
        points[LEFT_HIP],
        points[LEFT_KNEE],
        points[LEFT_ANKLE]
    )

    right_knee_angle = angle(
        points[RIGHT_HIP],
        points[RIGHT_KNEE],
        points[RIGHT_ANKLE]
    )

    knee_angles = [
        x for x in (
            left_knee_angle,
            right_knee_angle
        )
        if x is not None
    ]

    knee_angle = (
        sum(knee_angles)
        /
        len(knee_angles)
        if knee_angles
        else None
    )

    # --------------------------------------------------------
    # Hip angles
    # --------------------------------------------------------

    left_hip_angle = angle(
        points[LEFT_SHOULDER],
        points[LEFT_HIP],
        points[LEFT_KNEE]
    )

    right_hip_angle = angle(
        points[RIGHT_SHOULDER],
        points[RIGHT_HIP],
        points[RIGHT_KNEE]
    )

    hip_angles = [
        x for x in (
            left_hip_angle,
            right_hip_angle
        )
        if x is not None
    ]

    hip_angle = (
        sum(hip_angles)
        /
        len(hip_angles)
        if hip_angles
        else None
    )

    # --------------------------------------------------------
    # Lower body visibility
    # --------------------------------------------------------

    lower_visible = lower_body_visibility(
        points,
        frame_height,
        bbox
    )

    # --------------------------------------------------------
    # Leg / torso ratio
    # --------------------------------------------------------

    leg_lengths = []

    left_leg_valid = (
        valid_point(
            points[LEFT_HIP]
        )
        and
        valid_point(
            points[LEFT_KNEE]
        )
        and
        valid_point(
            points[LEFT_ANKLE]
        )
    )

    if left_leg_valid:

        leg_lengths.append(
            distance(
                points[LEFT_HIP],
                points[LEFT_KNEE]
            )
            +
            distance(
                points[LEFT_KNEE],
                points[LEFT_ANKLE]
            )
        )

    right_leg_valid = (
        valid_point(
            points[RIGHT_HIP]
        )
        and
        valid_point(
            points[RIGHT_KNEE]
        )
        and
        valid_point(
            points[RIGHT_ANKLE]
        )
    )

    if right_leg_valid:

        leg_lengths.append(
            distance(
                points[RIGHT_HIP],
                points[RIGHT_KNEE]
            )
            +
            distance(
                points[RIGHT_KNEE],
                points[RIGHT_ANKLE]
            )
        )

    leg_length = (
        max(leg_lengths)
        if leg_lengths
        else 0.0
    )

    torso_length = distance(
        shoulder,
        hip
    )

    leg_torso_ratio = 0.0

    if torso_length > 0:

        leg_torso_ratio = (
            leg_length
            /
            torso_length
        )

    # --------------------------------------------------------
    # Body quality
    # --------------------------------------------------------

    upper_points = [
        points[LEFT_SHOULDER],
        points[RIGHT_SHOULDER],
        points[LEFT_HIP],
        points[RIGHT_HIP]
    ]

    lower_points = [
        points[LEFT_KNEE],
        points[RIGHT_KNEE],
        points[LEFT_ANKLE],
        points[RIGHT_ANKLE]
    ]

    upper_score = (
        sum(
            valid_point(p)
            for p in upper_points
        )
        /
        len(upper_points)
    )

    lower_score = (
        sum(
            valid_point(p)
            for p in lower_points
        )
        /
        len(lower_points)
    )

    body_quality = (
        0.45 * upper_score
        +
        0.55 * lower_score
    )

    return {
        "valid": True,
        "upper_visible": upper_visible,
        "lower_body_visible": lower_visible,
        "torso_angle": torso_angle,
        "knee_angle": knee_angle,
        "hip_angle": hip_angle,
        "body_quality": clamp(body_quality),
        "leg_torso_ratio": leg_torso_ratio
    }


# ============================================================
# POSTURE CLASSIFICATION
# ============================================================

def classify_posture(
    features,
    track
):

    if not features["valid"]:

        return (
            "limited_view",
            0.0,
            "insufficient body landmarks"
        )

    # --------------------------------------------------------
    # CRITICAL:
    # Never classify upper-body-only detection as standing.
    # --------------------------------------------------------

    if not features["lower_body_visible"]:

        return (
            "limited_view",
            features["body_quality"],
            "lower body not reliably visible"
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

    body_quality = features[
        "body_quality"
    ]

    leg_torso_ratio = features[
        "leg_torso_ratio"
    ]

    if knee_angle is None:

        return (
            "limited_view",
            body_quality,
            "knee landmarks unavailable"
        )

    # --------------------------------------------------------
    # FALLING
    # --------------------------------------------------------

    if torso_angle >= FALL_TORSO_ANGLE:

        confidence = clamp(
            0.65
            +
            (
                torso_angle
                -
                FALL_TORSO_ANGLE
            )
            /
            40.0
            *
            0.35
        )

        confidence *= (
            0.6
            +
            0.4 * body_quality
        )

        return (
            "falling",
            confidence,
            "horizontal torso orientation"
        )

    # --------------------------------------------------------
    # SITTING SCORE
    # --------------------------------------------------------

    sitting_score = 0.0

    if knee_angle < STRONG_SITTING_KNEE_ANGLE:

        sitting_score += 0.50

    elif knee_angle < SITTING_KNEE_ANGLE:

        sitting_score += 0.35

    if hip_angle is not None:

        if hip_angle < 115:

            sitting_score += 0.25

        elif hip_angle < 135:

            sitting_score += 0.15

    if leg_torso_ratio > 0:

        if leg_torso_ratio < 0.85:

            sitting_score += 0.20

        elif leg_torso_ratio < 1.05:

            sitting_score += 0.10

    sitting_confidence = clamp(
        sitting_score
        +
        0.15 * body_quality
    )

    # --------------------------------------------------------
    # STANDING SCORE
    # --------------------------------------------------------

    standing_score = 0.0

    if knee_angle >= STANDING_KNEE_ANGLE:

        standing_score += 0.50

    elif knee_angle >= 155:

        standing_score += 0.20

    if torso_angle <= UPRIGHT_TORSO_ANGLE:

        standing_score += 0.30

    if leg_torso_ratio >= 1.0:

        standing_score += 0.20

    standing_confidence = clamp(
        standing_score
        +
        0.15 * body_quality
    )

    # --------------------------------------------------------
    # WALKING
    # --------------------------------------------------------

    movement = getattr(
        track,
        "body_movement",
        0.0
    )

    walking_confidence = 0.0

    if movement >= WALKING_MOVEMENT_THRESHOLD:

        walking_confidence = clamp(
            0.55
            +
            min(
                0.35,
                movement / 40.0
            )
        )

    # --------------------------------------------------------
    # Choose best state
    # --------------------------------------------------------

    scores = {
        "sitting": sitting_confidence,
        "standing": standing_confidence,
        "walking": walking_confidence
    }

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    state, confidence = ranked[0]

    second_confidence = ranked[1][1]

    # --------------------------------------------------------
    # Ambiguous classification
    # --------------------------------------------------------

    if (
        confidence < MIN_POSTURE_CONFIDENCE
        or
        (
            confidence - second_confidence
            < 0.10
            and
            confidence < 0.75
        )
    ):

        return (
            "uncertain",
            confidence,
            (
                f"sitting={sitting_confidence:.2f}; "
                f"standing={standing_confidence:.2f}; "
                f"walking={walking_confidence:.2f}"
            )
        )

    return (
        state,
        confidence,
        (
            f"sitting={sitting_confidence:.2f}; "
            f"standing={standing_confidence:.2f}; "
            f"walking={walking_confidence:.2f}"
        )
    )


# ============================================================
# POSTURE SMOOTHING
# ============================================================

def stabilize_posture(
    track,
    detected_state,
    confidence,
    now
):

    if not hasattr(
        track,
        "candidate_state"
    ):

        track.candidate_state = None

    if not hasattr(
        track,
        "candidate_started"
    ):

        track.candidate_started = None

    if not hasattr(
        track,
        "posture_history"
    ):

        track.posture_history = deque(
            maxlen=POSTURE_HISTORY_SIZE
        )

    # --------------------------------------------------------
    # Limited view must be immediate.
    # --------------------------------------------------------

    if detected_state == "limited_view":

        track.candidate_state = None
        track.candidate_started = None

        track.posture_history.clear()

        return "limited_view"

    # --------------------------------------------------------
    # Fall should be responsive.
    # --------------------------------------------------------

    if detected_state == "falling":

        track.candidate_state = None
        track.candidate_started = None

        return "falling"

    # --------------------------------------------------------
    # Uncertain
    # --------------------------------------------------------

    if detected_state == "uncertain":

        if track.state not in (
            "unknown",
            "limited_view",
            "uncertain"
        ):

            return track.state

        return "uncertain"

    # --------------------------------------------------------
    # Same state
    # --------------------------------------------------------

    if detected_state == track.state:

        track.candidate_state = None
        track.candidate_started = None

        track.posture_history.clear()

        return track.state

    # --------------------------------------------------------
    # New candidate
    # --------------------------------------------------------

    if (
        track.candidate_state
        != detected_state
    ):

        track.candidate_state = (
            detected_state
        )

        track.candidate_started = now

        track.posture_history.clear()

    track.posture_history.append(
        detected_state
    )

    recent = list(
        track.posture_history
    )[
        -POSTURE_CONFIRM_FRAMES:
    ]

    frame_confirmed = (
        len(recent)
        >= POSTURE_CONFIRM_FRAMES
        and
        all(
            state == detected_state
            for state in recent
        )
    )

    time_confirmed = (
        track.candidate_started
        is not None
        and
        now
        -
        track.candidate_started
        >= POSTURE_CONFIRM_SECONDS
    )

    if (
        frame_confirmed
        and
        time_confirmed
    ):

        track.candidate_state = None
        track.candidate_started = None

        track.posture_history.clear()

        return detected_state

    if track.state != "unknown":

        return track.state

    return detected_state


# ============================================================
# ZONE
# ============================================================

def inside_zone(
    center,
    width,
    height
):

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


def draw_zone(
    frame
):

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

    p1 = (
        int(x1 * w),
        int(y1 * h)
    )

    p2 = (
        int(x2 * w),
        int(y2 * h)
    )

    cv2.rectangle(
        frame,
        p1,
        p2,
        (255, 180, 0),
        2
    )

    cv2.putText(
        frame,
        "MONITORING ZONE",
        (
            p1[0],
            max(
                25,
                p1[1] - 8
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 180, 0),
        2
    )


# ============================================================
# CSV LOGGING
# ============================================================

def log_csv_event(
    track_id,
    event,
    activity,
    details,
    image_path
):

    stamp = timestamp()

    with EVENT_LOG.open(
        "a",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            stamp,
            track_id,
            event,
            activity,
            details,
            image_path
        ])

    print(
        f"[ALERT] {stamp} | "
        f"Person {track_id} | "
        f"{event} | "
        f"{details}"
    )


# ============================================================
# EVENT PIPELINE
# ============================================================

def capture_and_log(
    frame,
    track_id,
    event_type,
    activity,
    details,
    category="events"
):

    try:

        event_id, image_path = (
            capture_event(
                frame=frame,
                person_id=track_id,
                event_type=event_type,
                activity=activity,
                details=details,
                category=category
            )
        )

        log_csv_event(
            track_id,
            event_type,
            activity,
            details,
            image_path
        )

        return (
            event_id,
            image_path
        )

    except Exception as exc:

        print(
            f"[ERROR] Could not capture "
            f"{event_type} for Person "
            f"{track_id}: {exc}"
        )

        return (
            None,
            None
        )


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

    previous = (
        track.event_alert_times.get(
            event_name
        )
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
# SKELETON
# ============================================================

def draw_skeleton(
    frame,
    points,
    color
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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AI CAMERA TRACKER")
    print("Smart Posture + Limited View + Event Monitoring")
    print("=" * 70)

    ensure_model()
    ensure_dirs()

    init_db()

    print(
        "[INFO] SQLite database initialized."
    )

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

    print(
        "[INFO] Camera started."
    )

    print(
        "[INFO] Press C to capture manually."
    )

    print(
        "[INFO] Press Q to quit."
    )

    timestamp_ms = 0

    previous_time = time.monotonic()

    try:

        while True:

            ok, frame = cap.read()

            if not ok:

                print(
                    "[ERROR] Camera frame "
                    "could not be read."
                )

                break

            now = time.monotonic()

            elapsed = (
                now
                -
                previous_time
            )

            previous_time = now

            timestamp_ms += max(
                1,
                int(
                    elapsed * 1000
                )
            )

            h, w = frame.shape[:2]

            # =================================================
            # POSE DETECTION
            # =================================================

            detections = detector.detect(
                frame,
                timestamp_ms
            )

            # =================================================
            # TRACKING
            # =================================================

            tracks = tracker.update(
                detections,
                now
            )

            # =================================================
            # ZONE
            # =================================================

            draw_zone(frame)

            # =================================================
            # PROCESS PEOPLE
            # =================================================

            for track in tracks:

                if track.missed:
                    continue

                points = track.landmarks

                if len(points) < 33:
                    continue

                # ------------------------------------------------
                # Body features
                # ------------------------------------------------

                features = get_body_features(
                    points,
                    h,
                    track.bbox
                )

                # ------------------------------------------------
                # Movement
                # ------------------------------------------------

                previous_center = getattr(
                    track,
                    "previous_center",
                    None
                )

                if previous_center is None:

                    movement = 0.0

                else:

                    movement = distance(
                        track.center,
                        previous_center
                    )

                track.previous_center = (
                    track.center
                )

                track.body_movement = movement

                # ------------------------------------------------
                # Classification
                # ------------------------------------------------

                detected_state, confidence, reason = (
                    classify_posture(
                        features,
                        track
                    )
                )

                previous_state = (
                    track.state
                )

                state = stabilize_posture(
                    track,
                    detected_state,
                    confidence,
                    now
                )

                # ------------------------------------------------
                # State change
                # ------------------------------------------------

                if state != previous_state:

                    track.previous_state = (
                        previous_state
                    )

                    track.state = state

                    track.state_started = now

                    track.sitting_alerted = False
                    track.standing_alerted = False

                    # =========================================
                    # POSTURE CHANGE
                    # =========================================

                    if state in (
                        "sitting",
                        "standing",
                        "walking",
                        "falling"
                    ):

                        if can_alert(
                            track,
                            f"STATE_{state}",
                            now
                        ):

                            details = (
                                f"previous="
                                f"{previous_state}; "
                                f"confidence="
                                f"{confidence:.2f}; "
                                f"torso_angle="
                                f"{features['torso_angle']}; "
                                f"knee_angle="
                                f"{features['knee_angle']}; "
                                f"hip_angle="
                                f"{features['hip_angle']}; "
                                f"reason={reason}"
                            )

                            category = (
                                "fall"
                                if state == "falling"
                                else "posture"
                            )

                            capture_and_log(
                                frame,
                                track.id,
                                "POSTURE_CHANGE",
                                state,
                                details,
                                category
                            )

                            mark_alert(
                                track,
                                f"STATE_{state}",
                                now
                            )

                    # =========================================
                    # LIMITED VIEW
                    # =========================================

                    elif state == "limited_view":

                        if can_alert(
                            track,
                            "LIMITED_VIEW",
                            now,
                            5
                        ):

                            details = (
                                "Lower body not "
                                "reliably visible. "
                                "Posture classification "
                                "suppressed to prevent "
                                "false standing/sitting."
                            )

                            capture_and_log(
                                frame,
                                track.id,
                                "LIMITED_VIEW",
                                "limited_view",
                                details,
                                "limited_view"
                            )

                            mark_alert(
                                track,
                                "LIMITED_VIEW",
                                now
                            )

                # =================================================
                # FALL DETECTION
                # =================================================

                if (
                    state != "limited_view"
                    and
                    features["lower_body_visible"]
                ):

                    if (
                        valid_point(
                            points[LEFT_HIP]
                        )
                        and
                        valid_point(
                            points[RIGHT_HIP]
                        )
                    ):

                        hip = midpoint(
                            points[LEFT_HIP],
                            points[RIGHT_HIP]
                        )

                        hip_y = (
                            hip[1]
                            /
                            h
                        )

                        previous_hip_y = getattr(
                            track,
                            "previous_hip_y",
                            None
                        )

                        if previous_hip_y is None:

                            hip_drop = 0.0

                        else:

                            hip_drop = (
                                hip_y
                                -
                                previous_hip_y
                            )

                        track.previous_hip_y = hip_y

                        # Sudden downward movement.
                        if (
                            hip_drop
                            >= FALL_DROP_RATIO
                        ):

                            if (
                                track.fall_candidate_started
                                is None
                            ):

                                track.fall_candidate_started = (
                                    now
                                )

                            fall_duration = (
                                now
                                -
                                track.fall_candidate_started
                            )

                            torso_horizontal = (
                                features["torso_angle"]
                                >= FALL_TORSO_ANGLE
                            )

                            if (
                                fall_duration
                                >= FALL_CONFIRM_SECONDS
                                and
                                torso_horizontal
                                and
                                can_alert(
                                    track,
                                    "FALL",
                                    now,
                                    15
                                )
                            ):

                                track.state = "falling"

                                details = (
                                    f"hip_drop="
                                    f"{hip_drop:.3f}; "
                                    f"torso_angle="
                                    f"{features['torso_angle']:.1f}; "
                                    f"confirmation="
                                    f"{fall_duration:.2f}s"
                                )

                                capture_and_log(
                                    frame,
                                    track.id,
                                    "FALL_DETECTED",
                                    "falling",
                                    details,
                                    "fall"
                                )

                                mark_alert(
                                    track,
                                    "FALL",
                                    now
                                )

                                track.fall_alerted_at = now

                        else:

                            track.fall_candidate_started = None

                # =================================================
                # STATE TIMER
                # =================================================

                state_elapsed = (
                    now
                    -
                    track.state_started
                )

                # =================================================
                # SITTING DWELL
                # =================================================

                if (
                    state == "sitting"
                    and
                    state_elapsed
                    >= SITTING_DWELL_SECONDS
                    and
                    not track.sitting_alerted
                ):

                    details = (
                        f"Sitting for "
                        f"{fmt(state_elapsed)}."
                    )

                    capture_and_log(
                        frame,
                        track.id,
                        "SITTING_DWELL_THRESHOLD",
                        state,
                        details,
                        "dwell"
                    )

                    track.sitting_alerted = True

                # =================================================
                # STANDING DWELL
                # =================================================

                if (
                    state == "standing"
                    and
                    state_elapsed
                    >= STANDING_DWELL_SECONDS
                    and
                    not track.standing_alerted
                ):

                    details = (
                        f"Standing for "
                        f"{fmt(state_elapsed)}."
                    )

                    capture_and_log(
                        frame,
                        track.id,
                        "STANDING_DWELL_THRESHOLD",
                        state,
                        details,
                        "dwell"
                    )

                    track.standing_alerted = True

                # =================================================
                # MONITORING ZONE
                # =================================================

                in_zone = inside_zone(
                    track.center,
                    w,
                    h
                )

                if not hasattr(
                    track,
                    "was_in_zone"
                ):

                    track.was_in_zone = False

                if not hasattr(
                    track,
                    "zone_initialized"
                ):

                    track.zone_initialized = False

                # ------------------------------------------------
                # Zone entry
                # ------------------------------------------------

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
                        state,
                        details,
                        "zone"
                    )

                    track.was_in_zone = True
                    track.zone_initialized = True

                # ------------------------------------------------
                # Zone exit
                # ------------------------------------------------

                elif (
                    not in_zone
                    and
                    track.was_in_zone
                ):

                    zone_duration = 0.0

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
                        state,
                        details,
                        "zone"
                    )

                    track.zone_started = None

                    track.zone_alerted = False

                    track.was_in_zone = False

                # ------------------------------------------------
                # Zone dwell
                # ------------------------------------------------

                zone_elapsed = 0.0

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
                            f"Inside monitoring "
                            f"zone for "
                            f"{fmt(zone_elapsed)}."
                        )

                        capture_and_log(
                            frame,
                            track.id,
                            "ZONE_DWELL_THRESHOLD",
                            state,
                            details,
                            "zone"
                        )

                        track.zone_alerted = True

                # =================================================
                # DRAW PERSON
                # =================================================

                x1, y1, x2, y2 = map(
                    int,
                    track.bbox
                )

                display_state = state

                # ------------------------------------------------
                # Color
                # ------------------------------------------------

                if state == "falling":

                    box_color = (
                        0,
                        0,
                        255
                    )

                elif state == "limited_view":

                    box_color = (
                        0,
                        165,
                        255
                    )

                elif state == "sitting":

                    box_color = (
                        255,
                        0,
                        255
                    )

                elif state == "standing":

                    box_color = (
                        0,
                        255,
                        0
                    )

                elif state == "walking":

                    box_color = (
                        255,
                        255,
                        0
                    )

                elif state == "uncertain":

                    box_color = (
                        0,
                        200,
                        255
                    )

                else:

                    box_color = (
                        255,
                        255,
                        255
                    )

                # ------------------------------------------------
                # Confidence
                # ------------------------------------------------

                if state in (
                    "limited_view",
                    "uncertain"
                ):

                    confidence_text = "--"

                else:

                    confidence_text = (
                        f"{confidence * 100:.0f}%"
                    )

                # ------------------------------------------------
                # Main label
                # ------------------------------------------------

                label = (
                    f"ID {track.id} | "
                    f"{display_state.upper()} | "
                    f"{fmt(state_elapsed)} | "
                    f"Conf {confidence_text}"
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
                    0.52,
                    box_color,
                    2
                )

                # ------------------------------------------------
                # Skeleton
                # ------------------------------------------------

                draw_skeleton(
                    frame,
                    points,
                    box_color
                )

                # =================================================
                # LIMITED VIEW WARNING
                # =================================================

                if state == "limited_view":

                    cv2.putText(
                        frame,
                        "LIMITED VIEW",
                        (
                            25,
                            40
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 165, 255),
                        3
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

                if state == "falling":

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

            # =================================================
            # CONTROLS
            # =================================================

            cv2.putText(
                frame,
                "C = capture image | Q = quit",
                (
                    20,
                    h - 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            # =================================================
            # DISPLAY
            # =================================================

            cv2.imshow(
                "AI Camera Tracker - Smart Posture",
                frame
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            # =================================================
            # MANUAL CAPTURE
            # =================================================

            if key == ord("c"):

                active_tracks = [
                    track
                    for track in tracks
                    if not track.missed
                ]

                if active_tracks:

                    selected = active_tracks[0]

                    capture_and_log(
                        frame,
                        selected.id,
                        "MANUAL_CAPTURE",
                        selected.state,
                        "Manual image capture requested.",
                        "manual"
                    )

                else:

                    capture_and_log(
                        frame,
                        0,
                        "MANUAL_CAPTURE",
                        "unknown",
                        "Manual image capture; no person detected.",
                        "manual"
                    )

            # =================================================
            # QUIT
            # =================================================

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