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
# CONFIG HELPER
# ============================================================

def cfg(name, default):
    return getattr(config, name, default)


CAMERA_INDEX = cfg(
    "CAMERA_INDEX",
    0
)

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
    0.08
)

FALL_CONFIRM_SECONDS = cfg(
    "FALL_CONFIRM_SECONDS",
    1.2
)

ALERT_COOLDOWN_SECONDS = cfg(
    "ALERT_COOLDOWN_SECONDS",
    10.0
)

SITTING_DWELL_SECONDS = cfg(
    "SITTING_DWELL_SECONDS",
    120.0
)

STANDING_DWELL_SECONDS = cfg(
    "STANDING_DWELL_SECONDS",
    300.0
)

ZONE_DWELL_SECONDS = cfg(
    "ZONE_DWELL_SECONDS",
    120.0
)


# ============================================================
# POSTURE SETTINGS
# ============================================================

# Knee angle thresholds.
SITTING_KNEE_ANGLE = 155.0
STANDING_KNEE_ANGLE = 168.0

# Torso angle from vertical.
UPRIGHT_TORSO_ANGLE = 32.0

# Fall torso threshold.
FALL_TORSO_ANGLE = cfg(
    "FALL_TORSO_ANGLE",
    55.0
)

# Movement threshold.
WALKING_MOVEMENT_THRESHOLD = 8.0

# Temporal smoothing.
POSTURE_CONFIRM_FRAMES = 5
POSTURE_CONFIRM_SECONDS = 0.7
POSTURE_HISTORY_SIZE = 9

# Confidence.
MIN_POSTURE_CONFIDENCE = 0.62


# ============================================================
# MEDIAPIPE LANDMARK INDICES
# ============================================================

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_ELBOW = 13
RIGHT_ELBOW = 14

LEFT_WRIST = 15
RIGHT_WRIST = 16

LEFT_HIP = 23
RIGHT_HIP = 24

LEFT_KNEE = 25
RIGHT_KNEE = 26

LEFT_ANKLE = 27
RIGHT_ANKLE = 28


# ============================================================
# SETUP
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
        "posture",
        "limited_view",
        "zone",
        "fall",
        "manual"
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
# TIME
# ============================================================

def timestamp():

    return datetime.now().isoformat(
        timespec="seconds"
    )


def fmt(seconds):

    seconds = max(
        0,
        int(seconds)
    )

    hours = seconds // 3600
    minutes = (
        seconds % 3600
    ) // 60
    secs = seconds % 60

    if hours:

        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{secs:02d}"
        )

    return (
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


# ============================================================
# LANDMARK VALIDATION
# ============================================================

def landmark_valid(point):

    if point is None:
        return False

    try:

        if len(point) < 2:
            return False

        x = float(point[0])
        y = float(point[1])

        if not math.isfinite(x):
            return False

        if not math.isfinite(y):
            return False

        if x < 0 or y < 0:
            return False

        return True

    except (
        TypeError,
        ValueError
    ):

        return False


def point_inside_frame(
    point,
    width,
    height,
    margin=0
):

    if not landmark_valid(point):
        return False

    x = float(point[0])
    y = float(point[1])

    return (
        margin <= x <= width - margin
        and
        margin <= y <= height - margin
    )


def point_near_bottom(
    point,
    frame_height,
    ratio=0.94
):

    if not landmark_valid(point):
        return True

    return (
        float(point[1])
        >=
        frame_height * ratio
    )


# ============================================================
# BODY VISIBILITY
# ============================================================

def check_upper_body_visibility(points):

    required = [
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_HIP,
        RIGHT_HIP
    ]

    return all(
        landmark_valid(points[i])
        for i in required
    )


def check_knees_visible(
    points,
    frame_width,
    frame_height
):
    """
    IMPORTANT:

    This is the main LIMITED VIEW gate.

    Standing/sitting classification is NOT allowed unless
    both knees are genuinely available inside the frame.

    Ankles are NOT required.

    This means a person visible only from the waist/chest
    down to the thighs will be classified as LIMITED VIEW,
    not STANDING.
    """

    if len(points) < 33:
        return False

    left_knee = points[LEFT_KNEE]
    right_knee = points[RIGHT_KNEE]

    # Both knees must have usable coordinates.
    if not landmark_valid(left_knee):
        return False

    if not landmark_valid(right_knee):
        return False

    # Both knees must actually be inside the camera frame.
    if not point_inside_frame(
        left_knee,
        frame_width,
        frame_height
    ):
        return False

    if not point_inside_frame(
        right_knee,
        frame_width,
        frame_height
    ):
        return False

    # If both knees are sitting directly at the bottom
    # edge, treat them as clipped by the camera.
    if (
        point_near_bottom(
            left_knee,
            frame_height
        )
        and
        point_near_bottom(
            right_knee,
            frame_height
        )
    ):

        return False

    return True


def lower_body_visibility(
    points,
    frame_width,
    frame_height
):

    if not check_upper_body_visibility(
        points
    ):

        return False

    return check_knees_visible(
        points,
        frame_width,
        frame_height
    )


# ============================================================
# GEOMETRY
# ============================================================

def midpoint(a, b):

    return (
        (
            float(a[0])
            +
            float(b[0])
        ) / 2.0,

        (
            float(a[1])
            +
            float(b[1])
        ) / 2.0
    )


def distance(a, b):

    return math.hypot(
        float(a[0]) - float(b[0]),
        float(a[1]) - float(b[1])
    )


def angle(a, b, c):

    if not all(
        landmark_valid(point)
        for point in (
            a,
            b,
            c
        )
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

    denom = (
        np.linalg.norm(ba)
        *
        np.linalg.norm(bc)
    )

    if denom == 0:

        return None

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


def torso_angle(
    shoulder,
    hip
):

    if not (
        landmark_valid(shoulder)
        and
        landmark_valid(hip)
    ):

        return None

    dx = (
        float(hip[0])
        -
        float(shoulder[0])
    )

    dy = (
        float(hip[1])
        -
        float(shoulder[1])
    )

    # Angle relative to the vertical axis.
    return abs(
        math.degrees(
            math.atan2(
                dx,
                dy
            )
        )
    )


def thigh_angle_from_vertical(
    hip,
    knee
):

    if not (
        landmark_valid(hip)
        and
        landmark_valid(knee)
    ):

        return None

    dx = (
        float(knee[0])
        -
        float(hip[0])
    )

    dy = (
        float(knee[1])
        -
        float(hip[1])
    )

    return abs(
        math.degrees(
            math.atan2(
                dx,
                dy
            )
        )
    )


# ============================================================
# BODY FEATURES
# ============================================================

def get_body_features(
    points,
    frame_width,
    frame_height
):

    features = {
        "valid": False,
        "lower_body_visible": False,

        "torso_angle": None,
        "knee_angle": None,
        "left_knee_angle": None,
        "right_knee_angle": None,

        "hip_angle": None,

        "left_thigh_angle": None,
        "right_thigh_angle": None,

        "body_quality": 0.0,

        "hip": None
    }

    if len(points) < 33:
        return features

    upper_visible = (
        check_upper_body_visibility(
            points
        )
    )

    if not upper_visible:
        return features

    # --------------------------------------------------------
    # Core body points
    # --------------------------------------------------------

    shoulder = midpoint(
        points[LEFT_SHOULDER],
        points[RIGHT_SHOULDER]
    )

    hip = midpoint(
        points[LEFT_HIP],
        points[RIGHT_HIP]
    )

    features["hip"] = hip

    features["torso_angle"] = (
        torso_angle(
            shoulder,
            hip
        )
    )

    # --------------------------------------------------------
    # Knee visibility
    # --------------------------------------------------------

    knees_visible = check_knees_visible(
        points,
        frame_width,
        frame_height
    )

    features[
        "lower_body_visible"
    ] = knees_visible

    if not knees_visible:

        features["valid"] = True

        # Upper-body quality is still useful.
        visible = 0

        for idx in [
            LEFT_SHOULDER,
            RIGHT_SHOULDER,
            LEFT_HIP,
            RIGHT_HIP
        ]:

            if landmark_valid(
                points[idx]
            ):
                visible += 1

        features["body_quality"] = (
            visible / 4.0
        )

        return features

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

    features[
        "left_knee_angle"
    ] = left_knee_angle

    features[
        "right_knee_angle"
    ] = right_knee_angle

    valid_knees = [
        value
        for value in [
            left_knee_angle,
            right_knee_angle
        ]
        if value is not None
    ]

    if valid_knees:

        features["knee_angle"] = (
            sum(valid_knees)
            /
            len(valid_knees)
        )

    # --------------------------------------------------------
    # Hip angle
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

    valid_hips = [
        value
        for value in [
            left_hip_angle,
            right_hip_angle
        ]
        if value is not None
    ]

    if valid_hips:

        features["hip_angle"] = (
            sum(valid_hips)
            /
            len(valid_hips)
        )

    # --------------------------------------------------------
    # Thigh orientation
    #
    # Useful when sitting but ankles are outside
    # the camera.
    # --------------------------------------------------------

    features[
        "left_thigh_angle"
    ] = thigh_angle_from_vertical(
        points[LEFT_HIP],
        points[LEFT_KNEE]
    )

    features[
        "right_thigh_angle"
    ] = thigh_angle_from_vertical(
        points[RIGHT_HIP],
        points[RIGHT_KNEE]
    )

    # --------------------------------------------------------
    # Body quality
    # --------------------------------------------------------

    important_points = [
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_HIP,
        RIGHT_HIP,
        LEFT_KNEE,
        RIGHT_KNEE
    ]

    visible_count = sum(
        landmark_valid(
            points[idx]
        )
        for idx in important_points
    )

    features["body_quality"] = (
        visible_count
        /
        len(important_points)
    )

    features["valid"] = True

    return features


# ============================================================
# POSTURE CLASSIFICATION
# ============================================================

def classify_posture(
    features,
    track
):
    """
    Classification priority:

        1. LIMITED VIEW
        2. FALLING
        3. SITTING
        4. STANDING
        5. WALKING
        6. UNCERTAIN

    LIMITED VIEW is deliberately checked first.

    Therefore an upper-body-only image can NEVER become
    STANDING simply because the torso is upright.
    """

    if not features["valid"]:

        return (
            "limited_view",
            0.0,
            "insufficient body landmarks"
        )

    # ========================================================
    # 1. LIMITED VIEW
    # ========================================================

    if not features[
        "lower_body_visible"
    ]:

        return (
            "limited_view",
            features["body_quality"],
            (
                "both knees are not reliably "
                "visible inside the camera frame"
            )
        )

    torso = features[
        "torso_angle"
    ]

    knee = features[
        "knee_angle"
    ]

    hip_angle = features[
        "hip_angle"
    ]

    left_thigh = features[
        "left_thigh_angle"
    ]

    right_thigh = features[
        "right_thigh_angle"
    ]

    # ========================================================
    # 2. HORIZONTAL BODY / FALL CANDIDATE
    # ========================================================

    if (
        torso is not None
        and
        torso >= FALL_TORSO_ANGLE
    ):

        confidence = min(
            1.0,
            0.65
            +
            (
                torso
                -
                FALL_TORSO_ANGLE
            )
            / 50.0
        )

        return (
            "falling",
            confidence,
            "torso orientation is strongly horizontal"
        )

    # ========================================================
    # 3. SITTING
    # ========================================================

    sitting_score = 0.0

    # Bent knees are the strongest signal.
    if knee is not None:

        if knee < 140:

            sitting_score += 0.55

        elif knee < 150:

            sitting_score += 0.45

        elif knee < SITTING_KNEE_ANGLE:

            sitting_score += 0.30

    # Thigh becomes more horizontal when sitting.
    valid_thighs = [
        value
        for value in [
            left_thigh,
            right_thigh
        ]
        if value is not None
    ]

    if valid_thighs:

        average_thigh_angle = (
            sum(valid_thighs)
            /
            len(valid_thighs)
        )

        if average_thigh_angle >= 55:

            sitting_score += 0.30

        elif average_thigh_angle >= 40:

            sitting_score += 0.15

    # Hip geometry.
    if hip_angle is not None:

        if hip_angle < 115:

            sitting_score += 0.20

        elif hip_angle < 135:

            sitting_score += 0.10

    # ========================================================
    # 4. STANDING
    # ========================================================

    standing_score = 0.0

    if knee is not None:

        if knee >= STANDING_KNEE_ANGLE:

            standing_score += 0.50

        elif knee >= 160:

            standing_score += 0.35

        elif knee >= 150:

            standing_score += 0.15

    # Upright torso.
    if (
        torso is not None
        and
        torso <= UPRIGHT_TORSO_ANGLE
    ):

        standing_score += 0.30

    elif (
        torso is not None
        and
        torso <= 42
    ):

        standing_score += 0.15

    # Both thighs relatively vertical.
    if valid_thighs:

        average_thigh_angle = (
            sum(valid_thighs)
            /
            len(valid_thighs)
        )

        if average_thigh_angle <= 25:

            standing_score += 0.20

        elif average_thigh_angle <= 40:

            standing_score += 0.10

    # ========================================================
    # 5. WALKING
    # ========================================================

    movement = getattr(
        track,
        "body_movement",
        0.0
    )

    walking_score = 0.0

    if (
        movement
        >=
        WALKING_MOVEMENT_THRESHOLD
    ):

        walking_score = min(
            0.90,
            0.50
            +
            movement / 50.0
        )

        # Walking should not beat a very strong
        # sitting/standing geometry decision.
        if sitting_score >= 0.70:
            walking_score *= 0.5

        if standing_score >= 0.75:
            walking_score *= 0.7

    # ========================================================
    # RANK
    # ========================================================

    scores = {
        "sitting": sitting_score,
        "standing": standing_score,
        "walking": walking_score
    }

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    state = ranked[0][0]
    confidence = ranked[0][1]

    second_confidence = (
        ranked[1][1]
    )

    # ========================================================
    # STRONG SITTING
    # ========================================================

    if (
        knee is not None
        and
        knee < 145
        and
        sitting_score >= 0.55
    ):

        return (
            "sitting",
            min(
                1.0,
                sitting_score
                +
                0.15
            ),
            "knees strongly bent"
        )

    # ========================================================
    # STRONG STANDING
    # ========================================================

    if (
        knee is not None
        and
        knee >= STANDING_KNEE_ANGLE
        and
        torso is not None
        and
        torso <= UPRIGHT_TORSO_ANGLE
    ):

        return (
            "standing",
            min(
                1.0,
                standing_score
                +
                0.15
            ),
            "knees extended and torso upright"
        )

    # ========================================================
    # AMBIGUITY
    # ========================================================

    if (
        confidence < MIN_POSTURE_CONFIDENCE
        or
        (
            confidence
            -
            second_confidence
            < 0.10
        )
    ):

        return (
            "uncertain",
            confidence,
            (
                f"sitting={sitting_score:.2f}; "
                f"standing={standing_score:.2f}; "
                f"walking={walking_score:.2f}"
            )
        )

    return (
        state,
        confidence,
        (
            f"sitting={sitting_score:.2f}; "
            f"standing={standing_score:.2f}; "
            f"walking={walking_score:.2f}"
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
    # LIMITED VIEW MUST BE IMMEDIATE
    # --------------------------------------------------------

    if (
        detected_state
        ==
        "limited_view"
    ):

        track.candidate_state = None
        track.candidate_started = None

        track.posture_history.clear()

        return "limited_view"

    # --------------------------------------------------------
    # FALL MUST BE RESPONSIVE
    # --------------------------------------------------------

    if (
        detected_state
        ==
        "falling"
    ):

        track.candidate_state = None
        track.candidate_started = None

        return "falling"

    # --------------------------------------------------------
    # UNCERTAIN
    # --------------------------------------------------------

    if (
        detected_state
        ==
        "uncertain"
    ):

        if track.state in (
            "unknown",
            "limited_view",
            "uncertain"
        ):

            return "uncertain"

        return track.state

    # --------------------------------------------------------
    # Same state
    # --------------------------------------------------------

    if (
        detected_state
        ==
        track.state
    ):

        track.candidate_state = None
        track.candidate_started = None

        track.posture_history.clear()

        return track.state

    # --------------------------------------------------------
    # New candidate
    # --------------------------------------------------------

    if (
        track.candidate_state
        !=
        detected_state
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
        >=
        POSTURE_CONFIRM_FRAMES
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
        (
            now
            -
            track.candidate_started
        )
        >=
        POSTURE_CONFIRM_SECONDS
    )

    if (
        frame_confirmed
        and
        time_confirmed
        and
        confidence
        >=
        MIN_POSTURE_CONFIDENCE
    ):

        track.candidate_state = None
        track.candidate_started = None

        track.posture_history.clear()

        return detected_state

    # If no reliable previous state exists,
    # use the current candidate.
    if track.state in (
        "unknown",
        "limited_view",
        "uncertain"
    ):

        return detected_state

    return track.state


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
            0.60,
            0.10,
            0.98,
            0.95
        )
    )

    x1, y1, x2, y2 = zone

    return (
        x1 * width
        <=
        center[0]
        <=
        x2 * width
        and
        y1 * height
        <=
        center[1]
        <=
        y2 * height
    )


def draw_zone(frame):

    h, w = frame.shape[:2]

    zone = cfg(
        "ZONE",
        (
            0.60,
            0.10,
            0.98,
            0.95
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
# SKELETON
# ============================================================

def draw_skeleton(
    frame,
    points,
    color
):

    connections = [

        # shoulders
        (11, 12),

        # left arm
        (11, 13),
        (13, 15),

        # right arm
        (12, 14),
        (14, 16),

        # torso
        (11, 23),
        (12, 24),
        (23, 24),

        # left leg
        (23, 25),
        (25, 27),

        # right leg
        (24, 26),
        (26, 28),

        # feet
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

        if not (
            landmark_valid(
                points[a]
            )
            and
            landmark_valid(
                points[b]
            )
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

        if landmark_valid(point):

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

        # Additional CSV logging.
        with EVENT_LOG.open(
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                timestamp(),
                track_id,
                event_type,
                activity,
                details,
                image_path
            ])

        print(
            f"[EVENT #{event_id}] "
            f"{event_type} | "
            f"Person {track_id} | "
            f"{activity} | "
            f"{details}"
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
# ALERT COOLDOWN
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
        now
        -
        previous
        >=
        cooldown
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
# FALL DETECTION
# ============================================================

def update_fall_detection(
    track,
    features,
    previous_state,
    now
):

    """
    Person-fall detection.

    A fall candidate requires:

        1. Lower body visible.
        2. Person was previously upright/moving.
        3. Hip moves downward rapidly.
        4. Torso becomes strongly horizontal.
        5. Condition persists for confirmation time.

    This intentionally does NOT classify a falling object.
    Object detection would be a separate module.
    """

    if not features[
        "lower_body_visible"
    ]:

        track.fall_candidate_started = None

        return False

    hip = features["hip"]

    if hip is None:

        track.fall_candidate_started = None

        return False

    hip_y = (
        hip[1]
        /
        max(
            1,
            1
        )
    )

    # We store normalized hip position.
    hip_y = (
        hip[1]
        /
        current_frame_height
    )


# ============================================================
# MAIN FALL LOGIC
# ============================================================

def process_fall(
    track,
    features,
    previous_state,
    now,
    frame,
    frame_height
):

    if not features[
        "lower_body_visible"
    ]:

        track.fall_candidate_started = None

        return False

    hip = features["hip"]

    if hip is None:

        track.fall_candidate_started = None

        return False

    hip_y = (
        float(hip[1])
        /
        float(frame_height)
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

    torso = features[
        "torso_angle"
    ]

    if torso is None:

        track.fall_candidate_started = None

        return False

    # --------------------------------------------------------
    # Detect sudden downward motion.
    # --------------------------------------------------------

    rapid_drop = (
        hip_drop
        >=
        FALL_DROP_RATIO
    )

    horizontal_body = (
        torso
        >=
        FALL_TORSO_ANGLE
    )

    previously_upright = (
        previous_state
        in (
            "standing",
            "walking"
        )
    )

    if (
        rapid_drop
        and
        previously_upright
    ):

        if (
            track.fall_candidate_started
            is None
        ):

            track.fall_candidate_started = now

    # --------------------------------------------------------
    # Once torso becomes horizontal, retain the candidate
    # long enough to confirm it.
    # --------------------------------------------------------

    if (
        track.fall_candidate_started
        is not None
        and
        horizontal_body
    ):

        duration = (
            now
            -
            track.fall_candidate_started
        )

        if (
            duration
            >=
            FALL_CONFIRM_SECONDS
        ):

            if can_alert(
                track,
                "FALL",
                now,
                ALERT_COOLDOWN_SECONDS
            ):

                details = (
                    f"previous_state="
                    f"{previous_state}; "
                    f"hip_drop="
                    f"{hip_drop:.3f}; "
                    f"torso_angle="
                    f"{torso:.1f}; "
                    f"confirmation="
                    f"{duration:.2f}s"
                )

                capture_and_log(
                    frame,
                    track.id,
                    "FALL_DETECTED",
                    "falling",
                    details,
                    category="fall"
                )

                mark_alert(
                    track,
                    "FALL",
                    now
                )

                track.fall_alerted_at = now

                track.fall_candidate_started = None

                return True

    # --------------------------------------------------------
    # Cancel stale candidate.
    # --------------------------------------------------------

    if (
        not rapid_drop
        and
        not horizontal_body
    ):

        track.fall_candidate_started = None

    return False


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("AI CAMERA TRACKER")
    print("=" * 65)

    ensure_model()
    ensure_dirs()

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------

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

    previous_time = (
        time.monotonic()
    )

    # ========================================================
    # CAMERA LOOP
    # ========================================================

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

            elapsed_frame = (
                now
                -
                previous_time
            )

            previous_time = now

            timestamp_ms += max(
                1,
                int(
                    elapsed_frame
                    *
                    1000
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
            # Monitoring zone
            # ------------------------------------------------

            draw_zone(
                frame
            )

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
                    w,
                    h
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

                track.body_movement = (
                    movement
                )

                # =================================================
                # CLASSIFICATION
                # =================================================

                detected_state, confidence, reason = (
                    classify_posture(
                        features,
                        track
                    )
                )

                previous_state = (
                    track.state
                )

                # =================================================
                # FALL DETECTION
                # =================================================

                fall_confirmed = process_fall(
                    track,
                    features,
                    previous_state,
                    now,
                    frame,
                    h
                )

                if fall_confirmed:

                    detected_state = (
                        "falling"
                    )

                    confidence = 1.0

                    reason = (
                        "confirmed rapid body "
                        "drop with horizontal torso"
                    )

                # =================================================
                # STATE SMOOTHING
                # =================================================

                state = stabilize_posture(
                    track,
                    detected_state,
                    confidence,
                    now
                )

                # =================================================
                # STATE CHANGE
                # =================================================

                if state != previous_state:

                    track.previous_state = (
                        previous_state
                    )

                    track.state = (
                        state
                    )

                    track.state_started = (
                        now
                    )

                    track.sitting_alerted = (
                        False
                    )

                    track.standing_alerted = (
                        False
                    )

                    # ------------------------------------------------
                    # NORMAL POSTURE EVENT
                    # ------------------------------------------------

                    if state in (
                        "sitting",
                        "standing",
                        "walking",
                        "falling"
                    ):

                        if can_alert(
                            track,
                            f"STATE_{state}",
                            now,
                            3
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
                                f"reason="
                                f"{reason}"
                            )

                            category = (
                                "fall"
                                if state
                                ==
                                "falling"
                                else
                                "posture"
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

                    # ------------------------------------------------
                    # LIMITED VIEW EVENT
                    # ------------------------------------------------

                    elif (
                        state
                        ==
                        "limited_view"
                    ):

                        if can_alert(
                            track,
                            "LIMITED_VIEW",
                            now,
                            5
                        ):

                            details = (
                                "Both knees are not "
                                "reliably visible inside "
                                "the camera frame. "
                                "Standing/sitting "
                                "classification suppressed."
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
                # DISPLAY STATE
                # =================================================

                display_state = (
                    track.state
                )

                if (
                    display_state
                    ==
                    "unknown"
                ):

                    display_state = (
                        detected_state
                    )

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
                    state
                    ==
                    "sitting"
                    and
                    state_elapsed
                    >=
                    SITTING_DWELL_SECONDS
                    and
                    not track.sitting_alerted
                ):

                    details = (
                        f"{fmt(state_elapsed)} "
                        "sitting"
                    )

                    capture_and_log(
                        frame,
                        track.id,
                        "SITTING_DWELL_THRESHOLD",
                        "sitting",
                        details,
                        "events"
                    )

                    track.sitting_alerted = (
                        True
                    )

                # =================================================
                # STANDING DWELL
                # =================================================

                if (
                    state
                    ==
                    "standing"
                    and
                    state_elapsed
                    >=
                    STANDING_DWELL_SECONDS
                    and
                    not track.standing_alerted
                ):

                    details = (
                        f"{fmt(state_elapsed)} "
                        "standing"
                    )

                    capture_and_log(
                        frame,
                        track.id,
                        "STANDING_DWELL_THRESHOLD",
                        "standing",
                        details,
                        "events"
                    )

                    track.standing_alerted = (
                        True
                    )

                # =================================================
                # ZONE
                # =================================================

                in_zone = inside_zone(
                    track.center,
                    w,
                    h
                )

                # ------------------------------------------------
                # First initialization
                # ------------------------------------------------

                if not track.zone_initialized:

                    track.was_in_zone = (
                        in_zone
                    )

                    track.zone_initialized = (
                        True
                    )

                    if in_zone:

                        track.zone_started = (
                            now
                        )

                    else:

                        track.zone_started = (
                            None
                        )

                # ------------------------------------------------
                # ENTRY
                # ------------------------------------------------

                elif (
                    in_zone
                    and
                    not track.was_in_zone
                ):

                    track.zone_started = (
                        now
                    )

                    track.zone_alerted = (
                        False
                    )

                    details = (
                        "Person entered "
                        "monitoring zone."
                    )

                    if can_alert(
                        track,
                        "ZONE_ENTRY",
                        now,
                        2
                    ):

                        capture_and_log(
                            frame,
                            track.id,
                            "ZONE_ENTRY",
                            state,
                            details,
                            "zone"
                        )

                        mark_alert(
                            track,
                            "ZONE_ENTRY",
                            now
                        )

                    print(
                        f"[ZONE] Person "
                        f"{track.id} ENTERED "
                        f"monitoring zone"
                    )

                # ------------------------------------------------
                # EXIT
                # ------------------------------------------------

                elif (
                    not in_zone
                    and
                    track.was_in_zone
                ):

                    zone_elapsed = 0

                    if (
                        track.zone_started
                        is not None
                    ):

                        zone_elapsed = (
                            now
                            -
                            track.zone_started
                        )

                    details = (
                        "Person exited "
                        "monitoring zone; "
                        f"dwell="
                        f"{fmt(zone_elapsed)}."
                    )

                    if can_alert(
                        track,
                        "ZONE_EXIT",
                        now,
                        2
                    ):

                        capture_and_log(
                            frame,
                            track.id,
                            "ZONE_EXIT",
                            state,
                            details,
                            "zone"
                        )

                        mark_alert(
                            track,
                            "ZONE_EXIT",
                            now
                        )

                    print(
                        f"[ZONE] Person "
                        f"{track.id} EXITED "
                        f"monitoring zone"
                    )

                    track.zone_started = (
                        None
                    )

                    track.zone_alerted = (
                        False
                    )

                # ------------------------------------------------
                # Update previous zone state
                # ------------------------------------------------

                track.was_in_zone = (
                    in_zone
                )

                # ------------------------------------------------
                # Zone timer
                # ------------------------------------------------

                if in_zone:

                    if (
                        track.zone_started
                        is None
                    ):

                        track.zone_started = (
                            now
                        )

                    zone_elapsed = (
                        now
                        -
                        track.zone_started
                    )

                    # ------------------------------------------------
                    # Zone dwell event
                    # ------------------------------------------------

                    if (
                        zone_elapsed
                        >=
                        ZONE_DWELL_SECONDS
                        and
                        not track.zone_alerted
                    ):

                        details = (
                            f"{fmt(zone_elapsed)} "
                            "inside monitoring zone"
                        )

                        capture_and_log(
                            frame,
                            track.id,
                            "ZONE_DWELL_THRESHOLD",
                            display_state,
                            details,
                            "zone"
                        )

                        track.zone_alerted = (
                            True
                        )

                else:

                    zone_elapsed = 0

                # =================================================
                # DRAW PERSON
                # =================================================

                x1, y1, x2, y2 = map(
                    int,
                    track.bbox
                )

                # ------------------------------------------------
                # Colors
                # ------------------------------------------------

                if (
                    display_state
                    ==
                    "falling"
                ):

                    box_color = (
                        0,
                        0,
                        255
                    )

                elif (
                    display_state
                    ==
                    "limited_view"
                ):

                    box_color = (
                        0,
                        165,
                        255
                    )

                elif (
                    display_state
                    ==
                    "sitting"
                ):

                    box_color = (
                        255,
                        0,
                        255
                    )

                elif (
                    display_state
                    ==
                    "standing"
                ):

                    box_color = (
                        0,
                        255,
                        0
                    )

                elif (
                    display_state
                    ==
                    "walking"
                ):

                    box_color = (
                        255,
                        255,
                        0
                    )

                elif (
                    display_state
                    ==
                    "uncertain"
                ):

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

                if display_state in (
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

                if (
                    display_state
                    ==
                    "limited_view"
                ):

                    cv2.putText(
                        frame,
                        "LIMITED VIEW",
                        (
                            25,
                            40
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (
                            0,
                            165,
                            255
                        ),
                        3
                    )

                    cv2.putText(
                        frame,
                        "Knees not visible - posture suppressed",
                        (
                            25,
                            70
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (
                            0,
                            165,
                            255
                        ),
                        2
                    )

                # =================================================
                # FALL WARNING
                # =================================================

                if (
                    display_state
                    ==
                    "falling"
                ):

                    cv2.putText(
                        frame,
                        "POSSIBLE FALL - CONFIRMING",
                        (
                            25,
                            40
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (
                            0,
                            0,
                            255
                        ),
                        3
                    )

            # ====================================================
            # CONTROLS
            # ====================================================

            cv2.putText(
                frame,
                "C = capture image | Q = quit",
                (
                    20,
                    h - 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (
                    255,
                    255,
                    255
                ),
                2
            )

            # ====================================================
            # DISPLAY
            # ====================================================

            cv2.imshow(
                "AI Camera Tracker - Smart Posture",
                frame
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            # ====================================================
            # MANUAL CAPTURE
            # ====================================================

            if key == ord("c"):

                active_tracks = [
                    track
                    for track in tracks
                    if not track.missed
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
                        "manual"
                    )

                else:

                    capture_and_log(
                        frame,
                        0,
                        "MANUAL_CAPTURE",
                        "unknown",
                        "Manual capture; no person detected.",
                        "manual"
                    )

            # ====================================================
            # QUIT
            # ====================================================

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