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
from object_detector import ObjectDetector
from deepcamera_adapter import DeepCameraDetector
from security_alerts import SecurityAlertManager


# ============================================================
# PREDICTIVE SUBJECT TRACKING / VIRTUAL PTZ
# ============================================================

class PredictiveSubjectTracker:
    """
    Predictive subject tracking layer.

    Tracks the currently selected person and estimates:
    - current position
    - movement velocity
    - movement speed
    - predicted future position
    - virtual pan direction
    - virtual tilt direction

    This does NOT physically move the camera.
    It provides the intelligence required for a future
    PTZ camera integration.
    """

    def __init__(
        self,
        frame_width,
        frame_height,
        prediction_frames=8,
        smoothing=0.20,
    ):

        self.frame_width = frame_width
        self.frame_height = frame_height

        self.prediction_frames = prediction_frames
        self.smoothing = smoothing

        self.target_id = None
        self.previous_position = None

        self.predicted_position = (
            frame_width / 2.0,
            frame_height / 2.0
        )

        self.velocity = (
            0.0,
            0.0
        )

        self.speed = 0.0

    # ========================================================
    # UPDATE
    # ========================================================

    def update(self, tracks):

        visible_tracks = [
            track
            for track in tracks
            if not track.missed
        ]

        if not visible_tracks:
            return None

        target = None

        if self.target_id is not None:

            for track in visible_tracks:

                if track.id == self.target_id:

                    target = track
                    break

        if target is None:

            target = visible_tracks[0]
            self.target_id = target.id
            self.previous_position = None

        current_position = (
            float(target.center[0]),
            float(target.center[1])
        )

        if self.previous_position is None:

            velocity_x = 0.0
            velocity_y = 0.0

        else:

            velocity_x = (
                current_position[0]
                -
                self.previous_position[0]
            )

            velocity_y = (
                current_position[1]
                -
                self.previous_position[1]
            )

        self.previous_position = current_position

        self.velocity = (
            velocity_x,
            velocity_y
        )

        self.speed = math.hypot(
            velocity_x,
            velocity_y
        )

        predicted_x = (
            current_position[0]
            +
            velocity_x
            *
            self.prediction_frames
        )

        predicted_y = (
            current_position[1]
            +
            velocity_y
            *
            self.prediction_frames
        )

        predicted_x = max(
            0.0,
            min(
                float(self.frame_width),
                predicted_x
            )
        )

        predicted_y = max(
            0.0,
            min(
                float(self.frame_height),
                predicted_y
            )
        )

        self.predicted_position = (
            self.predicted_position[0]
            +
            (
                predicted_x
                -
                self.predicted_position[0]
            )
            *
            self.smoothing,

            self.predicted_position[1]
            +
            (
                predicted_y
                -
                self.predicted_position[1]
            )
            *
            self.smoothing
        )

        frame_center_x = (
            self.frame_width / 2.0
        )

        frame_center_y = (
            self.frame_height / 2.0
        )

        error_x = (
            self.predicted_position[0]
            -
            frame_center_x
        )

        error_y = (
            self.predicted_position[1]
            -
            frame_center_y
        )

        horizontal_threshold = (
            self.frame_width * 0.08
        )

        vertical_threshold = (
            self.frame_height * 0.08
        )

        if error_x > horizontal_threshold:
            pan_direction = "RIGHT"

        elif error_x < -horizontal_threshold:
            pan_direction = "LEFT"

        else:
            pan_direction = "CENTER"

        if error_y > vertical_threshold:
            tilt_direction = "DOWN"

        elif error_y < -vertical_threshold:
            tilt_direction = "UP"

        else:
            tilt_direction = "CENTER"

        return {
            "person_id": target.id,
            "current_position": current_position,
            "predicted_position": self.predicted_position,
            "velocity": self.velocity,
            "speed": self.speed,
            "pan_direction": pan_direction,
            "tilt_direction": tilt_direction,
            "error_x": error_x,
            "error_y": error_y,
        }

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.target_id = None

        self.previous_position = None

        self.predicted_position = (
            self.frame_width / 2.0,
            self.frame_height / 2.0
        )

        self.velocity = (
            0.0,
            0.0
        )

        self.speed = 0.0


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

    return getattr(
        config,
        name,
        default
    )


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
    10
)

SITTING_DWELL_SECONDS = cfg(
    "SITTING_DWELL_SECONDS",
    120
)

STANDING_DWELL_SECONDS = cfg(
    "STANDING_DWELL_SECONDS",
    300
)

ZONE_DWELL_SECONDS = cfg(
    "ZONE_DWELL_SECONDS",
    120
)


# ============================================================
# POSTURE SETTINGS
# ============================================================

MIN_LOWER_BODY_POINTS = 4

SITTING_KNEE_ANGLE = 155

STANDING_KNEE_ANGLE = 168

UPRIGHT_TORSO_ANGLE = 32

STRONG_SITTING_KNEE_ANGLE = 145

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
# MODEL / DIRECTORY SETUP
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

    for folder in (
        "fall",
        "limited_view",
        "zone",
        "posture",
        "manual"
    ):

        (
            CAPTURE_DIR / folder
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    if not EVENT_LOG.exists():

        with EVENT_LOG.open(
            "w",
            newline=""
        ) as f:

            csv.writer(f).writerow(
                [
                    "timestamp",
                    "track_id",
                    "event",
                    "details",
                    "image_path"
                ]
            )


# ============================================================
# EVENT LOGGING
# ============================================================

def log_event(
    track_id,
    event,
    details,
    image_path=""
):

    stamp = datetime.now().isoformat(
        timespec="seconds"
    )

    with EVENT_LOG.open(
        "a",
        newline=""
    ) as f:

        csv.writer(f).writerow(
            [
                stamp,
                track_id,
                event,
                details,
                image_path
            ]
        )

    print(
        f"[ALERT] {stamp} | "
        f"Person {track_id} | "
        f"{event} | "
        f"{details}"
    )


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

        log_event(
            track_id,
            event_type,
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
        float(a[0]) - float(b[0]),
        float(a[1]) - float(b[1])
    )


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


# ============================================================
# ZONE
# ============================================================

def inside_zone(
    center,
    w,
    h
):

    x1, y1, x2, y2 = cfg(
        "ZONE",
        (
            0.60,
            0.10,
            0.98,
            0.95
        )
    )

    return (
        x1 * w <= center[0] <= x2 * w
        and
        y1 * h <= center[1] <= y2 * h
    )


def draw_zone(frame):

    h, w = frame.shape[:2]

    x1, y1, x2, y2 = cfg(
        "ZONE",
        (
            0.60,
            0.10,
            0.98,
            0.95
        )
    )

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
# LANDMARK VALIDATION
# ============================================================

def landmark_valid(point):

    if point is None:
        return False

    try:

        x, y = point

        if not math.isfinite(
            float(x)
        ):
            return False

        if not math.isfinite(
            float(y)
        ):
            return False

        if x < 0 or y < 0:
            return False

        return True

    except (
        TypeError,
        ValueError
    ):

        return False


def valid_point(point):

    return landmark_valid(
        point
    )


def point_near_bottom(
    point,
    frame_height
):

    if not valid_point(point):
        return True

    return (
        point[1]
        >=
        frame_height * 0.94
    )


# ============================================================
# LOWER BODY VISIBILITY
# ============================================================

def check_upper_body_visibility(
    points
):

    required = [
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_HIP,
        RIGHT_HIP
    ]

    return all(
        idx < len(points)
        and landmark_valid(
            points[idx]
        )
        for idx in required
    )


def check_knees_visible(
    points,
    frame_width,
    frame_height
):

    if len(points) < 33:
        return False

    left_knee = points[
        LEFT_KNEE
    ]

    right_knee = points[
        RIGHT_KNEE
    ]

    left_ankle = points[
        LEFT_ANKLE
    ]

    right_ankle = points[
        RIGHT_ANKLE
    ]

    knees_valid = (
        landmark_valid(
            left_knee
        )
        and
        landmark_valid(
            right_knee
        )
    )

    if not knees_valid:
        return False

    knees_at_bottom = (
        point_near_bottom(
            left_knee,
            frame_height
        )
        and
        point_near_bottom(
            right_knee,
            frame_height
        )
    )

    ankles_valid = (
        landmark_valid(
            left_ankle
        )
        and
        landmark_valid(
            right_ankle
        )
    )

    ankles_at_bottom = (
        point_near_bottom(
            left_ankle,
            frame_height
        )
        and
        point_near_bottom(
            right_ankle,
            frame_height
        )
    )

    if (
        knees_at_bottom
        and
        not ankles_valid
    ):

        return False

    if ankles_at_bottom:
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

def angle(
    a,
    b,
    c
):

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
        landmark_valid(
            shoulder
        )
        and
        landmark_valid(
            hip
        )
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
        "leg_torso_ratio": 0.0,
        "shoulder_hip_ratio": 0.0
    }

    if len(points) < 33:
        return features

    shoulder_valid = (
        landmark_valid(
            points[LEFT_SHOULDER]
        )
        and
        landmark_valid(
            points[RIGHT_SHOULDER]
        )
    )

    hip_valid = (
        landmark_valid(
            points[LEFT_HIP]
        )
        and
        landmark_valid(
            points[RIGHT_HIP]
        )
    )

    if not (
        shoulder_valid
        and
        hip_valid
    ):

        return features

    shoulder = midpoint(
        points[LEFT_SHOULDER],
        points[RIGHT_SHOULDER]
    )

    hip = midpoint(
        points[LEFT_HIP],
        points[RIGHT_HIP]
    )

    torso = torso_angle(
        shoulder,
        hip
    )

    features[
        "torso_angle"
    ] = torso

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

    knee_angles = [
        value
        for value in (
            left_knee_angle,
            right_knee_angle
        )
        if value is not None
    ]

    if knee_angles:

        features[
            "knee_angle"
        ] = (
            sum(knee_angles)
            /
            len(knee_angles)
        )

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
        value
        for value in (
            left_hip_angle,
            right_hip_angle
        )
        if value is not None
    ]

    if hip_angles:

        features[
            "hip_angle"
        ] = (
            sum(hip_angles)
            /
            len(hip_angles)
        )

    if (
        landmark_valid(
            points[LEFT_HIP]
        )
        and
        landmark_valid(
            points[LEFT_KNEE]
        )
    ):

        dx = (
            points[LEFT_KNEE][0]
            -
            points[LEFT_HIP][0]
        )

        dy = (
            points[LEFT_KNEE][1]
            -
            points[LEFT_HIP][1]
        )

        features[
            "left_thigh_angle"
        ] = abs(
            math.degrees(
                math.atan2(
                    dx,
                    dy
                )
            )
        )

    if (
        landmark_valid(
            points[RIGHT_HIP]
        )
        and
        landmark_valid(
            points[RIGHT_KNEE]
        )
    ):

        dx = (
            points[RIGHT_KNEE][0]
            -
            points[RIGHT_HIP][0]
        )

        dy = (
            points[RIGHT_KNEE][1]
            -
            points[RIGHT_HIP][1]
        )

        features[
            "right_thigh_angle"
        ] = abs(
            math.degrees(
                math.atan2(
                    dx,
                    dy
                )
            )
        )

    features[
        "lower_body_visible"
    ] = lower_body_visibility(
        points,
        frame_width,
        frame_height
    )

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

    valid_upper = sum(
        landmark_valid(point)
        for point in upper_points
    )

    valid_lower = sum(
        landmark_valid(point)
        for point in lower_points
    )

    features[
        "body_quality"
    ] = (
        valid_upper
        +
        valid_lower
    ) / 8.0

    leg_lengths = []

    if all(
        landmark_valid(points[idx])
        for idx in (
            LEFT_HIP,
            LEFT_KNEE,
            LEFT_ANKLE
        )
    ):

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

    if all(
        landmark_valid(points[idx])
        for idx in (
            RIGHT_HIP,
            RIGHT_KNEE,
            RIGHT_ANKLE
        )
    ):

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

    if torso_length > 0:

        features[
            "leg_torso_ratio"
        ] = (
            leg_length
            /
            torso_length
        )

        features[
            "shoulder_hip_ratio"
        ] = (
            torso_length
            /
            frame_height
        )

    features["valid"] = True

    return features


# ============================================================
# CONFIDENCE
# ============================================================

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
            "insufficient landmarks"
        )

    if not features[
        "lower_body_visible"
    ]:

        return (
            "limited_view",
            features[
                "body_quality"
            ],
            "lower body not reliably visible"
        )

    knee_angle = features[
        "knee_angle"
    ]

    torso = features[
        "torso_angle"
    ]

    hip_angle = features[
        "hip_angle"
    ]

    leg_torso_ratio = features[
        "leg_torso_ratio"
    ]

    body_quality = features[
        "body_quality"
    ]

    if knee_angle is None:

        return (
            "limited_view",
            body_quality,
            "knee landmarks unavailable"
        )

    if torso is not None and (
        torso >= FALL_TORSO_ANGLE
    ):

        confidence = clamp(
            0.65
            +
            (
                torso
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

    sitting_score = 0.0

    if knee_angle < (
        STRONG_SITTING_KNEE_ANGLE
    ):

        sitting_score += 0.50

    elif knee_angle < (
        SITTING_KNEE_ANGLE
    ):

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

    standing_score = 0.0

    if knee_angle >= (
        STANDING_KNEE_ANGLE
    ):

        standing_score += 0.50

    elif knee_angle >= 155:

        standing_score += 0.20

    if (
        torso is not None
        and
        torso <= UPRIGHT_TORSO_ANGLE
    ):

        standing_score += 0.30

    if leg_torso_ratio >= 1.0:

        standing_score += 0.20

    standing_confidence = clamp(
        standing_score
        +
        0.15 * body_quality
    )

    movement = getattr(
        track,
        "body_movement",
        0.0
    )

    walking_confidence = 0.0

    if (
        movement
        >=
        WALKING_MOVEMENT_THRESHOLD
    ):

        walking_confidence = clamp(
            0.55
            +
            min(
                0.35,
                movement / 40.0
            )
        )

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

    if (
        confidence
        <
        MIN_POSTURE_CONFIDENCE
        or
        (
            confidence
            -
            second_confidence
            <
            0.10
            and
            confidence
            <
            0.75
        )
    ):

        return (
            "uncertain",
            confidence,
            (
                f"sitting="
                f"{sitting_confidence:.2f}; "
                f"standing="
                f"{standing_confidence:.2f}; "
                f"walking="
                f"{walking_confidence:.2f}"
            )
        )

    return (
        state,
        confidence,
        (
            f"sitting="
            f"{sitting_confidence:.2f}; "
            f"standing="
            f"{standing_confidence:.2f}; "
            f"walking="
            f"{walking_confidence:.2f}"
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

    if (
        detected_state
        ==
        "limited_view"
    ):

        track.candidate_state = None
        track.candidate_started = None

        track.posture_history.clear()

        return "limited_view"

    if (
        detected_state
        ==
        "falling"
    ):

        track.candidate_state = None
        track.candidate_started = None

        return "falling"

    if (
        detected_state
        ==
        "uncertain"
    ):

        if track.state not in (
            "unknown",
            "limited_view",
            "uncertain"
        ):

            return track.state

        return "uncertain"

    if (
        detected_state
        ==
        track.state
    ):

        track.candidate_state = None
        track.candidate_started = None

        track.posture_history.clear()

        return track.state

    if (
        track.candidate_state
        !=
        detected_state
    ):

        track.candidate_state = (
            detected_state
        )

        track.candidate_started = (
            now
        )

        track.posture_history.clear()

        track.posture_history.append(
            detected_state
        )

        return track.state

    track.posture_history.append(
        detected_state
    )

    if len(
        track.posture_history
    ) < POSTURE_CONFIRM_FRAMES:

        return track.state

    recent = list(
        track.posture_history
    )

    same_frames = sum(
        state == detected_state
        for state in recent
    )

    if (
        same_frames
        <
        POSTURE_CONFIRM_FRAMES
    ):

        return track.state

    if (
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
    ):

        track.candidate_state = None
        track.candidate_started = None

        track.posture_history.clear()

        return detected_state

    return track.state


# ============================================================
# FALL DETECTION
# ============================================================

def process_fall(
    track,
    features,
    previous_state,
    now,
    frame,
    frame_height
):

    torso = features.get(
        "torso_angle"
    )

    if torso is None:
        return False

    hip_drop = 0.0

    points = track.landmarks

    if (
        len(points) >= 33
        and
        landmark_valid(
            points[LEFT_HIP]
        )
        and
        landmark_valid(
            points[RIGHT_HIP]
        )
    ):

        hip = midpoint(
            points[LEFT_HIP],
            points[RIGHT_HIP]
        )

        current_hip_y = (
            hip[1]
            /
            frame_height
        )

        previous_hip_y = getattr(
            track,
            "previous_hip_y",
            None
        )

        if previous_hip_y is not None:

            hip_drop = (
                current_hip_y
                -
                previous_hip_y
            )

        track.previous_hip_y = (
            current_hip_y
        )

    rapid_drop = (
        hip_drop
        >=
        FALL_DROP_RATIO
    )

    horizontal = (
        torso
        >=
        FALL_TORSO_ANGLE
    )

    came_from_upright = (
        previous_state
        in
        (
            "standing",
            "walking"
        )
    )

    if (
        horizontal
        and
        (
            rapid_drop
            or
            came_from_upright
        )
    ):

        if (
            track.fall_candidate_started
            is None
        ):

            track.fall_candidate_started = (
                now
            )

        elapsed = (
            now
            -
            track.fall_candidate_started
        )

        if (
            elapsed
            >=
            FALL_CONFIRM_SECONDS
        ):

            last_alert = getattr(
                track,
                "fall_alerted_at",
                None
            )

            if (
                last_alert is None
                or
                (
                    now
                    -
                    last_alert
                    >=
                    ALERT_COOLDOWN_SECONDS
                )
            ):

                details = (
                    "Confirmed fall: "
                    f"torso_angle="
                    f"{torso:.1f}; "
                    f"hip_drop="
                    f"{hip_drop:.3f}; "
                    f"previous_state="
                    f"{previous_state}"
                )

                capture_and_log(
                    frame,
                    track.id,
                    "FALL_DETECTED",
                    "falling",
                    details,
                    "fall"
                )

                track.fall_alerted_at = (
                    now
                )

            return True

    else:

        track.fall_candidate_started = (
            None
        )

    return False


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
# OBJECT DETECTION DISPLAY
# ============================================================

def draw_objects(
    frame,
    object_detections
):

    h, w = frame.shape[:2]

    x1, y1, x2, y2 = cfg(
        "ZONE",
        (
            0.60,
            0.10,
            0.98,
            0.95
        )
    )

    zone_x1 = int(
        x1 * w
    )

    zone_y1 = int(
        y1 * h
    )

    zone_x2 = int(
        x2 * w
    )

    zone_y2 = int(
        y2 * h
    )

    visible_objects = []

    for obj in object_detections:

        class_name = str(
            obj.get(
                "class_name",
                "object"
            )
        )

        if (
            class_name.lower()
            ==
            "person"
        ):

            continue

        confidence = float(
            obj.get(
                "confidence",
                0.0
            )
        )

        x1_obj, y1_obj, x2_obj, y2_obj = (
            map(
                int,
                obj["bbox"]
            )
        )

        cx, cy = map(
            int,
            obj["center"]
        )

        if not (
            zone_x1 <= cx <= zone_x2
            and
            zone_y1 <= cy <= zone_y2
        ):

            continue

        visible_objects.append(
            obj
        )

        object_color = (
            255,
            200,
            0
        )

        cv2.rectangle(
            frame,
            (
                x1_obj,
                y1_obj
            ),
            (
                x2_obj,
                y2_obj
            ),
            object_color,
            2
        )

        label = (
            f"{class_name.upper()} "
            f"{confidence * 100:.0f}%"
        )

        cv2.putText(
            frame,
            label,
            (
                x1_obj,
                max(
                    20,
                    y1_obj - 8
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            object_color,
            2,
            cv2.LINE_AA
        )

    return visible_objects


# ============================================================
# OBJECT SUMMARY
# ============================================================

def draw_object_summary(
    frame,
    objects
):

    if not objects:

        cv2.putText(
            frame,
            "Objects: None detected",
            (
                20,
                105
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                255,
                200,
                0
            ),
            2
        )

        return

    counts = {}

    for obj in objects:

        name = str(
            obj["class_name"]
        )

        counts[name] = (
            counts.get(
                name,
                0
            )
            + 1
        )

    summary = "Objects: "

    summary += ", ".join(
        f"{name} ({count})"
        for name, count
        in counts.items()
    )

    cv2.putText(
        frame,
        summary[:140],
        (
            20,
            105
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (
            255,
            200,
            0
        ),
        2
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("AI CAMERA TRACKER")
    print("=" * 60)

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
    # Pose detector
    # --------------------------------------------------------

    detector = PoseDetector(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Person tracker
    # --------------------------------------------------------

    tracker = CentroidTracker(
        MAX_TRACK_DISTANCE,
        MAX_MISSED_FRAMES
    )

    # --------------------------------------------------------
    # DEEPCAMERA OBJECT DETECTOR
    # --------------------------------------------------------
    # DeepCamera's current detection skill uses YOLO 2026/YOLO26
    # and communicates detections as JSON objects.  We keep the
    # existing detector as a safety fallback so none of the current
    # monitoring features stop working if the YOLO26 model is not
    # available locally.
    # --------------------------------------------------------

    print(
        "[INFO] Loading DeepCamera object detector..."
    )

    legacy_object_detector = ObjectDetector(
        model_path="yolo11n.pt",
        confidence=0.50,
        iou=0.45
    )

    object_detector = DeepCameraDetector(
        model_path=cfg(
            "DEEPCAMERA_MODEL_PATH",
            "yolo26n.pt"
        ),
        confidence=cfg(
            "DEEPCAMERA_CONFIDENCE",
            0.50
        ),
        iou=cfg(
            "DEEPCAMERA_IOU",
            0.45
        ),
        fallback=legacy_object_detector
    )

    print(
        "[INFO] DeepCamera object detector ready."
    )

    # --------------------------------------------------------
    # ADMIN SECURITY MANAGER
    # --------------------------------------------------------

    try:

        security_manager = (
            SecurityAlertManager(
                cooldown_seconds=30
            )
        )

        security_enabled = True

        print(
            "[INFO] Admin security "
            "alert manager ready."
        )

    except Exception as exc:

        security_manager = None
        security_enabled = False

        print(
            "[WARNING] Security alert "
            f"manager unavailable: {exc}"
        )

        print(
            "[WARNING] Existing camera "
            "features will continue running."
        )

    # --------------------------------------------------------
    # Predictive subject tracker
    # --------------------------------------------------------

    predictive_tracker = (
        PredictiveSubjectTracker(
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
            prediction_frames=8,
            smoothing=0.20
        )
    )

    print(
        "[INFO] Predictive subject "
        "tracking ready."
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

    timestamp_ms = 0

    previous_time = (
        time.monotonic()
    )

    try:

        while True:

            ok, frame = cap.read()

            if not ok:

                print(
                    "[ERROR] Failed to read "
                    "camera frame."
                )

                break

            now = time.monotonic()

            timestamp_ms += max(
                1,
                int(
                    (
                        now
                        -
                        previous_time
                    )
                    *
                    1000
                )
            )

            previous_time = now

            h, w = frame.shape[:2]

            # =================================================
            # POSE DETECTION
            # =================================================

            detections = detector.detect(
                frame,
                timestamp_ms
            )

            # =================================================
            # PERSON TRACKING
            # =================================================

            tracks = tracker.update(
                detections,
                now
            )

            # =================================================
            # PREDICTIVE SUBJECT TRACKING
            # =================================================

            prediction = (
                predictive_tracker.update(
                    tracks
                )
            )

            # =================================================
            # YOLO OBJECT DETECTION
            # =================================================

            object_detections = (
                object_detector.detect(
                    frame
                )
            )

            visible_objects = draw_objects(
                frame,
                object_detections
            )

            draw_object_summary(
                frame,
                visible_objects
            )

            # =================================================
            # ADMIN SECURITY - THREAT DETECTION
            # =================================================
            #
            # IMPORTANT:
            # Use ALL YOLO detections here, not only
            # visible_objects, so a threat anywhere in the
            # camera frame can be escalated.
            #
            # The security manager itself decides whether
            # the detected class is a configured threat.
            # =================================================

            if (
                security_enabled
                and
                security_manager is not None
            ):

                security_objects = []

                for obj in (
                    object_detections
                    or []
                ):

                    label = str(
                        obj.get(
                            "class_name",
                            ""
                        )
                    ).strip()

                    if not label:
                        continue

                    if (
                        label.lower()
                        ==
                        "person"
                    ):
                        continue

                    security_objects.append(
                        {
                            "label": label,
                            "confidence": float(
                                obj.get(
                                    "confidence",
                                    0.0
                                )
                            )
                        }
                    )

                if security_objects:

                    try:

                        security_manager.evaluate_objects(
                            frame=frame,
                            person_id=0,
                            detected_objects=security_objects
                        )

                    except Exception as exc:

                        print(
                            "[WARNING] Threat "
                            "security check failed: "
                            f"{exc}"
                        )

            # =================================================
            # MONITORING ZONE
            # =================================================

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

                # =================================================
                # BODY FEATURES
                # =================================================

                features = get_body_features(
                    points,
                    w,
                    h
                )

                # =================================================
                # MOVEMENT
                # =================================================

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

                (
                    detected_state,
                    confidence,
                    reason
                ) = classify_posture(
                    features,
                    track
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

                # =================================================
                # ADMIN SECURITY - CONFIRMED FALL
                # =================================================

                if fall_confirmed:

                    if (
                        security_enabled
                        and
                        security_manager is not None
                    ):

                        try:

                            security_manager.evaluate_fall(
                                frame=frame,
                                person_id=track.id,
                                details=(
                                    "Fall confirmed by "
                                    "the existing camera "
                                    "fall-detection "
                                    "pipeline."
                                )
                            )

                        except Exception as exc:

                            print(
                                "[WARNING] Security "
                                "fall alert failed: "
                                f"{exc}"
                            )

                    detected_state = (
                        "falling"
                    )

                    confidence = 1.0

                    reason = (
                        "confirmed rapid "
                        "body drop / "
                        "horizontal torso"
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

                if (
                    state
                    !=
                    previous_state
                ):

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
                                "Lower body "
                                "not reliably visible; "
                                "posture classification "
                                "suppressed."
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

                    # ---------------------------------------------
                    # ADMIN SECURITY - ZONE ENTRY
                    # ---------------------------------------------

                    if (
                        security_enabled
                        and
                        security_manager is not None
                    ):

                        try:

                            security_manager.evaluate_restricted_zone(
                                frame=frame,
                                person_id=track.id,
                                details=(
                                    "Person entered "
                                    "the configured "
                                    "monitoring/security "
                                    "zone."
                                )
                            )

                        except Exception as exc:

                            print(
                                "[WARNING] Security "
                                "zone alert failed: "
                                f"{exc}"
                            )

                    print(
                        f"[ZONE] Person "
                        f"{track.id} ENTERED "
                        f"monitoring zone"
                    )

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

                track.was_in_zone = (
                    in_zone
                )

                zone_elapsed = 0

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

                # =================================================
                # PERSON DISPLAY
                # =================================================

                x1, y1, x2, y2 = (
                    track.bbox
                )

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

                if state in (
                    "limited_view",
                    "uncertain"
                ):

                    confidence_text = "--"

                else:

                    confidence_text = (
                        f"{confidence * 100:.0f}%"
                    )

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
                        int(x1),
                        int(y1)
                    ),
                    (
                        int(x2),
                        int(y2)
                    ),
                    box_color,
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (
                        int(x1),
                        max(
                            25,
                            int(y1) - 10
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
            # PREDICTIVE TRACKING DISPLAY
            # =================================================

            if prediction is not None:

                current_x, current_y = (
                    prediction[
                        "current_position"
                    ]
                )

                predicted_x, predicted_y = (
                    prediction[
                        "predicted_position"
                    ]
                )

                cv2.circle(
                    frame,
                    (
                        int(current_x),
                        int(current_y)
                    ),
                    6,
                    (255, 255, 255),
                    -1
                )

                cv2.circle(
                    frame,
                    (
                        int(predicted_x),
                        int(predicted_y)
                    ),
                    9,
                    (0, 255, 255),
                    2
                )

                cv2.line(
                    frame,
                    (
                        int(current_x),
                        int(current_y)
                    ),
                    (
                        int(predicted_x),
                        int(predicted_y)
                    ),
                    (0, 255, 255),
                    2
                )

                frame_center_x = int(
                    w / 2
                )

                frame_center_y = int(
                    h / 2
                )

                cv2.circle(
                    frame,
                    (
                        frame_center_x,
                        frame_center_y
                    ),
                    8,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    (
                        f"Predictive Track | "
                        f"ID {prediction['person_id']}"
                    ),
                    (
                        20,
                        165
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    frame,
                    (
                        f"PTZ: "
                        f"{prediction['pan_direction']} / "
                        f"{prediction['tilt_direction']}"
                    ),
                    (
                        20,
                        190
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    frame,
                    (
                        f"Speed: "
                        f"{prediction['speed']:.1f} px/frame"
                    ),
                    (
                        20,
                        215
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    frame,
                    "PREDICTED POSITION",
                    (
                        int(predicted_x) + 10,
                        int(predicted_y)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            else:

                cv2.putText(
                    frame,
                    "Predictive Track: NO TARGET",
                    (
                        20,
                        165
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (180, 180, 180),
                    2,
                    cv2.LINE_AA
                )

            # =================================================
            # OBJECT COUNT
            # =================================================

            object_count = len(
                visible_objects
            )

            cv2.putText(
                frame,
                f"Detected objects: {object_count}",
                (
                    20,
                    240
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (
                    255,
                    200,
                    0
                ),
                2
            )

            # =================================================
            # SECURITY STATUS
            # =================================================

            if security_enabled:

                security_status = (
                    "ADMIN SECURITY: ACTIVE"
                )

                security_color = (
                    0,
                    255,
                    0
                )

            else:

                security_status = (
                    "ADMIN SECURITY: OFFLINE"
                )

                security_color = (
                    0,
                    165,
                    255
                )

            cv2.putText(
                frame,
                security_status,
                (
                    20,
                    265
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                security_color,
                2
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
                (
                    255,
                    255,
                    255
                ),
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

        try:
            object_detector.close()
        except Exception:
            pass

        cv2.destroyAllWindows()

        print(
            "[INFO] Camera tracker stopped."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()