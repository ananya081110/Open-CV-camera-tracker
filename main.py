import csv
import math
import time
from datetime import datetime
from pathlib import Path

import cv2

import config
from database import init_db
from detector import PoseDetector
from download_model import MODEL_PATH
from event_manager import capture_event
from tracker import CentroidTracker


ROOT = Path(__file__).resolve().parent

LOG_DIR = ROOT / "logs"
ALERT_DIR = ROOT / "alerts"
EVENT_LOG = LOG_DIR / "events.csv"


# ============================================================
# MODEL / DIRECTORY SETUP
# ============================================================

def ensure_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Pose model not found. Run: python download_model.py"
        )


def ensure_dirs():
    LOG_DIR.mkdir(exist_ok=True)
    ALERT_DIR.mkdir(exist_ok=True)

    (ROOT / "captured" / "fall").mkdir(
        parents=True,
        exist_ok=True
    )

    (ROOT / "captured" / "manual").mkdir(
        parents=True,
        exist_ok=True
    )

    if not EVENT_LOG.exists():
        with EVENT_LOG.open("w", newline="") as f:
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

    with EVENT_LOG.open("a", newline="") as f:
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


# ============================================================
# ZONE DETECTION
# ============================================================

def inside_zone(center, w, h):
    x1, y1, x2, y2 = config.ZONE

    return (
        x1 * w <= center[0] <= x2 * w
        and
        y1 * h <= center[1] <= y2 * h
    )


# ============================================================
# BODY VISIBILITY
# ============================================================

def landmark_valid(point):
    """
    Check whether a landmark contains a usable coordinate.
    """

    if point is None:
        return False

    try:
        x, y = point

        if not math.isfinite(float(x)):
            return False

        if not math.isfinite(float(y)):
            return False

        if x < 0 or y < 0:
            return False

        return True

    except (TypeError, ValueError):
        return False


def check_lower_body_visibility(points, frame_height):
    """
    Determine whether enough lower-body landmarks are
    available to make a reliable posture decision.

    MediaPipe landmarks:
        23 = left hip
        24 = right hip
        25 = left knee
        26 = right knee
        27 = left ankle
        28 = right ankle
    """

    required_indices = [
        23,
        24,
        25,
        26,
        27,
        28
    ]

    valid_points = 0

    for idx in required_indices:

        if idx >= len(points):
            continue

        if landmark_valid(points[idx]):
            valid_points += 1

    # Require hips + at least one complete leg
    hips_visible = (
        landmark_valid(points[23])
        and
        landmark_valid(points[24])
    )

    left_leg_visible = (
        landmark_valid(points[25])
        and
        landmark_valid(points[27])
    )

    right_leg_visible = (
        landmark_valid(points[26])
        and
        landmark_valid(points[28])
    )

    if (
        hips_visible
        and
        (
            left_leg_visible
            or
            right_leg_visible
        )
    ):
        return True

    return False


# ============================================================
# ACTIVITY CLASSIFICATION
# ============================================================

def classify(
    torso_angle,
    knee_angle,
    lower_body_visible
):
    """
    Classify the person's activity.

    If the lower body isn't sufficiently visible,
    do NOT make an unreliable standing/sitting/fall
    classification.
    """

    # --------------------------------------------------------
    # LIMITED VIEW
    # --------------------------------------------------------

    if not lower_body_visible:
        return "limited_view"

    # --------------------------------------------------------
    # FALL
    # --------------------------------------------------------

    if torso_angle >= 55:
        return "falling"

    # --------------------------------------------------------
    # SITTING
    # --------------------------------------------------------

    if knee_angle < 165:
        return "sitting"

    # --------------------------------------------------------
    # STANDING
    # --------------------------------------------------------

    if (
        torso_angle < 35
        and
        knee_angle >= 165
    ):
        return "standing"

    # --------------------------------------------------------
    # WALKING / OTHER
    # --------------------------------------------------------

    return "walking"


# ============================================================
# TIME FORMAT
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


# ============================================================
# SKELETON DRAWING
# ============================================================

def draw_skeleton(frame, points):

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
            not landmark_valid(points[a])
            or
            not landmark_valid(points[b])
        ):
            continue

        cv2.line(
            frame,
            points[a],
            points[b],
            (0, 255, 0),
            2
        )

    for point in points:

        if landmark_valid(point):

            cv2.circle(
                frame,
                point,
                3,
                (255, 255, 255),
                -1
            )


# ============================================================
# SAVE EVENT
# ============================================================

def save_event(
    frame,
    track_id,
    event_type,
    activity,
    details,
    category
):
    """
    Save image + SQLite event + CSV event.
    """

    try:

        event_id, image_path = capture_event(
            frame=frame,
            person_id=track_id,
            event_type=event_type,
            activity=activity,
            details=details,
            category=category
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

    except Exception as e:

        print(
            f"[ERROR] Could not save event: {e}"
        )

        return (
            None,
            None
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------

    ensure_model()
    ensure_dirs()

    init_db()

    detector = PoseDetector(
        MODEL_PATH
    )

    tracker = CentroidTracker(
        config.MAX_TRACK_DISTANCE_PX,
        config.MAX_MISSED_FRAMES
    )

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        config.CAMERA_INDEX
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Camera could not be opened. "
            "Check macOS camera permission."
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        config.FRAME_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        config.FRAME_HEIGHT
    )

    timestamp_ms = 0

    previous = time.monotonic()

    # ========================================================
    # CAMERA LOOP
    # ========================================================

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            # ------------------------------------------------
            # TIME
            # ------------------------------------------------

            now = time.monotonic()

            timestamp_ms += max(
                1,
                int(
                    (now - previous) * 1000
                )
            )

            previous = now

            h, w = frame.shape[:2]

            # ------------------------------------------------
            # AI POSE DETECTION
            # ------------------------------------------------

            detections = detector.detect(
                frame,
                timestamp_ms
            )

            # ------------------------------------------------
            # PERSON TRACKING
            # ------------------------------------------------

            tracks = tracker.update(
                detections,
                now
            )

            # =================================================
            # MONITORING ZONE
            # =================================================

            zx1, zy1, zx2, zy2 = config.ZONE

            cv2.rectangle(
                frame,

                (
                    int(zx1 * w),
                    int(zy1 * h)
                ),

                (
                    int(zx2 * w),
                    int(zy2 * h)
                ),

                (255, 180, 0),

                2
            )

            cv2.putText(
                frame,

                "MONITORING ZONE",

                (
                    int(zx1 * w),
                    max(
                        25,
                        int(zy1 * h) - 8
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (255, 180, 0),

                2
            )

            # =================================================
            # PROCESS EACH TRACK
            # =================================================

            for track in tracks:

                if track.missed:
                    continue

                pts = track.landmarks

                # ------------------------------------------------
                # BASIC LANDMARK SAFETY
                # ------------------------------------------------

                if len(pts) < 33:
                    continue

                # ------------------------------------------------
                # SHOULDER CENTER
                # ------------------------------------------------

                if (
                    not landmark_valid(pts[11])
                    or
                    not landmark_valid(pts[12])
                ):
                    continue

                shoulder = (
                    (
                        pts[11][0]
                        +
                        pts[12][0]
                    ) / 2,

                    (
                        pts[11][1]
                        +
                        pts[12][1]
                    ) / 2
                )

                # ------------------------------------------------
                # HIP CENTER
                # ------------------------------------------------

                if (
                    not landmark_valid(pts[23])
                    or
                    not landmark_valid(pts[24])
                ):
                    continue

                hip = (
                    (
                        pts[23][0]
                        +
                        pts[24][0]
                    ) / 2,

                    (
                        pts[23][1]
                        +
                        pts[24][1]
                    ) / 2
                )

                # =================================================
                # TORSO ANGLE
                # =================================================

                torso_angle = abs(
                    math.degrees(
                        math.atan2(
                            hip[0]
                            -
                            shoulder[0],

                            hip[1]
                            -
                            shoulder[1]
                        )
                    )
                )

                # =================================================
                # LOWER BODY VISIBILITY
                # =================================================

                lower_body_visible = (
                    check_lower_body_visibility(
                        pts,
                        h
                    )
                )

                # =================================================
                # KNEE ANGLE
                # =================================================

                if lower_body_visible:

                    left_knee_angle = angle(
                        pts[23],
                        pts[25],
                        pts[27]
                    )

                    right_knee_angle = angle(
                        pts[24],
                        pts[26],
                        pts[28]
                    )

                    knee_angle = (
                        left_knee_angle
                        +
                        right_knee_angle
                    ) / 2

                else:

                    # Don't use unreliable knee data
                    knee_angle = 180.0

                # =================================================
                # ACTIVITY CLASSIFICATION
                # =================================================

                state = classify(
                    torso_angle,
                    knee_angle,
                    lower_body_visible
                )

                # =================================================
                # STATE CHANGE
                # =================================================

                if state != track.state:

                    track.previous_state = (
                        track.state
                    )

                    track.state = state

                    track.state_started = now

                    if state != "sitting":
                        track.sitting_alerted = False

                    if state != "standing":
                        track.standing_alerted = False

                # =================================================
                # HIP MOVEMENT
                # =================================================

                hip_y = hip[1] / h

                hip_drop = (
                    hip_y
                    -
                    track.previous_hip_y

                    if track.previous_hip_y
                    is not None

                    else 0
                )

                track.previous_hip_y = hip_y

                # =================================================
                # FALL DETECTION
                # =================================================
                #
                # IMPORTANT:
                # Fall detection is only performed when
                # the lower body is visible.
                #
                # This prevents an upper-body-only camera
                # view from producing unreliable fall alerts.
                # =================================================

                if lower_body_visible:

                    if (
                        track.previous_state
                        in (
                            "standing",
                            "walking"
                        )

                        and

                        state == "falling"

                        and

                        hip_drop
                        >=
                        config.FALL_DROP_RATIO
                    ):

                        if (
                            track.fall_candidate_started
                            is None
                        ):

                            track.fall_candidate_started = now

                else:

                    track.fall_candidate_started = None

                # ------------------------------------------------
                # RESET FALL CANDIDATE
                # ------------------------------------------------

                if state != "falling":

                    track.fall_candidate_started = None

                # =================================================
                # CONFIRMED FALL
                # =================================================

                if (
                    lower_body_visible

                    and

                    track.fall_candidate_started
                    is not None

                    and

                    now
                    -
                    track.fall_candidate_started
                    >=
                    config.FALL_CONFIRM_SECONDS

                    and

                    (
                        track.fall_alerted_at
                        is None

                        or

                        now
                        -
                        track.fall_alerted_at
                        >=
                        config.ALERT_COOLDOWN_SECONDS
                    )
                ):

                    details = (
                        f"torso_angle="
                        f"{torso_angle:.1f}, "

                        f"hip_drop="
                        f"{hip_drop:.3f}"
                    )

                    # ------------------------------------------------
                    # SAVE FALL IMAGE + DATABASE EVENT
                    # ------------------------------------------------

                    save_event(
                        frame=frame,

                        track_id=track.id,

                        event_type=
                        "FALL_DETECTED",

                        activity="falling",

                        details=details,

                        category="fall"
                    )

                    track.fall_alerted_at = now

                    track.fall_candidate_started = None

                # =================================================
                # STATE ELAPSED TIME
                # =================================================

                elapsed = (
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

                    elapsed
                    >=
                    config.SITTING_DWELL_SECONDS
                ):

                    if not track.sitting_alerted:

                        details = (
                            f"{fmt(elapsed)} "
                            "sitting"
                        )

                        save_event(
                            frame=frame,

                            track_id=track.id,

                            event_type=
                            "SITTING_DWELL_THRESHOLD",

                            activity="sitting",

                            details=details,

                            category="events"
                        )

                        track.sitting_alerted = True

                # =================================================
                # STANDING DWELL
                # =================================================

                if (
                    state == "standing"

                    and

                    elapsed
                    >=
                    config.STANDING_DWELL_SECONDS
                ):

                    if not track.standing_alerted:

                        details = (
                            f"{fmt(elapsed)} "
                            "standing"
                        )

                        save_event(
                            frame=frame,

                            track_id=track.id,

                            event_type=
                            "STANDING_DWELL_THRESHOLD",

                            activity="standing",

                            details=details,

                            category="events"
                        )

                        track.standing_alerted = True

                # =================================================
                # ZONE DETECTION
                # =================================================

                in_zone = inside_zone(
                    track.center,
                    w,
                    h
                )

                if in_zone:

                    if track.zone_started is None:

                        track.zone_started = now

                    zone_elapsed = (
                        now
                        -
                        track.zone_started
                    )

                    if (
                        zone_elapsed
                        >=
                        config.ZONE_DWELL_SECONDS
                    ):

                        if not track.zone_alerted:

                            details = (
                                f"{fmt(zone_elapsed)} "
                                "in zone"
                            )

                            save_event(
                                frame=frame,

                                track_id=track.id,

                                event_type=
                                "ZONE_DWELL_THRESHOLD",

                                activity=state,

                                details=details,

                                category="events"
                            )

                            track.zone_alerted = True

                else:

                    track.zone_started = None

                    track.zone_alerted = False

                    zone_elapsed = 0

                # =================================================
                # DRAW PERSON
                # =================================================

                x1, y1, x2, y2 = map(
                    int,
                    track.bbox
                )

                # ------------------------------------------------
                # DISPLAY STATE
                # ------------------------------------------------

                if state == "limited_view":

                    display_state = "LIMITED VIEW"

                else:

                    display_state = state.upper()

                label = (
                    f"ID {track.id} | "
                    f"{display_state} | "
                    f"{fmt(elapsed)}"
                )

                if in_zone:

                    label += (
                        f" | Zone "
                        f"{fmt(zone_elapsed)}"
                    )

                # ------------------------------------------------
                # BOUNDING BOX
                # ------------------------------------------------

                if state == "limited_view":

                    box_color = (
                        0,
                        165,
                        255
                    )

                elif state == "falling":

                    box_color = (
                        0,
                        0,
                        255
                    )

                else:

                    box_color = (
                        0,
                        220,
                        0
                    )

                cv2.rectangle(
                    frame,

                    (x1, y1),

                    (x2, y2),

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

                # ------------------------------------------------
                # DRAW SKELETON
                # ------------------------------------------------

                draw_skeleton(
                    frame,
                    pts
                )

                # =================================================
                # LIMITED VIEW WARNING
                # =================================================

                if state == "limited_view":

                    cv2.putText(
                        frame,

                        "LIMITED VIEW - FULL BODY REQUIRED",

                        (25, 40),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.7,

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

                        (25, 40),

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

                (20, h - 20),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (255, 255, 255),

                2
            )

            # =================================================
            # DISPLAY
            # =================================================

            cv2.imshow(
                "AI Camera Tracker - V3",
                frame
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            # =================================================
            # MANUAL IMAGE CAPTURE
            # =================================================

            if key == ord("c"):

                save_event(
                    frame=frame,

                    track_id=0,

                    event_type=
                    "MANUAL_CAPTURE",

                    activity="unknown",

                    details=
                    "Manual snapshot captured",

                    category="manual"
                )

            # =================================================
            # QUIT
            # =================================================

            if key == ord("q"):
                break

    finally:

        cap.release()

        detector.close()

        cv2.destroyAllWindows()


# ============================================================
# ANGLE CALCULATION
# ============================================================

def angle(a, b, c):

    import numpy as np

    a, b, c = map(
        lambda p:
        np.array(
            p,
            dtype=float
        ),

        (a, b, c)
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

    cosine = np.clip(
        np.dot(ba, bc)
        /
        denom,

        -1,

        1
    )

    return math.degrees(
        math.acos(cosine)
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":
    main()