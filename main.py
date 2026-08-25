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
# POSTURE SETTINGS
# ============================================================

# Minimum number of lower-body landmarks required.
MIN_LOWER_BODY_POINTS = 4

# Knee angle thresholds.
SITTING_KNEE_ANGLE = 155
STANDING_KNEE_ANGLE = 168

# Torso angle.
UPRIGHT_TORSO_ANGLE = 32

# Minimum number of frames required before changing
# between normal posture states.
POSTURE_CONFIRM_FRAMES = 5

# Movement threshold for walking.
WALKING_MOVEMENT_THRESHOLD = 8


# ============================================================
# MODEL / DIRECTORY SETUP
# ============================================================

def ensure_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Pose model not found. Run: python download_model.py"
        )


def ensure_dirs():

    LOG_DIR.mkdir(
        exist_ok=True
    )

    ALERT_DIR.mkdir(
        exist_ok=True
    )

    (ROOT / "captured" / "fall").mkdir(
        parents=True,
        exist_ok=True
    )

    (ROOT / "captured" / "manual").mkdir(
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


# ============================================================
# ZONE
# ============================================================

def inside_zone(
    center,
    w,
    h
):

    x1, y1, x2, y2 = config.ZONE

    return (
        x1 * w <= center[0] <= x2 * w
        and
        y1 * h <= center[1] <= y2 * h
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


# ============================================================
# BODY VISIBILITY
# ============================================================

def get_body_visibility(points):

    upper_indices = [
        11, 12, 13, 14, 15, 16
    ]

    lower_indices = [
        23, 24,
        25, 26,
        27, 28
    ]

    upper_visible = sum(
        landmark_valid(points[i])
        for i in upper_indices
    )

    lower_visible = sum(
        landmark_valid(points[i])
        for i in lower_indices
    )

    return (
        upper_visible,
        lower_visible
    )


def check_lower_body_visibility(points):

    """
    Lower body must contain enough landmarks
    before we attempt sitting/standing classification.
    """

    required = [
        23, 24,
        25, 26,
        27, 28
    ]

    visible = sum(
        landmark_valid(points[i])
        for i in required
    )

    if visible >= MIN_LOWER_BODY_POINTS:

        hips_visible = (
            landmark_valid(points[23])
            and
            landmark_valid(points[24])
        )

        if hips_visible:
            return True

    return False


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def distance(a, b):

    if (
        not landmark_valid(a)
        or
        not landmark_valid(b)
    ):

        return 0

    return math.hypot(
        a[0] - b[0],
        a[1] - b[1]
    )


def calculate_torso_angle(
    shoulder,
    hip
):

    return abs(
        math.degrees(
            math.atan2(
                hip[0] - shoulder[0],
                hip[1] - shoulder[1]
            )
        )
    )


def angle(
    a,
    b,
    c
):

    import numpy as np

    if not all(
        landmark_valid(p)
        for p in (a, b, c)
    ):

        return 180.0

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
# POSTURE CLASSIFICATION
# ============================================================

def classify_posture(
    points,
    track,
    frame_height
):

    """
    Improved posture classifier.

    IMPORTANT:
    If lower body isn't visible, we do NOT guess standing.

    We return limited_view instead.
    """

    if len(points) < 33:

        return (
            "limited_view",
            0,
            180
        )

    upper_visible, lower_visible = (
        get_body_visibility(points)
    )

    # --------------------------------------------------------
    # Upper body visibility
    # --------------------------------------------------------

    if upper_visible < 4:

        return (
            "limited_view",
            lower_visible,
            180
        )

    # --------------------------------------------------------
    # Lower body visibility
    # --------------------------------------------------------

    if not check_lower_body_visibility(
        points
    ):

        return (
            "limited_view",
            lower_visible,
            180
        )

    # --------------------------------------------------------
    # Hip / shoulder
    # --------------------------------------------------------

    if (
        not landmark_valid(points[11])
        or
        not landmark_valid(points[12])
        or
        not landmark_valid(points[23])
        or
        not landmark_valid(points[24])
    ):

        return (
            "limited_view",
            lower_visible,
            180
        )

    shoulder = (
        (
            points[11][0]
            +
            points[12][0]
        ) / 2,

        (
            points[11][1]
            +
            points[12][1]
        ) / 2
    )

    hip = (
        (
            points[23][0]
            +
            points[24][0]
        ) / 2,

        (
            points[23][1]
            +
            points[24][1]
        ) / 2
    )

    torso_angle = calculate_torso_angle(
        shoulder,
        hip
    )

    # --------------------------------------------------------
    # Knee angles
    # --------------------------------------------------------

    left_knee = angle(
        points[23],
        points[25],
        points[27]
    )

    right_knee = angle(
        points[24],
        points[26],
        points[28]
    )

    valid_knees = []

    if landmark_valid(points[25]) and landmark_valid(points[27]):
        valid_knees.append(left_knee)

    if landmark_valid(points[26]) and landmark_valid(points[28]):
        valid_knees.append(right_knee)

    if not valid_knees:

        return (
            "limited_view",
            lower_visible,
            180
        )

    knee_angle = sum(
        valid_knees
    ) / len(
        valid_knees
    )

    # --------------------------------------------------------
    # Leg geometry
    # --------------------------------------------------------

    left_leg = 0
    right_leg = 0

    if (
        landmark_valid(points[25])
        and
        landmark_valid(points[27])
    ):

        left_leg = distance(
            points[25],
            points[27]
        )

    if (
        landmark_valid(points[26])
        and
        landmark_valid(points[28])
    ):

        right_leg = distance(
            points[26],
            points[28]
        )

    leg_length = max(
        left_leg,
        right_leg
    )

    torso_length = distance(
        shoulder,
        hip
    )

    # --------------------------------------------------------
    # Fall
    # --------------------------------------------------------

    if torso_angle >= 55:

        return (
            "falling",
            lower_visible,
            knee_angle
        )

    # --------------------------------------------------------
    # Sitting
    # --------------------------------------------------------

    sitting_score = 0

    # Bent knees strongly indicate sitting.
    if knee_angle < SITTING_KNEE_ANGLE:

        sitting_score += 3

    elif knee_angle < 165:

        sitting_score += 1

    # Sitting usually creates a shorter visible
    # lower-body extension than standing.
    if torso_length > 0 and leg_length > 0:

        leg_torso_ratio = (
            leg_length /
            torso_length
        )

        if leg_torso_ratio < 1.0:

            sitting_score += 1

    # Hip and knee should be relatively close
    # vertically in a seated posture.
    if (
        landmark_valid(points[25])
        and
        landmark_valid(points[26])
    ):

        knee_y = (
            points[25][1]
            +
            points[26][1]
        ) / 2

        vertical_gap = abs(
            knee_y - hip[1]
        )

        if torso_length > 0:

            normalized_gap = (
                vertical_gap /
                torso_length
            )

            if normalized_gap < 0.9:

                sitting_score += 1

    if sitting_score >= 3:

        return (
            "sitting",
            lower_visible,
            knee_angle
        )

    # --------------------------------------------------------
    # Standing
    # --------------------------------------------------------

    standing_score = 0

    # Straight knees
    if knee_angle >= STANDING_KNEE_ANGLE:

        standing_score += 2

    # Upright torso
    if torso_angle < UPRIGHT_TORSO_ANGLE:

        standing_score += 1

    # Long legs
    if torso_length > 0 and leg_length > 0:

        leg_torso_ratio = (
            leg_length /
            torso_length
        )

        if leg_torso_ratio >= 1.0:

            standing_score += 1

    if standing_score >= 3:

        # Movement can indicate walking instead.
        if track.body_movement >= WALKING_MOVEMENT_THRESHOLD:

            return (
                "walking",
                lower_visible,
                knee_angle
            )

        return (
            "standing",
            lower_visible,
            knee_angle
        )

    # --------------------------------------------------------
    # Walking / uncertain
    # --------------------------------------------------------

    if (
        track.body_movement
        >=
        WALKING_MOVEMENT_THRESHOLD
    ):

        return (
            "walking",
            lower_visible,
            knee_angle
        )

    return (
        "uncertain",
        lower_visible,
        knee_angle
    )


# ============================================================
# TEMPORAL POSTURE SMOOTHING
# ============================================================

def stabilize_posture(
    track,
    detected_state,
    now
):

    """
    Prevents one or two bad frames from immediately
    changing the person's state.
    """

    # Fall is handled immediately.
    if detected_state == "falling":

        return "falling"

    # Limited view should be immediate.
    if detected_state == "limited_view":

        track.candidate_state = None
        track.candidate_started = None

        return "limited_view"

    # Uncertain should not force a posture change.
    if detected_state == "uncertain":

        return track.state if track.state not in (
            "unknown",
            "limited_view"
        ) else "uncertain"

    # Same as current state.
    if detected_state == track.state:

        track.candidate_state = None
        track.candidate_started = None

        return track.state

    # New candidate posture.
    if track.candidate_state != detected_state:

        track.candidate_state = detected_state
        track.candidate_started = now

        track.posture_history.clear()

    track.posture_history.append(
        detected_state
    )

    # Require repeated consistent frames.
    if (
        len(track.posture_history)
        >= POSTURE_CONFIRM_FRAMES
    ):

        recent = list(
            track.posture_history
        )[-POSTURE_CONFIRM_FRAMES:]

        if all(
            state == detected_state
            for state in recent
        ):

            track.candidate_state = None
            track.candidate_started = None
            track.posture_history.clear()

            return detected_state

    # Keep previous stable state.
    if track.state != "unknown":

        return track.state

    return detected_state


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
# SKELETON
# ============================================================

def draw_skeleton(
    frame,
    points
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

    # ========================================================
    # CAMERA
    # ========================================================

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

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            now = time.monotonic()

            timestamp_ms += max(
                1,
                int(
                    (now - previous)
                    * 1000
                )
            )

            previous = now

            h, w = frame.shape[:2]

            # =================================================
            # DETECTION
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
            # PROCESS PEOPLE
            # =================================================

            for track in tracks:

                if track.missed:
                    continue

                pts = track.landmarks

                if len(pts) < 33:
                    continue

                # =================================================
                # CLASSIFY POSTURE
                # =================================================

                detected_state, lower_visible_count, knee_angle = (
                    classify_posture(
                        pts,
                        track,
                        h
                    )
                )

                state = stabilize_posture(
                    track,
                    detected_state,
                    now
                )

                # =================================================
                # SHOULDERS / HIPS
                # =================================================

                if (
                    landmark_valid(pts[11])
                    and
                    landmark_valid(pts[12])
                    and
                    landmark_valid(pts[23])
                    and
                    landmark_valid(pts[24])
                ):

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

                    torso_angle = (
                        calculate_torso_angle(
                            shoulder,
                            hip
                        )
                    )

                else:

                    hip = None
                    torso_angle = 0

                # =================================================
                # STATE CHANGE
                # =================================================

                if state != track.state:

                    track.previous_state = track.state
                    track.state = state
                    track.state_started = now

                    if state != "sitting":

                        track.sitting_alerted = False

                    if state != "standing":

                        track.standing_alerted = False

                # =================================================
                # HIP MOVEMENT / FALL
                # =================================================

                hip_drop = 0

                if hip is not None:

                    hip_y = (
                        hip[1] / h
                    )

                    if (
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
                    detected_state == "falling"
                    and
                    track.previous_state
                    in (
                        "standing",
                        "walking"
                    )
                    and
                    hip_drop >= config.FALL_DROP_RATIO
                ):

                    if (
                        track.fall_candidate_started
                        is None
                    ):

                        track.fall_candidate_started = now

                if detected_state != "falling":

                    track.fall_candidate_started = None

                # =================================================
                # CONFIRMED FALL
                # =================================================

                if (
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

                    save_event(
                        frame=frame,
                        track_id=track.id,
                        event_type="FALL_DETECTED",
                        activity="falling",
                        details=details,
                        category="fall"
                    )

                    track.fall_alerted_at = now
                    track.fall_candidate_started = None

                # =================================================
                # ELAPSED STATE TIME
                # =================================================

                elapsed = (
                    now
                    -
                    track.state_started
                )

                # =================================================
                # SITTING ALERT
                # =================================================

                if (
                    state == "sitting"
                    and
                    elapsed
                    >=
                    config.SITTING_DWELL_SECONDS
                ):

                    if not track.sitting_alerted:

                        save_event(
                            frame=frame,
                            track_id=track.id,
                            event_type="SITTING_DWELL_THRESHOLD",
                            activity="sitting",
                            details=(
                                f"{fmt(elapsed)} sitting"
                            ),
                            category="events"
                        )

                        track.sitting_alerted = True

                # =================================================
                # STANDING ALERT
                # =================================================

                if (
                    state == "standing"
                    and
                    elapsed
                    >=
                    config.STANDING_DWELL_SECONDS
                ):

                    if not track.standing_alerted:

                        save_event(
                            frame=frame,
                            track_id=track.id,
                            event_type="STANDING_DWELL_THRESHOLD",
                            activity="standing",
                            details=(
                                f"{fmt(elapsed)} standing"
                            ),
                            category="events"
                        )

                        track.standing_alerted = True

                # =================================================
                # ZONE
                # =================================================

                in_zone = inside_zone(
                    track.center,
                    w,
                    h
                )

                if not track.zone_initialized:

                    track.was_in_zone = in_zone
                    track.zone_initialized = True

                else:

                    if (
                        not track.was_in_zone
                        and
                        in_zone
                    ):

                        save_event(
                            frame=frame,
                            track_id=track.id,
                            event_type="ZONE_ENTRY",
                            activity=state,
                            details=(
                                "Person entered "
                                "monitoring zone"
                            ),
                            category="zone"
                        )

                    elif (
                        track.was_in_zone
                        and
                        not in_zone
                    ):

                        save_event(
                            frame=frame,
                            track_id=track.id,
                            event_type="ZONE_EXIT",
                            activity=state,
                            details=(
                                "Person exited "
                                "monitoring zone"
                            ),
                            category="zone"
                        )

                    track.was_in_zone = in_zone

                # =================================================
                # ZONE DWELL
                # =================================================

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

                            save_event(
                                frame=frame,
                                track_id=track.id,
                                event_type="ZONE_DWELL_THRESHOLD",
                                activity=state,
                                details=(
                                    f"{fmt(zone_elapsed)} "
                                    "in zone"
                                ),
                                category="events"
                            )

                            track.zone_alerted = True

                else:

                    track.zone_started = None
                    track.zone_alerted = False
                    zone_elapsed = 0

                # =================================================
                # DISPLAY
                # =================================================

                x1, y1, x2, y2 = map(
                    int,
                    track.bbox
                )

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

                # =================================================
                # BOX COLOR
                # =================================================

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

                elif state == "sitting":

                    box_color = (
                        255,
                        0,
                        255
                    )

                elif state == "uncertain":

                    box_color = (
                        0,
                        200,
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

                # =================================================
                # SKELETON
                # =================================================

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
                        "LIMITED VIEW - MOVE CAMERA / SHOW LOWER BODY",
                        (25, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 165, 255),
                        2
                    )

                # =================================================
                # UNCERTAIN WARNING
                # =================================================

                elif state == "uncertain":

                    cv2.putText(
                        frame,
                        "POSTURE UNCERTAIN",
                        (25, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 200, 255),
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
                "AI Camera Tracker - V4",
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

                save_event(
                    frame=frame,
                    track_id=0,
                    event_type="MANUAL_CAPTURE",
                    activity="unknown",
                    details="Manual snapshot captured",
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
# START
# ============================================================

if __name__ == "__main__":

    main()