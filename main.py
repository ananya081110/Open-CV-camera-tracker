import csv
import math
import time
from datetime import datetime
from pathlib import Path

import cv2

import config
from detector import PoseDetector
from download_model import MODEL_PATH
from tracker import CentroidTracker

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
ALERT_DIR = ROOT / "alerts"
EVENT_LOG = LOG_DIR / "events.csv"


def ensure_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Pose model not found. Run: python download_model.py"
        )


def ensure_dirs():
    LOG_DIR.mkdir(exist_ok=True)
    ALERT_DIR.mkdir(exist_ok=True)
    if not EVENT_LOG.exists():
        with EVENT_LOG.open("w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "track_id", "event", "details"]
            )


def log_event(track_id, event, details):
    stamp = datetime.now().isoformat(timespec="seconds")
    with EVENT_LOG.open("a", newline="") as f:
        csv.writer(f).writerow([stamp, track_id, event, details])
    print(f"[ALERT] {stamp} | Person {track_id} | {event} | {details}")


def inside_zone(center, w, h):
    x1, y1, x2, y2 = config.ZONE
    return (
        x1 * w <= center[0] <= x2 * w
        and y1 * h <= center[1] <= y2 * h
    )


def classify(torso_angle, knee_angle):
    if torso_angle >= config.FALL_ANGLE_DEG:
        return "falling"
    if knee_angle < config.SITTING_KNEE_ANGLE_DEG and torso_angle < 55:
        return "sitting"
    if torso_angle < config.STANDING_TORSO_ANGLE_DEG:
        return "standing"
    return "walking"


def fmt(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def draw_skeleton(frame, points):
    connections = [
        (11,12),(11,13),(13,15),(12,14),(14,16),
        (11,23),(12,24),(23,24),(23,25),(25,27),
        (24,26),(26,28),(27,29),(29,31),(28,30),(30,32)
    ]
    for a, b in connections:
        cv2.line(frame, points[a], points[b], (0, 255, 0), 2)
    for p in points:
        cv2.circle(frame, p, 3, (255, 255, 255), -1)


def main():
    ensure_model()
    ensure_dirs()

    detector = PoseDetector(MODEL_PATH)
    tracker = CentroidTracker(
        config.MAX_TRACK_DISTANCE_PX,
        config.MAX_MISSED_FRAMES,
    )

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            "Camera could not be opened. Check macOS camera permission."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    timestamp_ms = 0
    previous = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.monotonic()
            timestamp_ms += max(1, int((now - previous) * 1000))
            previous = now

            h, w = frame.shape[:2]
            detections = detector.detect(frame, timestamp_ms)
            tracks = tracker.update(detections, now)

            zx1, zy1, zx2, zy2 = config.ZONE
            cv2.rectangle(
                frame,
                (int(zx1*w), int(zy1*h)),
                (int(zx2*w), int(zy2*h)),
                (255, 180, 0), 2
            )
            cv2.putText(
                frame, "MONITORING ZONE",
                (int(zx1*w), max(25, int(zy1*h)-8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,180,0), 2
            )

            for track in tracks:
                if track.missed:
                    continue

                pts = track.landmarks
                shoulder = ((pts[11][0] + pts[12][0]) / 2,
                            (pts[11][1] + pts[12][1]) / 2)
                hip = ((pts[23][0] + pts[24][0]) / 2,
                       (pts[23][1] + pts[24][1]) / 2)

                torso_angle = abs(math.degrees(
                    math.atan2(hip[0]-shoulder[0], hip[1]-shoulder[1])
                ))

                knee_angle = (
                    angle(pts[23], pts[25], pts[27]) +
                    angle(pts[24], pts[26], pts[28])
                ) / 2

                state = classify(torso_angle, knee_angle)

                if state != track.state:
                    track.previous_state = track.state
                    track.state = state
                    track.state_started = now
                    if state != "sitting":
                        track.sitting_alerted = False
                    if state != "standing":
                        track.standing_alerted = False

                hip_y = hip[1] / h
                hip_drop = (
                    hip_y - track.previous_hip_y
                    if track.previous_hip_y is not None else 0
                )
                track.previous_hip_y = hip_y

                if (
                    track.previous_state in ("standing", "walking")
                    and state == "falling"
                    and hip_drop >= config.FALL_DROP_RATIO
                ):
                    if track.fall_candidate_started is None:
                        track.fall_candidate_started = now

                if state != "falling":
                    track.fall_candidate_started = None

                if (
                    track.fall_candidate_started is not None
                    and now - track.fall_candidate_started >= config.FALL_CONFIRM_SECONDS
                    and (
                        track.fall_alerted_at is None
                        or now - track.fall_alerted_at >= config.ALERT_COOLDOWN_SECONDS
                    )
                ):
                    snapshot = ALERT_DIR / (
                        f"fall_person_{track.id}_"
                        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    )
                    cv2.imwrite(str(snapshot), frame)
                    log_event(
                        track.id, "FALL_DETECTED",
                        f"torso_angle={torso_angle:.1f}, hip_drop={hip_drop:.3f}"
                    )
                    track.fall_alerted_at = now
                    track.fall_candidate_started = None

                elapsed = now - track.state_started

                if state == "sitting" and elapsed >= config.SITTING_DWELL_SECONDS:
                    if not track.sitting_alerted:
                        log_event(
                            track.id, "SITTING_DWELL_THRESHOLD",
                            f"{fmt(elapsed)} sitting"
                        )
                        track.sitting_alerted = True

                if state == "standing" and elapsed >= config.STANDING_DWELL_SECONDS:
                    if not track.standing_alerted:
                        log_event(
                            track.id, "STANDING_DWELL_THRESHOLD",
                            f"{fmt(elapsed)} standing"
                        )
                        track.standing_alerted = True

                in_zone = inside_zone(track.center, w, h)
                if in_zone:
                    if track.zone_started is None:
                        track.zone_started = now
                    zone_elapsed = now - track.zone_started
                    if zone_elapsed >= config.ZONE_DWELL_SECONDS:
                        if not track.zone_alerted:
                            log_event(
                                track.id, "ZONE_DWELL_THRESHOLD",
                                f"{fmt(zone_elapsed)} in zone"
                            )
                            track.zone_alerted = True
                else:
                    track.zone_started = None
                    track.zone_alerted = False
                    zone_elapsed = 0

                x1, y1, x2, y2 = map(int, track.bbox)
                label = f"ID {track.id} | {state.upper()} | {fmt(elapsed)}"
                if in_zone:
                    label += f" | Zone {fmt(zone_elapsed)}"

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,220,0), 2)
                cv2.putText(
                    frame, label, (x1, max(25,y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2
                )
                draw_skeleton(frame, pts)

                if state == "falling":
                    cv2.putText(
                        frame, "POSSIBLE FALL - CONFIRMING",
                        (25, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,0,255), 3
                    )

            cv2.putText(
                frame, "Q = quit",
                (20, h-20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255,255,255), 2
            )
            cv2.imshow("AI Camera Tracker - POC", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


def angle(a, b, c):
    import numpy as np
    a, b, c = map(lambda p: np.array(p, dtype=float), (a,b,c))
    ba, bc = a-b, c-b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 180.0
    return math.degrees(math.acos(np.clip(np.dot(ba,bc)/denom, -1, 1)))


if __name__ == "__main__":
    main()
