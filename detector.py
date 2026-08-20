import math

import cv2
import mediapipe as mp
import numpy as np

from config import (
    NUM_POSES,
    MIN_DETECTION_CONFIDENCE,
    MIN_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


def midpoint(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def angle(a, b, c):
    a, b, c = map(lambda p: np.array(p, dtype=float), (a, b, c))

    ba = a - b
    bc = c - b

    denom = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denom == 0:
        return 180.0

    cosine = np.clip(
        np.dot(ba, bc) / denom,
        -1.0,
        1.0
    )

    return math.degrees(math.acos(cosine))


class PoseDetector:

    def __init__(self, model_path):

        # IMPORTANT:
        # Force MediaPipe to use CPU instead of the
        # Metal/GPU backend on macOS.
        base_options = BaseOptions(
            model_asset_path=str(model_path),
            delegate=BaseOptions.Delegate.CPU,
        )

        options = PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,
            num_poses=NUM_POSES,
            min_pose_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )

        self.landmarker = PoseLandmarker.create_from_options(options)

    def detect(self, frame_bgr, timestamp_ms):

        height, width = frame_bgr.shape[:2]

        # Convert OpenCV BGR → RGB
        rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.landmarker.detect_for_video(
            image,
            timestamp_ms
        )

        detections = []

        for pose in result.pose_landmarks:

            points = [
                (
                    int(p.x * width),
                    int(p.y * height)
                )
                for p in pose
            ]

            if len(points) < 33:
                continue

            # Make sure enough landmarks are visible
            visible = [
                p
                for p in pose
                if p.visibility is None
                or p.visibility >= 0.35
            ]

            if len(visible) < 8:
                continue

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            x1 = max(0, min(xs))
            x2 = min(width - 1, max(xs))

            y1 = max(0, min(ys))
            y2 = min(height - 1, max(ys))

            # Shoulder midpoint
            shoulder = midpoint(
                points[11],
                points[12]
            )

            # Hip midpoint
            hip = midpoint(
                points[23],
                points[24]
            )

            # Person center
            center = (
                (x1 + x2) / 2,
                (y1 + y2) / 2
            )

            # Knee angles
            left_knee_angle = angle(
                points[23],
                points[25],
                points[27]
            )

            right_knee_angle = angle(
                points[24],
                points[26],
                points[28]
            )

            knee_angle = (
                left_knee_angle +
                right_knee_angle
            ) / 2

            # Torso angle
            # 0° ≈ upright
            # 90° ≈ horizontal
            torso_angle = abs(
                math.degrees(
                    math.atan2(
                        hip[0] - shoulder[0],
                        hip[1] - shoulder[1]
                    )
                )
            )

            detections.append(
                {
                    "center": center,

                    "landmarks": points,

                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),

                    "shoulder": shoulder,

                    "hip": hip,

                    "knee_angle": knee_angle,

                    "torso_angle": torso_angle,
                }
            )

        return detections

    def close(self):
        self.landmarker.close()