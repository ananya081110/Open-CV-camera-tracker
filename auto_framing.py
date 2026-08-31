import math
from collections import deque


class PredictiveAutoFramer:
    """
    Software-based PTZ / Auto-Framing controller.

    This does NOT physically move the camera.
    It calculates a virtual camera window that follows
    the selected person.

    Features:
    - Person position tracking
    - Velocity estimation
    - Future position prediction
    - Smooth virtual pan/tilt
    - Automatic digital zoom
    - Target locking
    """

    def __init__(
        self,
        frame_width,
        frame_height,
        crop_ratio=0.65,
        smoothing=0.15,
        prediction_frames=8,
        min_zoom=1.0,
        max_zoom=2.0,
    ):

        self.frame_width = frame_width
        self.frame_height = frame_height

        self.crop_ratio = crop_ratio
        self.smoothing = smoothing
        self.prediction_frames = prediction_frames

        self.min_zoom = min_zoom
        self.max_zoom = max_zoom

        self.target_id = None

        self.center_x = frame_width / 2
        self.center_y = frame_height / 2

        self.zoom = min_zoom

        self.previous_position = None

        self.position_history = deque(
            maxlen=8
        )

    # =========================================================
    # SET TARGET
    # =========================================================

    def set_target(self, person_id):

        self.target_id = person_id

    # =========================================================
    # CLEAR TARGET
    # =========================================================

    def clear_target(self):

        self.target_id = None
        self.previous_position = None
        self.position_history.clear()

    # =========================================================
    # DISTANCE
    # =========================================================

    @staticmethod
    def distance(a, b):

        return math.hypot(
            a[0] - b[0],
            a[1] - b[1],
        )

    # =========================================================
    # UPDATE
    # =========================================================

    def update(self, tracks):

        target = None

        # -----------------------------------------------------
        # Find selected person
        # -----------------------------------------------------

        if self.target_id is not None:

            for track in tracks:

                if track.id == self.target_id:

                    if track.missed == 0:
                        target = track

                    break

        # -----------------------------------------------------
        # If no target selected, choose first visible person
        # -----------------------------------------------------

        if target is None:

            visible_tracks = [
                track
                for track in tracks
                if track.missed == 0
            ]

            if not visible_tracks:

                return None

            target = visible_tracks[0]

            self.target_id = target.id

        # -----------------------------------------------------
        # Current position
        # -----------------------------------------------------

        current_x = float(
            target.center[0]
        )

        current_y = float(
            target.center[1]
        )

        current_position = (
            current_x,
            current_y,
        )

        self.position_history.append(
            current_position
        )

        # =====================================================
        # VELOCITY
        # =====================================================

        velocity_x = 0.0
        velocity_y = 0.0

        if self.previous_position is not None:

            velocity_x = (
                current_x
                -
                self.previous_position[0]
            )

            velocity_y = (
                current_y
                -
                self.previous_position[1]
            )

        self.previous_position = (
            current_x,
            current_y,
        )

        # =====================================================
        # PREDICT FUTURE POSITION
        # =====================================================

        predicted_x = (
            current_x
            +
            velocity_x
            *
            self.prediction_frames
        )

        predicted_y = (
            current_y
            +
            velocity_y
            *
            self.prediction_frames
        )

        predicted_x = max(
            0,
            min(
                self.frame_width,
                predicted_x,
            )
        )

        predicted_y = max(
            0,
            min(
                self.frame_height,
                predicted_y,
            )
        )

        # =====================================================
        # SMOOTH VIRTUAL CAMERA MOVEMENT
        # =====================================================

        self.center_x += (
            predicted_x
            -
            self.center_x
        ) * self.smoothing

        self.center_y += (
            predicted_y
            -
            self.center_y
        ) * self.smoothing

        # =====================================================
        # AUTO ZOOM
        # =====================================================

        x1, y1, x2, y2 = target.bbox

        person_width = max(
            1,
            x2 - x1
        )

        person_height = max(
            1,
            y2 - y1
        )

        person_size = max(
            person_width,
            person_height,
        )

        frame_reference = min(
            self.frame_width,
            self.frame_height,
        )

        # Desired person size in the virtual frame.
        desired_size = (
            frame_reference * 0.35
        )

        if person_size < desired_size:

            zoom_change = (
                desired_size
                /
                person_size
            )

            desired_zoom = min(
                self.max_zoom,
                max(
                    self.min_zoom,
                    zoom_change,
                ),
            )

        else:

            desired_zoom = self.min_zoom

        self.zoom += (
            desired_zoom
            -
            self.zoom
        ) * self.smoothing

        self.zoom = max(
            self.min_zoom,
            min(
                self.max_zoom,
                self.zoom,
            )
        )

        # =====================================================
        # RETURN TRACKING INFORMATION
        # =====================================================

        speed = math.hypot(
            velocity_x,
            velocity_y,
        )

        return {
            "person_id": target.id,

            "current_position": (
                current_x,
                current_y,
            ),

            "predicted_position": (
                predicted_x,
                predicted_y,
            ),

            "velocity": (
                velocity_x,
                velocity_y,
            ),

            "speed": speed,

            "camera_center": (
                self.center_x,
                self.center_y,
            ),

            "zoom": self.zoom,
        }

    # =========================================================
    # APPLY VIRTUAL PTZ
    # =========================================================

    def apply(self, frame):

        if frame is None:
            return frame

        zoom = max(
            self.min_zoom,
            min(
                self.max_zoom,
                self.zoom,
            )
        )

        # -----------------------------------------------------
        # Calculate crop dimensions
        # -----------------------------------------------------

        crop_width = int(
            self.frame_width / zoom
        )

        crop_height = int(
            self.frame_height / zoom
        )

        crop_width = max(
            1,
            min(
                self.frame_width,
                crop_width,
            )
        )

        crop_height = max(
            1,
            min(
                self.frame_height,
                crop_height,
            )
        )

        # -----------------------------------------------------
        # Center crop around virtual camera position
        # -----------------------------------------------------

        x1 = int(
            self.center_x
            -
            crop_width / 2
        )

        y1 = int(
            self.center_y
            -
            crop_height / 2
        )

        x1 = max(
            0,
            min(
                self.frame_width
                -
                crop_width,
                x1,
            )
        )

        y1 = max(
            0,
            min(
                self.frame_height
                -
                crop_height,
                y1,
            )
        )

        x2 = (
            x1
            +
            crop_width
        )

        y2 = (
            y1
            +
            crop_height
        )

        cropped = frame[
            y1:y2,
            x1:x2
        ]

        if cropped.size == 0:
            return frame

        # -----------------------------------------------------
        # Resize back to original frame size
        # -----------------------------------------------------

        import cv2

        output = cv2.resize(
            cropped,
            (
                self.frame_width,
                self.frame_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        return output