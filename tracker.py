import math
from collections import deque


class Track:
    def __init__(self, track_id, detection, now):
        self.id = track_id

        self.center = detection["center"]
        self.landmarks = detection["landmarks"]
        self.bbox = detection["bbox"]

        # =====================================================
        # ACTIVITY STATE
        # =====================================================

        self.state = "unknown"
        self.previous_state = "unknown"
        self.state_started = now

        # Candidate posture used for temporal smoothing
        self.candidate_state = None
        self.candidate_started = None

        # Recent posture classifications
        self.posture_history = deque(maxlen=8)

        # =====================================================
        # TRACKING
        # =====================================================

        self.last_seen = now
        self.missed = 0

        self.previous_center = self.center

        # =====================================================
        # BODY MOVEMENT
        # =====================================================

        self.previous_hip_y = None
        self.previous_hip_x = None

        self.body_movement = 0.0

        # Recent body movement history
        self.movement_history = deque(maxlen=12)

        # =====================================================
        # FALL DETECTION
        # =====================================================

        # Recent normalized hip positions:
        # (timestamp, normalized_hip_y)
        self.hip_history = deque(maxlen=12)

        # Recent torso angles:
        # (timestamp, torso_angle)
        self.torso_history = deque(maxlen=12)

        # Candidate fall start time
        self.fall_candidate_started = None

        # Last confirmed fall
        self.fall_alerted_at = None

        # Prevent repeated fall events
        self.fall_cooldown_until = 0.0

        # Individual fall signals
        self.fall_motion_detected = False
        self.fall_horizontal_detected = False
        self.fall_low_position_detected = False

        # =====================================================
        # ACTIVITY ALERTS
        # =====================================================

        self.sitting_alerted = False
        self.standing_alerted = False

        # =====================================================
        # ZONE TRACKING
        # =====================================================

        self.zone_started = None
        self.zone_alerted = False

        # Previous frame zone state
        self.was_in_zone = False

        # Prevent false entry/exit event on first frame
        self.zone_initialized = False

        # =====================================================
        # EVENT MANAGEMENT
        # =====================================================

        # Stores timestamps for event cooldowns.
        #
        # Example:
        # {
        #     "ZONE_ENTRY": 12345.0,
        #     "LIMITED_VIEW": 12350.0
        # }
        self.last_alerts = {}

    # =========================================================
    # UPDATE TRACK
    # =========================================================

    def update(self, detection, now):

        old_center = self.center

        self.center = detection["center"]
        self.landmarks = detection["landmarks"]
        self.bbox = detection["bbox"]

        self.previous_center = old_center

        self.last_seen = now
        self.missed = 0

        # -----------------------------------------------------
        # Calculate movement between frames
        # -----------------------------------------------------

        self.body_movement = math.hypot(
            self.center[0] - old_center[0],
            self.center[1] - old_center[1]
        )

        # Keep movement history for fall analysis.
        self.movement_history.append(
            (
                now,
                self.body_movement
            )
        )


class CentroidTracker:

    def __init__(
        self,
        max_distance=140,
        max_missed=20
    ):
        self.max_distance = max_distance
        self.max_missed = max_missed

        self.next_id = 1
        self.tracks = {}

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

    def update(
        self,
        detections,
        now
    ):

        # =====================================================
        # NO EXISTING TRACKS
        # =====================================================

        if not self.tracks:

            for detection in detections:

                self._add(
                    detection,
                    now
                )

            return list(
                self.tracks.values()
            )

        # =====================================================
        # INITIAL MATCHING STATE
        # =====================================================

        unmatched_tracks = set(
            self.tracks.keys()
        )

        unmatched_detections = set(
            range(len(detections))
        )

        pairs = []

        # =====================================================
        # CALCULATE TRACK ↔ DETECTION DISTANCES
        # =====================================================

        for tid, track in self.tracks.items():

            for i, detection in enumerate(
                detections
            ):

                distance = self.distance(
                    track.center,
                    detection["center"]
                )

                pairs.append(
                    (
                        distance,
                        tid,
                        i
                    )
                )

        # =====================================================
        # MATCH CLOSEST TRACK TO DETECTION
        # =====================================================

        for distance, tid, i in sorted(
            pairs
        ):

            # Because pairs are sorted by distance,
            # once we exceed the maximum distance we can stop.
            if distance > self.max_distance:
                break

            # Track or detection has already been matched.
            if (
                tid not in unmatched_tracks
                or
                i not in unmatched_detections
            ):
                continue

            # Update existing track.
            self.tracks[tid].update(
                detections[i],
                now
            )

            unmatched_tracks.remove(
                tid
            )

            unmatched_detections.remove(
                i
            )

        # =====================================================
        # ADD NEW PEOPLE
        # =====================================================

        for i in unmatched_detections:

            self._add(
                detections[i],
                now
            )

        # =====================================================
        # HANDLE MISSED TRACKS
        # =====================================================

        for tid in list(
            unmatched_tracks
        ):

            track = self.tracks[tid]

            track.missed += 1

            # Remove the track only after it has been
            # missing for the configured number of frames.
            if (
                track.missed
                >
                self.max_missed
            ):

                del self.tracks[
                    tid
                ]

        return list(
            self.tracks.values()
        )

    # =========================================================
    # ADD NEW TRACK
    # =========================================================

    def _add(
        self,
        detection,
        now
    ):

        self.tracks[
            self.next_id
        ] = Track(
            self.next_id,
            detection,
            now
        )

        self.next_id += 1