import math
from collections import deque


class Track:
    def __init__(self, track_id, detection, now):
        self.id = track_id

        self.center = detection["center"]
        self.landmarks = detection["landmarks"]
        self.bbox = detection["bbox"]

        # =====================================================
        # Activity state
        # =====================================================

        self.state = "unknown"
        self.previous_state = "unknown"
        self.state_started = now

        # Candidate posture used for temporal smoothing
        self.candidate_state = None
        self.candidate_started = None

        # Store recent posture classifications
        self.posture_history = deque(maxlen=8)

        # =====================================================
        # Tracking
        # =====================================================

        self.last_seen = now
        self.missed = 0

        self.previous_center = self.center

        # =====================================================
        # Body movement
        # =====================================================

        self.previous_hip_y = None
        self.previous_hip_x = None

        self.body_movement = 0.0

        # =====================================================
        # Fall detection
        # =====================================================

        self.fall_candidate_started = None
        self.fall_alerted_at = None

        # =====================================================
        # Activity alerts
        # =====================================================

        self.sitting_alerted = False
        self.standing_alerted = False

        # =====================================================
        # Zone tracking
        # =====================================================

        self.zone_started = None
        self.zone_alerted = False

        self.was_in_zone = False
        self.zone_initialized = False

    def update(self, detection, now):

        old_center = self.center

        self.center = detection["center"]
        self.landmarks = detection["landmarks"]
        self.bbox = detection["bbox"]

        self.previous_center = old_center

        self.last_seen = now
        self.missed = 0

        # Calculate movement between frames
        self.body_movement = math.hypot(
            self.center[0] - old_center[0],
            self.center[1] - old_center[1]
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

    @staticmethod
    def distance(a, b):

        return math.hypot(
            a[0] - b[0],
            a[1] - b[1]
        )

    def update(
        self,
        detections,
        now
    ):

        # =====================================================
        # No existing tracks
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
        # Matching
        # =====================================================

        unmatched_tracks = set(
            self.tracks.keys()
        )

        unmatched_detections = set(
            range(len(detections))
        )

        pairs = []

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
        # Match closest track to detection
        # =====================================================

        for distance, tid, i in sorted(
            pairs
        ):

            if distance > self.max_distance:
                break

            if (
                tid not in unmatched_tracks
                or
                i not in unmatched_detections
            ):
                continue

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
        # Add new people
        # =====================================================

        for i in unmatched_detections:

            self._add(
                detections[i],
                now
            )

        # =====================================================
        # Handle missed tracks
        # =====================================================

        for tid in list(
            unmatched_tracks
        ):

            self.tracks[tid].missed += 1

            if (
                self.tracks[tid].missed
                >
                self.max_missed
            ):

                del self.tracks[tid]

        return list(
            self.tracks.values()
        )

    def _add(
        self,
        detection,
        now
    ):

        self.tracks[self.next_id] = Track(
            self.next_id,
            detection,
            now
        )

        self.next_id += 1     
