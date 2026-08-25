import math
from collections import deque


class Track:
    def __init__(self, track_id, detection, now):
        self.id = track_id
        self.center = detection["center"]
        self.landmarks = detection["landmarks"]
        self.bbox = detection["bbox"]

        # -----------------------------------------
        # Activity state
        # -----------------------------------------
        self.state = "unknown"
        self.previous_state = "unknown"
        self.state_started = now

        # Candidate state used for smoothing.
        # A state must remain consistent for several
        # frames before becoming the actual state.
        self.candidate_state = "unknown"
        self.candidate_started = now

        # Keep recent classifications for temporal smoothing.
        self.state_history = deque(maxlen=12)

        # -----------------------------------------
        # Tracking
        # -----------------------------------------
        self.last_seen = now
        self.missed = 0

        # -----------------------------------------
        # Fall detection
        # -----------------------------------------
        self.previous_hip_y = None
        self.fall_candidate_started = None
        self.fall_alerted_at = None

        # -----------------------------------------
        # Activity alerts
        # -----------------------------------------
        self.sitting_alerted = False
        self.standing_alerted = False

        # -----------------------------------------
        # Zone tracking
        # -----------------------------------------
        self.zone_started = None
        self.zone_alerted = False

        self.was_in_zone = False
        self.zone_initialized = False

        # -----------------------------------------
        # Pose quality / visibility
        # -----------------------------------------
        self.lower_body_visible = False
        self.upper_body_visible = False

        # Number of consecutive frames where the
        # lower body has been poorly detected.
        self.limited_view_frames = 0

    def update(self, detection, now):
        self.center = detection["center"]
        self.landmarks = detection["landmarks"]
        self.bbox = detection["bbox"]

        self.last_seen = now
        self.missed = 0

    # -----------------------------------------
    # Activity smoothing
    # -----------------------------------------

    def add_activity(self, state):
        """
        Add a newly detected state to the history.

        This prevents one noisy frame from immediately
        changing the person's activity.
        """

        self.state_history.append(state)

    def stable_activity(self):
        """
        Return the most common recent activity.

        Limited-view and unknown are handled separately
        because they should not overwrite a reliable
        activity too aggressively.
        """

        if not self.state_history:
            return self.state

        valid_states = [
            s for s in self.state_history
            if s not in ("unknown",)
        ]

        if not valid_states:
            return self.state

        counts = {}

        for state in valid_states:
            counts[state] = counts.get(state, 0) + 1

        return max(
            counts,
            key=counts.get
        )

    # -----------------------------------------
    # State transition
    # -----------------------------------------

    def update_state(self, new_state, now):
        """
        Update the actual state only when the new
        state is sufficiently consistent.
        """

        if new_state == self.state:
            self.candidate_state = new_state
            self.candidate_started = now
            return False

        # If this is a completely new candidate,
        # start timing it.
        if new_state != self.candidate_state:
            self.candidate_state = new_state
            self.candidate_started = now
            return False

        return False


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
        # -----------------------------------------
        # No existing tracks
        # -----------------------------------------

        if not self.tracks:

            for detection in detections:
                self._add(
                    detection,
                    now
                )

            return list(
                self.tracks.values()
            )

        # -----------------------------------------
        # Initially everything is unmatched
        # -----------------------------------------

        unmatched_tracks = set(
            self.tracks.keys()
        )

        unmatched_detections = set(
            range(len(detections))
        )

        pairs = []

        # -----------------------------------------
        # Calculate distances
        # -----------------------------------------

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

        # -----------------------------------------
        # Match closest track ↔ detection
        # -----------------------------------------

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

        # -----------------------------------------
        # Add new detections
        # -----------------------------------------

        for i in unmatched_detections:

            self._add(
                detections[i],
                now
            )

        # -----------------------------------------
        # Handle missed tracks
        # -----------------------------------------

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