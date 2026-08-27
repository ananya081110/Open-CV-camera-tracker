import math
from collections import deque


# ============================================================
# RE-ID CONFIGURATION
# ============================================================

# How long a disappeared person's identity is remembered.
# Example: if Person 1 leaves and comes back within 120 seconds,
# the tracker can attempt to restore Person 1.
REID_MEMORY_SECONDS = 120.0

# Minimum similarity required before restoring an old ID.
# Higher = safer but harder to match.
REID_MIN_SCORE = 0.68

# Maximum distance at which a returning person can be
# considered for re-identification.
#
# This is intentionally larger than normal tracking distance
# because the person may re-enter from another part of the
# camera view.
REID_MAX_DISTANCE_MULTIPLIER = 5.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def bbox_center(bbox):
    """
    Calculate the center of a bounding box.

    bbox:
        x1, y1, x2, y2
    """

    x1, y1, x2, y2 = bbox

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    )


def bbox_dimensions(bbox):

    x1, y1, x2, y2 = bbox

    width = max(
        1.0,
        x2 - x1
    )

    height = max(
        1.0,
        y2 - y1
    )

    return width, height


def midpoint(a, b):

    return (
        (a[0] + b[0]) / 2.0,
        (a[1] + b[1]) / 2.0
    )


def point_distance(a, b):

    return math.hypot(
        a[0] - b[0],
        a[1] - b[1]
    )


# ============================================================
# BODY SHAPE SIGNATURE
# ============================================================

def build_body_signature(landmarks):
    """
    Build a normalized body-shape signature from MediaPipe
    pose landmarks.

    This is NOT facial recognition.

    It uses relative body geometry so that the same person
    can potentially be recognized after leaving and returning.
    """

    if not landmarks:
        return None

    # MediaPipe Pose landmark indices.
    #
    # 11 = left shoulder
    # 12 = right shoulder
    # 23 = left hip
    # 24 = right hip
    # 25 = left knee
    # 26 = right knee
    # 27 = left ankle
    # 28 = right ankle
    required_indices = [
        11,
        12,
        23,
        24,
        25,
        26,
        27,
        28
    ]

    if len(landmarks) <= max(
        required_indices
    ):
        return None

    points = []

    for index in required_indices:

        point = landmarks[index]

        if point is None:
            return None

        try:

            x = float(point[0])
            y = float(point[1])

        except (
            TypeError,
            ValueError,
            IndexError
        ):

            return None

        points.append(
            (x, y)
        )

    left_shoulder = points[0]
    right_shoulder = points[1]

    left_hip = points[2]
    right_hip = points[3]

    left_knee = points[4]
    right_knee = points[5]

    left_ankle = points[6]
    right_ankle = points[7]

    shoulder_center = midpoint(
        left_shoulder,
        right_shoulder
    )

    hip_center = midpoint(
        left_hip,
        right_hip
    )

    # Torso length provides a scale-independent reference.
    torso_length = point_distance(
        shoulder_center,
        hip_center
    )

    if torso_length < 0.001:
        return None

    signature = []

    # --------------------------------------------------------
    # Normalized landmark positions
    # --------------------------------------------------------

    for point in points:

        normalized_x = (
            point[0]
            -
            shoulder_center[0]
        ) / torso_length

        normalized_y = (
            point[1]
            -
            shoulder_center[1]
        ) / torso_length

        signature.extend([
            normalized_x,
            normalized_y
        ])

    # --------------------------------------------------------
    # Body proportions
    # --------------------------------------------------------

    shoulder_width = (
        point_distance(
            left_shoulder,
            right_shoulder
        )
        /
        torso_length
    )

    hip_width = (
        point_distance(
            left_hip,
            right_hip
        )
        /
        torso_length
    )

    left_leg_length = (
        point_distance(
            left_hip,
            left_knee
        )
        +
        point_distance(
            left_knee,
            left_ankle
        )
    ) / torso_length

    right_leg_length = (
        point_distance(
            right_hip,
            right_knee
        )
        +
        point_distance(
            right_knee,
            right_ankle
        )
    ) / torso_length

    signature.extend([
        shoulder_width,
        hip_width,
        left_leg_length,
        right_leg_length
    ])

    return tuple(
        signature
    )


def compare_body_signatures(
    signature_a,
    signature_b
):
    """
    Compare two normalized body signatures.

    Returns:
        0.0 = completely different
        1.0 = very similar
    """

    if (
        signature_a is None
        or
        signature_b is None
    ):
        return 0.0

    if len(signature_a) != len(
        signature_b
    ):
        return 0.0

    total_difference = 0.0

    for a, b in zip(
        signature_a,
        signature_b
    ):

        total_difference += abs(
            a - b
        )

    average_difference = (
        total_difference
        /
        len(signature_a)
    )

    # Convert distance into similarity.
    similarity = math.exp(
        -2.5 * average_difference
    )

    return max(
        0.0,
        min(
            1.0,
            similarity
        )
    )


# ============================================================
# TRACK
# ============================================================

class Track:

    def __init__(
        self,
        track_id,
        detection,
        now
    ):

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
        self.posture_history = deque(
            maxlen=8
        )

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
        self.movement_history = deque(
            maxlen=12
        )

        # =====================================================
        # FALL DETECTION
        # =====================================================

        # Recent normalized hip positions:
        # (timestamp, normalized_hip_y)
        self.hip_history = deque(
            maxlen=12
        )

        # Recent torso angles:
        # (timestamp, torso_angle)
        self.torso_history = deque(
            maxlen=12
        )

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

        # =====================================================
        # NEW: RE-IDENTIFICATION
        # =====================================================

        # Normalized body-shape signature.
        self.body_signature = (
            build_body_signature(
                self.landmarks
            )
        )

        # Indicates that this track has been restored from
        # the re-identification memory.
        self.reidentified = False

        # Number of times this persistent ID has been restored.
        self.reidentification_count = 0

        # Last time this identity was restored.
        self.last_reidentified_at = None

    # =========================================================
    # UPDATE TRACK
    # =========================================================

    def update(
        self,
        detection,
        now
    ):

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

        # -----------------------------------------------------
        # NEW: UPDATE BODY SIGNATURE
        # -----------------------------------------------------

        new_signature = (
            build_body_signature(
                self.landmarks
            )
        )

        if new_signature is not None:

            if self.body_signature is None:

                self.body_signature = (
                    new_signature
                )

            else:

                # Smooth the identity signature rather than
                # replacing it every frame.
                self.body_signature = tuple(
                    (
                        0.75 * old
                        +
                        0.25 * new
                    )
                    for old, new in zip(
                        self.body_signature,
                        new_signature
                    )
                )


# ============================================================
# CENTROID TRACKER
# ============================================================

class CentroidTracker:

    def __init__(
        self,
        max_distance=140,
        max_missed=20
    ):

        self.max_distance = (
            max_distance
        )

        self.max_missed = (
            max_missed
        )

        self.next_id = 1

        self.tracks = {}

        # =====================================================
        # NEW: LOST TRACK / RE-ID MEMORY
        # =====================================================

        # Tracks that have disappeared are kept here instead
        # of being immediately destroyed.
        #
        # Example:
        #
        # {
        #     1: Track(...),
        #     4: Track(...)
        # }
        #
        # If Person 1 comes back later, the tracker can restore
        # ID 1 instead of creating ID 5.
        self.lost_tracks = {}

        # How long to remember a person after disappearance.
        self.reid_memory_seconds = (
            REID_MEMORY_SECONDS
        )

    # =========================================================
    # DISTANCE
    # =========================================================

    @staticmethod
    def distance(
        a,
        b
    ):

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
        # CLEAN EXPIRED RE-ID MEMORY
        # =====================================================

        self._cleanup_lost_tracks(
            now
        )

        # =====================================================
        # NO EXISTING ACTIVE TRACKS
        # =====================================================

        if not self.tracks:

            unmatched_detections = set(
                range(
                    len(detections)
                )
            )

            # -------------------------------------------------
            # FIRST TRY RE-ID
            # -------------------------------------------------

            for i in list(
                unmatched_detections
            ):

                restored_track = (
                    self._try_reidentify(
                        detections[i],
                        now
                    )
                )

                if restored_track is not None:

                    unmatched_detections.remove(
                        i
                    )

            # -------------------------------------------------
            # CREATE NEW TRACKS
            # -------------------------------------------------

            for i in unmatched_detections:

                self._add(
                    detections[i],
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
            range(
                len(detections)
            )
        )

        pairs = []

        # =====================================================
        # CALCULATE TRACK ↔ DETECTION DISTANCES
        # =====================================================

        for tid, track in (
            self.tracks.items()
        ):

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
        # NEW: RE-ID UNMATCHED DETECTIONS
        # =====================================================

        # At this point:
        #
        # - matched detections = existing people
        # - unmatched detections = either new people OR
        #   people returning after disappearing
        #
        # Try re-identification before assigning new IDs.

        for i in list(
            unmatched_detections
        ):

            restored_track = (
                self._try_reidentify(
                    detections[i],
                    now
                )
            )

            if restored_track is not None:

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

            # -------------------------------------------------
            # IMPORTANT:
            #
            # We preserve the existing max_missed behaviour.
            #
            # The track continues existing normally for
            # max_missed frames.
            #
            # Only after that do we move it into RE-ID memory
            # rather than deleting it permanently.
            # -------------------------------------------------

            if (
                track.missed
                >
                self.max_missed
            ):

                # Move to lost identity memory.
                self.lost_tracks[
                    tid
                ] = track

                del self.tracks[
                    tid
                ]

        return list(
            self.tracks.values()
        )

    # =========================================================
    # TRY RE-IDENTIFICATION
    # =========================================================

    def _try_reidentify(
        self,
        detection,
        now
    ):

        if not self.lost_tracks:

            return None

        best_track_id = None

        best_score = 0.0

        # -----------------------------------------------------
        # Detection information
        # -----------------------------------------------------

        detection_center = (
            detection["center"]
        )

        detection_bbox = (
            detection["bbox"]
        )

        detection_landmarks = (
            detection["landmarks"]
        )

        detection_signature = (
            build_body_signature(
                detection_landmarks
            )
        )

        # -----------------------------------------------------
        # Maximum spatial distance for RE-ID
        # -----------------------------------------------------

        reid_max_distance = (
            self.max_distance
            *
            REID_MAX_DISTANCE_MULTIPLIER
        )

        # -----------------------------------------------------
        # Compare against lost identities
        # -----------------------------------------------------

        for track_id, lost_track in (
            self.lost_tracks.items()
        ):

            # -------------------------------------------------
            # Time validity
            # -------------------------------------------------

            lost_duration = (
                now
                -
                lost_track.last_seen
            )

            if (
                lost_duration
                >
                self.reid_memory_seconds
            ):

                continue

            # -------------------------------------------------
            # Position similarity
            # -------------------------------------------------

            position_distance = (
                self.distance(
                    lost_track.center,
                    detection_center
                )
            )

            if (
                position_distance
                >
                reid_max_distance
            ):

                # If someone reappears extremely far away,
                # don't use position as a match.
                spatial_score = 0.0

            else:

                spatial_score = max(
                    0.0,
                    1.0
                    -
                    (
                        position_distance
                        /
                        reid_max_distance
                    )
                )

            # -------------------------------------------------
            # Body-shape similarity
            # -------------------------------------------------

            body_score = (
                compare_body_signatures(
                    lost_track.body_signature,
                    detection_signature
                )
            )

            # -------------------------------------------------
            # Bounding-box shape similarity
            # -------------------------------------------------

            old_width, old_height = (
                bbox_dimensions(
                    lost_track.bbox
                )
            )

            new_width, new_height = (
                bbox_dimensions(
                    detection_bbox
                )
            )

            old_ratio = (
                old_width
                /
                old_height
            )

            new_ratio = (
                new_width
                /
                new_height
            )

            ratio_difference = abs(
                old_ratio
                -
                new_ratio
            )

            bbox_score = math.exp(
                -2.0
                *
                ratio_difference
            )

            # -------------------------------------------------
            # Combined RE-ID score
            # -------------------------------------------------

            #
            # Body shape:
            #     60%
            #
            # Bounding-box shape:
            #     20%
            #
            # Position:
            #     20%
            #

            score = (
                0.60 * body_score
                +
                0.20 * bbox_score
                +
                0.20 * spatial_score
            )

            # -------------------------------------------------
            # Keep best match
            # -------------------------------------------------

            if score > best_score:

                best_score = score

                best_track_id = track_id

        # =====================================================
        # CHECK THRESHOLD
        # =====================================================

        if (
            best_track_id is None
            or
            best_score < REID_MIN_SCORE
        ):

            return None

        # =====================================================
        # RESTORE ORIGINAL TRACK
        # =====================================================

        track = self.lost_tracks.pop(
            best_track_id
        )

        # Update it with the new detection.
        track.update(
            detection,
            now
        )

        # Restore state.
        track.reidentified = True

        track.reidentification_count += 1

        track.last_reidentified_at = now

        # Add back to active tracks using the ORIGINAL ID.
        self.tracks[
            track.id
        ] = track

        print(
            "[RE-ID] Restored "
            f"Person ID {track.id} "
            f"(score={best_score:.2f})"
        )

        return track

    # =========================================================
    # CLEAN LOST TRACKS
    # =========================================================

    def _cleanup_lost_tracks(
        self,
        now
    ):

        expired_ids = []

        for track_id, track in (
            self.lost_tracks.items()
        ):

            lost_duration = (
                now
                -
                track.last_seen
            )

            if (
                lost_duration
                >
                self.reid_memory_seconds
            ):

                expired_ids.append(
                    track_id
                )

        for track_id in expired_ids:

            print(
                "[RE-ID] Identity memory expired "
                f"for Person ID {track_id}"
            )

            del self.lost_tracks[
                track_id
            ]

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