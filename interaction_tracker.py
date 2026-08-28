import math
from collections import defaultdict


class PersonObjectInteraction:
    """
    Associates detected objects with tracked people.

    The association is based on:
    1. Person/object proximity
    2. Bounding-box overlap
    3. Persistence across multiple frames
    4. Object movement relative to the associated person

    This does NOT claim physical ownership or pickup.
    It reports a probable interaction/association.
    """

    def __init__(
        self,
        interaction_distance=150,
        confirm_frames=5,
        release_frames=8,
        movement_threshold=12,
    ):
        self.interaction_distance = interaction_distance
        self.confirm_frames = confirm_frames
        self.release_frames = release_frames
        self.movement_threshold = movement_threshold

        # Key:
        # (person_id, object_index)
        #
        # Value:
        # {
        #     "frames": int,
        #     "lost_frames": int,
        #     "active": bool,
        #     "first_seen": float,
        #     "last_seen": float,
        #     "previous_object_center": tuple,
        #     "object_movement": float,
        # }
        self.relationships = {}

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
    # BBOX OVERLAP
    # =========================================================

    @staticmethod
    def bbox_iou(box_a, box_b):

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        intersection_x1 = max(ax1, bx1)
        intersection_y1 = max(ay1, by1)
        intersection_x2 = min(ax2, bx2)
        intersection_y2 = min(ay2, by2)

        intersection_width = max(
            0,
            intersection_x2 - intersection_x1,
        )

        intersection_height = max(
            0,
            intersection_y2 - intersection_y1,
        )

        intersection_area = (
            intersection_width
            *
            intersection_height
        )

        if intersection_area <= 0:
            return 0.0

        area_a = max(
            1,
            (ax2 - ax1)
            *
            (ay2 - ay1),
        )

        area_b = max(
            1,
            (bx2 - bx1)
            *
            (by2 - by1),
        )

        union_area = (
            area_a
            +
            area_b
            -
            intersection_area
        )

        if union_area <= 0:
            return 0.0

        return (
            intersection_area
            /
            union_area
        )

    # =========================================================
    # UPDATE
    # =========================================================

    def update(
        self,
        tracks,
        objects,
        now,
    ):
        """
        Update person-object relationships.

        Returns a list of currently confirmed interactions.
        """

        current_pairs = set()
        interactions = []

        visible_tracks = [
            track
            for track in tracks
            if not track.missed
        ]

        # =====================================================
        # FIND PERSON ↔ OBJECT RELATIONSHIPS
        # =====================================================

        for track in visible_tracks:

            person_id = track.id

            person_center = track.center
            person_bbox = track.bbox

            for object_index, obj in enumerate(objects):

                object_center = obj["center"]
                object_bbox = obj["bbox"]

                distance = self.distance(
                    person_center,
                    object_center,
                )

                overlap = self.bbox_iou(
                    person_bbox,
                    object_bbox,
                )

                # -------------------------------------------------
                # Interaction condition
                #
                # Either:
                #   object overlaps person
                #
                # OR:
                #   object is sufficiently close to person
                # -------------------------------------------------

                associated = (
                    overlap >= 0.02
                    or
                    distance <= self.interaction_distance
                )

                if not associated:
                    continue

                pair_key = (
                    person_id,
                    object_index,
                )

                current_pairs.add(
                    pair_key
                )

                # -------------------------------------------------
                # Existing relationship
                # -------------------------------------------------

                if pair_key not in self.relationships:

                    self.relationships[
                        pair_key
                    ] = {
                        "frames": 1,
                        "lost_frames": 0,
                        "active": False,
                        "first_seen": now,
                        "last_seen": now,
                        "previous_object_center": object_center,
                        "object_movement": 0.0,
                    }

                else:

                    relationship = (
                        self.relationships[
                            pair_key
                        ]
                    )

                    relationship["frames"] += 1
                    relationship["lost_frames"] = 0
                    relationship["last_seen"] = now

                    previous_center = (
                        relationship[
                            "previous_object_center"
                        ]
                    )

                    movement = self.distance(
                        previous_center,
                        object_center,
                    )

                    relationship[
                        "object_movement"
                    ] = movement

                    relationship[
                        "previous_object_center"
                    ] = object_center

                relationship = (
                    self.relationships[
                        pair_key
                    ]
                )

                # -------------------------------------------------
                # Confirm interaction after persistence
                # -------------------------------------------------

                if (
                    not relationship["active"]
                    and
                    relationship["frames"]
                    >= self.confirm_frames
                ):

                    relationship["active"] = True

                # -------------------------------------------------
                # Add confirmed interaction
                # -------------------------------------------------

                if relationship["active"]:

                    interactions.append(
                        {
                            "person_id": person_id,
                            "object_index": object_index,
                            "object": obj,
                            "distance": distance,
                            "overlap": overlap,
                            "object_movement": relationship[
                                "object_movement"
                            ],
                            "moving_with_person": (
                                relationship[
                                    "object_movement"
                                ]
                                >= self.movement_threshold
                            ),
                            "first_seen": relationship[
                                "first_seen"
                            ],
                            "last_seen": relationship[
                                "last_seen"
                            ],
                        }
                    )

        # =====================================================
        # HANDLE DISAPPEARED RELATIONSHIPS
        # =====================================================

        for pair_key in list(
            self.relationships.keys()
        ):

            if pair_key in current_pairs:
                continue

            relationship = (
                self.relationships[
                    pair_key
                ]
            )

            relationship["lost_frames"] += 1

            if (
                relationship["lost_frames"]
                >= self.release_frames
            ):

                del self.relationships[
                    pair_key
                ]

        return interactions

    # =========================================================
    # CURRENT INTERACTIONS FOR PERSON
    # =========================================================

    def get_person_interactions(
        self,
        person_id,
    ):

        results = []

        for (
            pair,
            relationship,
        ) in self.relationships.items():

            stored_person_id, _ = pair

            if stored_person_id != person_id:
                continue

            if not relationship["active"]:
                continue

            results.append(
                relationship
            )

        return results