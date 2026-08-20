import math


class Track:
    def __init__(self, track_id, detection, now):
        self.id = track_id
        self.center = detection["center"]
        self.landmarks = detection["landmarks"]
        self.bbox = detection["bbox"]
        self.state = "unknown"
        self.previous_state = "unknown"
        self.state_started = now
        self.last_seen = now
        self.missed = 0
        self.previous_hip_y = None
        self.fall_candidate_started = None
        self.fall_alerted_at = None
        self.sitting_alerted = False
        self.standing_alerted = False
        self.zone_started = None
        self.zone_alerted = False

    def update(self, detection, now):
        self.center = detection["center"]
        self.landmarks = detection["landmarks"]
        self.bbox = detection["bbox"]
        self.last_seen = now
        self.missed = 0


class CentroidTracker:
    def __init__(self, max_distance=140, max_missed=20):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks = {}

    @staticmethod
    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def update(self, detections, now):
        if not self.tracks:
            for d in detections:
                self._add(d, now)
            return list(self.tracks.values())

        unmatched_tracks = set(self.tracks.keys())
        unmatched_detections = set(range(len(detections)))
        pairs = []

        for tid, track in self.tracks.items():
            for i, d in enumerate(detections):
                pairs.append((self.distance(track.center, d["center"]), tid, i))

        for dist, tid, i in sorted(pairs):
            if dist > self.max_distance:
                break
            if tid not in unmatched_tracks or i not in unmatched_detections:
                continue
            self.tracks[tid].update(detections[i], now)
            unmatched_tracks.remove(tid)
            unmatched_detections.remove(i)

        for i in unmatched_detections:
            self._add(detections[i], now)

        for tid in list(unmatched_tracks):
            self.tracks[tid].missed += 1
            if self.tracks[tid].missed > self.max_missed:
                del self.tracks[tid]

        return list(self.tracks.values())

    def _add(self, detection, now):
        self.tracks[self.next_id] = Track(self.next_id, detection, now)
        self.next_id += 1
