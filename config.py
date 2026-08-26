# ============================================================
# CAMERA
# ============================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


# ============================================================
# POSE DETECTION
# ============================================================

NUM_POSES = 4

MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5


# ============================================================
# FALL DETECTION
# ============================================================

# Torso angle at which the body is considered substantially
# horizontal.
FALL_ANGLE_DEG = 55.0

# Minimum normalized hip movement between frames that can
# indicate rapid downward movement.
FALL_DROP_RATIO = 0.035

# Minimum accumulated normalized hip movement over the
# fall sequence.
FALL_TOTAL_DROP_RATIO = 0.08

# Time that the fall candidate must persist before
# confirmation.
FALL_CONFIRM_SECONDS = 0.35

# Minimum number of historical frames required.
FALL_MIN_HISTORY_FRAMES = 4

# Strong horizontal body angle.
FALL_STRONG_TORSO_ANGLE = 70.0

# Prevent repeated fall captures.
FALL_ALERT_COOLDOWN_SECONDS = 15.0

# Existing general alert cooldown.
ALERT_COOLDOWN_SECONDS = 10.0


# ============================================================
# POSTURE CLASSIFICATION
# ============================================================

# Knee angle below this value indicates a bent knee /
# sitting posture.
SITTING_KNEE_ANGLE_DEG = 150.0

# Torso angle below this value is considered upright.
STANDING_TORSO_ANGLE_DEG = 35.0


# ============================================================
# DWELL TIMES
# ============================================================

# Sitting for 2 minutes.
SITTING_DWELL_SECONDS = 120.0

# Standing for 5 minutes.
STANDING_DWELL_SECONDS = 300.0

# Inside monitoring zone for 2 minutes.
ZONE_DWELL_SECONDS = 120.0


# ============================================================
# MONITORING ZONE
# ============================================================

# Normalized:
# x1, y1, x2, y2
ZONE = (
    0.60,
    0.10,
    0.98,
    0.95
)


# ============================================================
# TRACKING
# ============================================================

MAX_TRACK_DISTANCE_PX = 140

MAX_MISSED_FRAMES = 20