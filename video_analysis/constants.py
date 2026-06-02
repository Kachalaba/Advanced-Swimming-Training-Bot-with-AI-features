"""
Shared constants for video analysis modules.
Centralizes magic numbers to avoid duplication and ease tuning.
"""

# ---------------------------------------------------------------------------
# Pool / general swimming
# ---------------------------------------------------------------------------
POOL_LENGTH_METERS = 25.0

# ---------------------------------------------------------------------------
# Stroke analysis thresholds
# ---------------------------------------------------------------------------
STROKE_MIN_DURATION_SEC = 0.3
STROKE_MAX_DURATION_SEC = 5.0

HAND_ENTRY_ANGLE_MIN = 10  # degrees — valid range lower bound
HAND_ENTRY_ANGLE_MAX = 80  # degrees — valid range upper bound
OPTIMAL_HAND_ENTRY_ANGLE = 40  # degrees — biomechanically ideal

ELBOW_ANGLE_MIN = 60  # degrees — valid elbow catch range
ELBOW_ANGLE_MAX = 180  # degrees

ELBOW_SCORE_OPTIMAL_MIN = 90  # degrees — full score band
ELBOW_SCORE_OPTIMAL_MAX = 120
ELBOW_SCORE_GOOD_MIN = 80  # degrees — reduced-score band
ELBOW_SCORE_GOOD_MAX = 130
ELBOW_SCORE_GOOD_PCT = 80  # score value for good-but-not-optimal
ELBOW_SCORE_REFERENCE = 105  # degrees — centre of scoring curve

BREATHING_THRESHOLD_PX = 30  # pixels — lateral head offset for breath
BODY_ROLL_MAX_VALID = 90  # degrees — filter outliers
KICK_AMPLITUDE_SCALE = 45  # scale factor: normalised amplitude → degrees

# Scoring multipliers (score = 100 − deviation × multiplier)
HAND_ENTRY_SCORE_MULTIPLIER = 2
HEAD_STABILITY_MULTIPLIER = 2
BREATHING_REGULARITY_MULTIPLIER = 2
KICK_SYMMETRY_MULTIPLIER = 3

# ---------------------------------------------------------------------------
# Running analysis thresholds
# ---------------------------------------------------------------------------
RUN_PHASE_THRESHOLD_PX = 50  # pixels — stance/flight detection

GROUND_CONTACT_THRESHOLD_PX = 50  # pixels — ankle below hip for contact
GROUND_CONTACT_MIN_MS = 50  # ms — filter noise
GROUND_CONTACT_MAX_MS = 500  # ms — filter noise

HEEL_STRIKE_ANGLE_THRESHOLD = 10  # degrees — foot angle → heel strike
SEVERE_HEEL_STRIKE_THRESHOLD = 15  # degrees — adds injury risk
FOREFOOT_STRIKE_ANGLE_THRESHOLD = -5  # degrees — foot angle → forefoot

OVERSTRIDING_THRESHOLD_PX = -30  # pixels — foot ahead of CoM
LONG_CONTACT_TIME_MS = 300  # ms — adds injury risk
ARM_CROSSOVER_PCT_THRESHOLD = 30  # % crossover frames — flag detected

HIP_DROP_HIGH_THRESHOLD = 5  # degrees — adds injury risk
HIP_DROP_MULTIPLIER = 5  # score = 100 − drop × multiplier
BOUNCE_MULTIPLIER = 0.5  # score = 100 − osc × multiplier

EFFICIENCY_COMPONENT_WEIGHT = 0.2  # weight for each of 5 efficiency components

# Injury risk points per issue
INJURY_RISK_OVERSTRIDING = 30
INJURY_RISK_HIP_DROP = 25
INJURY_RISK_HEEL_STRIKE = 20
INJURY_RISK_LONG_CONTACT = 15
INJURY_RISK_ARM_CROSSOVER = 10

# ---------------------------------------------------------------------------
# MediaPipe Pose landmark indices (BlazePose 33-keypoint model)
# ---------------------------------------------------------------------------
MEDIAPIPE_NOSE = 0
MEDIAPIPE_LEFT_EAR = 7
MEDIAPIPE_RIGHT_EAR = 8
MEDIAPIPE_LEFT_SHOULDER = 11
MEDIAPIPE_RIGHT_SHOULDER = 12
MEDIAPIPE_LEFT_ELBOW = 13
MEDIAPIPE_RIGHT_ELBOW = 14
MEDIAPIPE_LEFT_WRIST = 15
MEDIAPIPE_RIGHT_WRIST = 16
MEDIAPIPE_LEFT_HIP = 23
MEDIAPIPE_RIGHT_HIP = 24
MEDIAPIPE_LEFT_KNEE = 25
MEDIAPIPE_RIGHT_KNEE = 26
MEDIAPIPE_LEFT_ANKLE = 27
MEDIAPIPE_RIGHT_ANKLE = 28

# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
MIN_DETECTION_CONFIDENCE = 0.3
MIN_TRACKING_CONFIDENCE = 0.3
SMOOTHING_WINDOW_SIZE = 10

# ---------------------------------------------------------------------------
# MediaPipe Pose configuration (shared across all analyzers)
# ---------------------------------------------------------------------------
MEDIAPIPE_POSE_CONFIG = {
    "static_image_mode": True,
    "model_complexity": 2,
    "enable_segmentation": True,
    "min_detection_confidence": MIN_DETECTION_CONFIDENCE,
    "min_tracking_confidence": MIN_TRACKING_CONFIDENCE,
}

MEDIAPIPE_POSE_VIDEO_CONFIG = {
    **MEDIAPIPE_POSE_CONFIG,
    "static_image_mode": False,
}

# ---------------------------------------------------------------------------
# Image preprocessing (CLAHE)
# ---------------------------------------------------------------------------
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID_SIZE = (8, 8)
CLAHE_CLIP_LIMIT_POSE = 2.5

DENOISE_H = 10
DENOISE_H_COLOR = 10
DENOISE_TEMPLATE_WINDOW = 7
DENOISE_SEARCH_WINDOW = 21

# ---------------------------------------------------------------------------
# EMA smoothing for skeleton tracking
# ---------------------------------------------------------------------------
EMA_ALPHA = 0.72
SKELETON_DISPLACEMENT_MAX_PX = 80
SKELETON_JOINT_RADIUS_PX = 2

# ---------------------------------------------------------------------------
# Hydrodynamics
# ---------------------------------------------------------------------------
WATER_DENSITY_KG_M3 = 1000
BASE_DRAG_COEFFICIENT = 0.4

# ---------------------------------------------------------------------------
# Cycling thresholds
# ---------------------------------------------------------------------------
PEDAL_PHASE_KNEE_DOWN_PX = 50
PEDAL_PHASE_KNEE_UP_PX = -30
OPTIMAL_CADENCE_RPM_MIN = 80
OPTIMAL_CADENCE_RPM_MAX = 100

# ---------------------------------------------------------------------------
# Running thresholds (additional)
# ---------------------------------------------------------------------------
OPTIMAL_CADENCE_SPM_MIN = 170
OPTIMAL_CADENCE_SPM_MAX = 190
