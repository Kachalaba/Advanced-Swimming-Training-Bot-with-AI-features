"""Video analysis module for swimming technique.

Modules:
    - frame_extractor: Extract frames from video with timestamps
    - swimmer_detector: YOLO detection + velocity tracking
    - biomechanics_analyzer: MediaPipe pose (33 keypoints)
    - split_analyzer: Splits, speed, tempo analysis
    - trajectory_analyzer: Bbox trajectory analysis
    - video_overlay: Annotated video generation
    - report_generator: PDF/JSON reports
"""

from .frame_extractor import extract_frames_from_video
from .swimmer_detector import SwimmerDetector, detect_swimmer_in_frames
from .biomechanics_analyzer import BiomechanicsAnalyzer, analyze_biomechanics
from .split_analyzer import analyze_swimming_video
from .trajectory_analyzer import TrajectoryAnalyzer, analyze_trajectory
from .video_overlay import VideoOverlayGenerator, generate_annotated_video
from .report_generator import ReportGenerator
from .swimming_pose_analyzer import SwimmingPoseAnalyzer, analyze_swimming_pose
from .ai_coach import AICoach, get_ai_coaching

__version__ = "2.3.0"

__all__ = [
    # Frame extraction
    "extract_frames_from_video",
    # Detection
    "SwimmerDetector",
    "detect_swimmer_in_frames",
    # Biomechanics (legacy)
    "BiomechanicsAnalyzer",
    "analyze_biomechanics",
    # NEW: Swimming-specific pose analysis
    "SwimmingPoseAnalyzer",
    "analyze_swimming_pose",
    # Splits
    "analyze_swimming_video",
    # Trajectory
    "TrajectoryAnalyzer",
    "analyze_trajectory",
    # Video
    "VideoOverlayGenerator",
    "generate_annotated_video",
    # Reports
    "ReportGenerator",
]
