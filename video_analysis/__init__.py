"""Video analysis module for swimming technique."""

from .video_overlay import VideoOverlayGenerator, generate_annotated_video
from .trajectory_analyzer import TrajectoryAnalyzer, analyze_trajectory
from .biomechanics_analyzer import BiomechanicsAnalyzer, analyze_biomechanics

__version__ = "2.0.0"

__all__ = [
    "VideoOverlayGenerator",
    "generate_annotated_video",
    "TrajectoryAnalyzer",
    "analyze_trajectory",
    "BiomechanicsAnalyzer",
    "analyze_biomechanics",
]
