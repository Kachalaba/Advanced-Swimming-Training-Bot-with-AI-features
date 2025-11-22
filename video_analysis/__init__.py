"""Video analysis module for swimming technique."""

from .video_overlay import VideoOverlayGenerator, generate_annotated_video

__version__ = "1.0.0"

__all__ = [
    "VideoOverlayGenerator",
    "generate_annotated_video",
]
