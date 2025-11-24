"""Video analysis module for swimming technique."""

__version__ = "1.0.0"

# Lazy imports keep heavy optional deps (cv2/torch) out of module import path.
__all__ = ["VideoOverlayGenerator", "generate_annotated_video"]


def __getattr__(name: str):
    if name in {"VideoOverlayGenerator", "generate_annotated_video"}:
        from .video_overlay import VideoOverlayGenerator, generate_annotated_video

        globals().update(
            {
                "VideoOverlayGenerator": VideoOverlayGenerator,
                "generate_annotated_video": generate_annotated_video,
            }
        )
        return globals()[name]
    raise AttributeError(f"module 'video_analysis' has no attribute {name!r}")
