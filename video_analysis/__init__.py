"""Video analysis module for swimming technique.

All heavy imports (cv2, mediapipe, torch) are deferred to submodules.
Importing ``video_analysis`` itself is lightweight and test-friendly.

Modules:
    - frame_extractor:       Extract frames from video with timestamps
    - swimmer_detector:      YOLO detection + velocity tracking
    - biomechanics_analyzer: MediaPipe pose (33 keypoints)
    - split_analyzer:        Splits, speed, tempo analysis
    - trajectory_analyzer:   Bbox trajectory analysis
    - video_overlay:         Annotated video generation
    - report_generator:      PDF/JSON reports
    - base_analyzer:         Shared utilities (BaseAnalyzer mixin)
    - analyzer_factory:      Cached factory functions (st.cache_resource)
    - service_layer:         AnalysisService + AICoachProtocol
"""


def __getattr__(name):
    """Lazy-import public names to keep the package import lightweight.

    Heavy dependencies (cv2, mediapipe, torch) are only loaded when a
    specific name is actually accessed, not on ``import video_analysis``.
    """
    _map = {
        "extract_frames_from_video":    (".frame_extractor",       "extract_frames_from_video"),
        "SwimmerDetector":              (".swimmer_detector",       "SwimmerDetector"),
        "detect_swimmer_in_frames":     (".swimmer_detector",       "detect_swimmer_in_frames"),
        "BiomechanicsAnalyzer":         (".biomechanics_analyzer",  "BiomechanicsAnalyzer"),
        "analyze_biomechanics":         (".biomechanics_analyzer",  "analyze_biomechanics"),
        "SwimmingPoseAnalyzer":         (".swimming_pose_analyzer", "SwimmingPoseAnalyzer"),
        "analyze_swimming_pose":        (".swimming_pose_analyzer", "analyze_swimming_pose"),
        "analyze_swimming_video":       (".split_analyzer",         "analyze_swimming_video"),
        "TrajectoryAnalyzer":           (".trajectory_analyzer",    "TrajectoryAnalyzer"),
        "analyze_trajectory":           (".trajectory_analyzer",    "analyze_trajectory"),
        "VideoOverlayGenerator":        (".video_overlay",          "VideoOverlayGenerator"),
        "generate_annotated_video":     (".video_overlay",          "generate_annotated_video"),
        "ReportGenerator":              (".report_generator",       "ReportGenerator"),
        "AICoach":                      (".ai_coach",               "AICoach"),
        "get_ai_coaching":              (".ai_coach",               "get_ai_coaching"),
        "BaseAnalyzer":                 (".base_analyzer",          "BaseAnalyzer"),
        "AnalysisService":              (".service_layer",          "AnalysisService"),
        "AICoachProtocol":              (".service_layer",          "AICoachProtocol"),
    }
    if name in _map:
        import importlib
        module_path, attr = _map[name]
        mod = importlib.import_module(module_path, package=__name__)
        return getattr(mod, attr)
    raise AttributeError(f"module 'video_analysis' has no attribute {name!r}")


__version__ = "2.3.0"

__all__ = [
    "AICoach", "get_ai_coaching",
    "extract_frames_from_video",
    "SwimmerDetector", "detect_swimmer_in_frames",
    "BiomechanicsAnalyzer", "analyze_biomechanics",
    "SwimmingPoseAnalyzer", "analyze_swimming_pose",
    "analyze_swimming_video",
    "TrajectoryAnalyzer", "analyze_trajectory",
    "VideoOverlayGenerator", "generate_annotated_video",
    "ReportGenerator",
    "BaseAnalyzer",
    "AnalysisService", "AICoachProtocol",
]
