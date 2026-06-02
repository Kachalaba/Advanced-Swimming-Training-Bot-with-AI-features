"""Unit tests for StrokeAnalyzer — division by zero guards and metric calculations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from video_analysis.stroke_analyzer import StrokeAnalysis, StrokeAnalyzer


def _make_empty_keypoints():
    """Return a list of empty keypoint dicts (no pose detected)."""
    return [{} for _ in range(30)]


def _make_keypoints_with_wrists(n_frames: int = 60, fps: float = 10.0):
    """Return minimal keypoints that include wrist landmarks."""
    frames = []
    for i in range(n_frames):
        # Simulate sinusoidal wrist movement to trigger stroke detection
        import math

        t = i / fps
        left_y = 0.5 + 0.3 * math.sin(2 * math.pi * 0.5 * t)
        right_y = 0.5 + 0.3 * math.sin(2 * math.pi * 0.5 * t + math.pi)
        frames.append(
            {
                11: {"x": 0.3, "y": 0.5, "visibility": 0.9},  # LEFT_SHOULDER
                12: {"x": 0.7, "y": 0.5, "visibility": 0.9},  # RIGHT_SHOULDER
                13: {"x": 0.25, "y": 0.5, "visibility": 0.9},  # LEFT_ELBOW
                14: {"x": 0.75, "y": 0.5, "visibility": 0.9},  # RIGHT_ELBOW
                15: {"x": 0.2, "y": left_y, "visibility": 0.9},  # LEFT_WRIST
                16: {"x": 0.8, "y": right_y, "visibility": 0.9},  # RIGHT_WRIST
                23: {"x": 0.35, "y": 0.7, "visibility": 0.9},  # LEFT_HIP
                24: {"x": 0.65, "y": 0.7, "visibility": 0.9},  # RIGHT_HIP
                27: {"x": 0.35, "y": 0.9, "visibility": 0.9},  # LEFT_ANKLE
                28: {"x": 0.65, "y": 0.9, "visibility": 0.9},  # RIGHT_ANKLE
                25: {"x": 0.35, "y": 0.8, "visibility": 0.9},  # LEFT_KNEE
                26: {"x": 0.65, "y": 0.8, "visibility": 0.9},  # RIGHT_KNEE
                0: {"x": 0.5, "y": 0.1, "visibility": 0.9},  # NOSE
                7: {"x": 0.45, "y": 0.1, "visibility": 0.9},  # LEFT_EAR
                8: {"x": 0.55, "y": 0.1, "visibility": 0.9},  # RIGHT_EAR
            }
        )
    return frames


class TestStrokeAnalyzerDivisionByZero:
    """Verify that StrokeAnalyzer doesn't raise ZeroDivisionError on edge inputs."""

    def test_empty_keypoints_no_crash(self):
        analyzer = StrokeAnalyzer(fps=10.0)
        result = analyzer.analyze([], fps=10.0)
        assert isinstance(result, StrokeAnalysis)

    def test_all_empty_dicts_no_crash(self):
        """30 frames with no landmarks — should return zero-valued result."""
        analyzer = StrokeAnalyzer(fps=10.0)
        result = analyzer.analyze(_make_empty_keypoints(), fps=10.0)
        assert isinstance(result, StrokeAnalysis)
        assert result.total_strokes == 0

    def test_single_frame_no_crash(self):
        """Single frame — no strokes possible, no division by zero."""
        analyzer = StrokeAnalyzer(fps=10.0)
        result = analyzer.analyze([{}], fps=10.0)
        assert result.total_strokes == 0

    def test_phases_distribution_sums_to_100_or_zero(self):
        """phases_distribution values must be valid percentages."""
        analyzer = StrokeAnalyzer(fps=10.0)
        result = analyzer.analyze(_make_keypoints_with_wrists(60, 10.0), fps=10.0)
        if result.phases_distribution:
            total = sum(result.phases_distribution.values())
            # Allow floating point tolerance
            assert abs(total - 100.0) < 1.0 or total == 0.0


class TestStrokeAnalyzerMetrics:
    """Verify output metrics are within valid ranges."""

    def test_symmetry_score_range(self):
        analyzer = StrokeAnalyzer(fps=10.0)
        result = analyzer.analyze(_make_keypoints_with_wrists(60, 10.0), fps=10.0)
        assert 0.0 <= result.symmetry_score <= 100.0

    def test_stroke_rate_non_negative(self):
        analyzer = StrokeAnalyzer(fps=10.0)
        result = analyzer.analyze(_make_keypoints_with_wrists(60, 10.0), fps=10.0)
        assert result.stroke_rate >= 0.0

    def test_body_roll_zero_when_no_landmarks(self):
        analyzer = StrokeAnalyzer(fps=10.0)
        result = analyzer.analyze(_make_empty_keypoints(), fps=10.0)
        # No valid landmarks → body roll defaults to 0
        assert result.avg_body_roll == 0.0

    def test_left_plus_right_equals_total(self):
        analyzer = StrokeAnalyzer(fps=10.0)
        result = analyzer.analyze(_make_keypoints_with_wrists(60, 10.0), fps=10.0)
        assert result.left_strokes + result.right_strokes == result.total_strokes

    def test_fps_zero_fallback(self):
        """fps=0 must not raise ZeroDivisionError."""
        analyzer = StrokeAnalyzer(fps=0.0)
        result = analyzer.analyze(_make_empty_keypoints(), fps=0.0)
        assert isinstance(result, StrokeAnalysis)
