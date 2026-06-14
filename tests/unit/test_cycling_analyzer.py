"""Unit tests for CyclingAnalyzer."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

cv2 = pytest.importorskip("cv2", reason="cv2 not installed — skipping CyclingAnalyzer tests")

from tests.fixtures.mock_keypoints import make_cycling_frames  # noqa: E402
from tests.fixtures.mock_keypoints import make_empty_frames
from video_analysis.cycling_analyzer import CyclingAnalysis  # noqa: E402
from video_analysis.cycling_analyzer import CyclingAnalyzer


class TestCyclingAnalyzerEmpty:

    def test_empty_input_no_crash(self):
        analyzer = CyclingAnalyzer(fps=30.0)
        result = analyzer.analyze([], fps=30.0)
        assert isinstance(result, CyclingAnalysis)

    def test_empty_frames_no_crash(self):
        analyzer = CyclingAnalyzer(fps=30.0)
        result = analyzer.analyze(make_empty_frames(30), fps=30.0)
        assert isinstance(result, CyclingAnalysis)
        assert result.cadence == 0.0
        assert result.efficiency_score == 0.0
        assert result.bike_fit_score == 0.0


class TestCyclingAnalyzerMetrics:

    def test_cycling_frames_produce_result(self):
        frames = make_cycling_frames(n=180, fps=30.0)
        analyzer = CyclingAnalyzer(fps=30.0)
        result = analyzer.analyze(frames, fps=30.0)
        assert isinstance(result, CyclingAnalysis)
        assert result.cadence >= 0.0

    def test_scores_in_valid_range(self):
        frames = make_cycling_frames(n=90, fps=30.0)
        analyzer = CyclingAnalyzer(fps=30.0)
        result = analyzer.analyze(frames, fps=30.0)
        for attr in ("bike_fit_score", "ankling_score", "pedal_smoothness"):
            val = getattr(result, attr, None)
            if val is not None:
                assert 0.0 <= val <= 100.0, f"{attr}={val} out of range"

    def test_inherits_base_calculate_angle(self):
        """Verify BaseAnalyzer inheritance is intact."""
        from video_analysis.base_analyzer import BaseAnalyzer

        analyzer = CyclingAnalyzer(fps=30.0)
        assert isinstance(analyzer, BaseAnalyzer)
        angle = analyzer._calculate_angle((0, 1), (0, 0), (1, 0))
        assert abs(angle - 90.0) < 0.01

    def test_no_duplicate_calculate_angle(self):
        """Confirm _calculate_angle is inherited, not re-defined in CyclingAnalyzer."""
        from video_analysis.base_analyzer import BaseAnalyzer
        from video_analysis.cycling_analyzer import CyclingAnalyzer

        # Method should come from BaseAnalyzer, not CyclingAnalyzer
        assert CyclingAnalyzer._calculate_angle is BaseAnalyzer._calculate_angle

    def test_top_knee_angle_is_flexed_and_bottom_angle_is_extended(self):
        analyzer = CyclingAnalyzer(fps=30.0)

        result = analyzer._calculate_stats(
            knee_left=[72.0, 72.0, 110.0, 148.0, 148.0],
            knee_right=[72.0, 74.0, 112.0, 146.0, 148.0],
            hip_angles=[42.0],
            shoulder_positions=[(0.0, 0.0), (1.0, 1.0)],
            total_frames=90,
        )

        assert result.avg_knee_angle_top == 72.0
        assert result.avg_knee_angle_bottom == 148.0
        assert result.knee_range == 76.0
        assert result.saddle_height_score == 100.0

    def test_knee_extrema_ignore_single_frame_pose_outliers(self):
        analyzer = CyclingAnalyzer(fps=30.0)

        result = analyzer._calculate_stats(
            knee_left=[5.0, 72.0, 74.0, 110.0, 146.0, 148.0, 179.0],
            knee_right=[72.0, 74.0, 110.0, 146.0, 148.0],
            hip_angles=[42.0],
            shoulder_positions=[(0.0, 0.0), (1.0, 1.0)],
            total_frames=90,
        )

        assert result.avg_knee_angle_top > 50.0
        assert result.avg_knee_angle_bottom < 165.0

    def test_missing_arm_evidence_is_excluded_from_bike_fit_score(self):
        analyzer = CyclingAnalyzer(fps=30.0)

        result = analyzer._calculate_stats(
            knee_left=[72.0, 72.0, 148.0, 148.0],
            knee_right=[72.0, 72.0, 148.0, 148.0],
            hip_angles=[40.0, 41.0],
            shoulder_positions=[(10.0, 10.0), (10.0, 10.0)],
            total_frames=90,
        )

        assert result.stack_score == 0.0
        assert result.bike_fit_score == 100.0

    def test_reusing_analyzer_does_not_leak_previous_clip_state(self):
        analyzer = CyclingAnalyzer(fps=30.0)
        first_clip = make_cycling_frames(n=90, fps=30.0)
        for index, frame in enumerate(first_clip):
            sway = math.sin(index / 5) * 0.03
            frame.update(
                {
                    11: {"x": 0.40 + sway, "y": 0.30, "visibility": 0.9},
                    12: {"x": 0.60 + sway, "y": 0.30, "visibility": 0.9},
                    13: {"x": 0.35, "y": 0.42, "visibility": 0.9},
                    14: {"x": 0.65, "y": 0.42, "visibility": 0.9},
                    15: {"x": 0.30, "y": 0.55, "visibility": 0.9},
                    16: {"x": 0.70, "y": 0.55, "visibility": 0.9},
                }
            )

        first = analyzer.analyze(first_clip, fps=30.0)
        second_clip = make_cycling_frames(n=90, fps=30.0)
        for frame in second_clip:
            for landmark in (11, 12, 13, 14, 15, 16):
                frame.pop(landmark, None)
        second = analyzer.analyze(second_clip, fps=30.0)

        assert first.reach_angle > 0
        assert second.reach_angle == 0
        assert second.lateral_sway == 0
        assert second.vertical_bounce == 0
        assert second.stack_score == 0
