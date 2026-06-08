"""Unit tests for RunningAnalyzer."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

cv2 = pytest.importorskip("cv2", reason="cv2 not installed — skipping RunningAnalyzer tests")

from tests.fixtures.mock_keypoints import make_empty_frames  # noqa: E402
from tests.fixtures.mock_keypoints import make_running_frames, make_static_standing_frames
from video_analysis.running_analyzer import RunningAnalysis  # noqa: E402
from video_analysis.running_analyzer import RunningAnalyzer


class TestRunningAnalyzerEmpty:

    def test_empty_input_no_crash(self):
        analyzer = RunningAnalyzer(fps=30.0)
        result = analyzer.analyze([], fps=30.0)
        assert isinstance(result, RunningAnalysis)

    def test_empty_frames_no_crash(self):
        analyzer = RunningAnalyzer(fps=30.0)
        result = analyzer.analyze(make_empty_frames(30), fps=30.0)
        assert isinstance(result, RunningAnalysis)
        assert result.cadence == 0.0

    def test_static_frames_no_crash(self):
        """Standing still — should produce zero or near-zero cadence."""
        analyzer = RunningAnalyzer(fps=30.0)
        result = analyzer.analyze(make_static_standing_frames(60), fps=30.0)
        assert isinstance(result, RunningAnalysis)


class TestRunningAnalyzerMetrics:

    def test_running_frames_produce_cadence(self):
        frames = make_running_frames(n=180, fps=30.0)
        analyzer = RunningAnalyzer(fps=30.0)
        result = analyzer.analyze(frames, fps=30.0)
        assert isinstance(result, RunningAnalysis)
        assert 150.0 <= result.cadence <= 185.0

    def test_cadence_uses_observed_step_intervals_when_pose_is_missing(self):
        frames = make_running_frames(n=90, fps=30.0) + make_empty_frames(90)
        result = RunningAnalyzer(fps=30.0).analyze(frames, fps=30.0)
        assert 150.0 <= result.cadence <= 185.0

    def test_vertical_oscillation_ignores_single_pose_outlier(self):
        frames = make_running_frames(n=180, fps=30.0)
        frames[40][23]["y"] = 10.0
        frames[40][24]["y"] = 10.0
        result = RunningAnalyzer(fps=30.0).analyze(frames, fps=30.0)
        assert result.avg_vertical_osc < 0.1

    def test_scores_in_valid_range(self):
        frames = make_running_frames(n=90, fps=30.0)
        analyzer = RunningAnalyzer(fps=30.0)
        result = analyzer.analyze(frames, fps=30.0)
        for attr in ("efficiency_score", "hip_drop_score", "foot_strike_score"):
            val = getattr(result, attr, None)
            if val is not None:
                assert 0.0 <= val <= 100.0, f"{attr}={val} out of range"

    def test_injury_risk_non_negative(self):
        frames = make_running_frames(n=90, fps=30.0)
        analyzer = RunningAnalyzer(fps=30.0)
        result = analyzer.analyze(frames, fps=30.0)
        assert result.injury_risk_score >= 0

    def test_inherits_base_calculate_angle(self):
        """Verify BaseAnalyzer methods are accessible after inheritance."""
        from video_analysis.base_analyzer import BaseAnalyzer

        analyzer = RunningAnalyzer(fps=30.0)
        assert isinstance(analyzer, BaseAnalyzer)
        angle = analyzer._calculate_angle((1, 0), (0, 0), (0, 1))
        assert abs(angle - 90.0) < 0.01
