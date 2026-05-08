"""Unit tests for CyclingAnalyzer."""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

cv2 = pytest.importorskip("cv2", reason="cv2 not installed — skipping CyclingAnalyzer tests")

from tests.fixtures.mock_keypoints import (
    make_empty_frames,
    make_cycling_frames,
)
from video_analysis.cycling_analyzer import CyclingAnalyzer, CyclingAnalysis


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
        import inspect
        from video_analysis.cycling_analyzer import CyclingAnalyzer
        from video_analysis.base_analyzer import BaseAnalyzer
        # Method should come from BaseAnalyzer, not CyclingAnalyzer
        assert CyclingAnalyzer._calculate_angle is BaseAnalyzer._calculate_angle
