"""Unit tests for bilateral rehabilitation ROM analysis."""

import pytest

from tests.fixtures.mock_keypoints import make_rehab_shoulder_flexion_frames
from video_analysis.rehab_analyzer import RehabAnalyzer


def test_empty_input_returns_zero_report():
    result = RehabAnalyzer(fps=10.0).analyze([], protocol="shoulder_flexion")

    assert result["protocol"] == "shoulder_flexion"
    assert result["total_correct_reps"] == 0
    assert result["symmetry"]["asymmetry_index"] == 0.0
    assert result["target_metrics"]["left"]["rom"] == 0.0
    assert result["feedback"]


def test_unknown_protocol_is_rejected():
    analyzer = RehabAnalyzer(fps=10.0)

    with pytest.raises(ValueError, match="Unknown rehabilitation protocol"):
        analyzer.analyze([], protocol="not-a-protocol")


def test_shoulder_flexion_reports_bilateral_rom_and_deficit():
    frames = make_rehab_shoulder_flexion_frames()

    result = RehabAnalyzer(fps=10.0).analyze(frames, protocol="shoulder_flexion")

    left = result["target_metrics"]["left"]
    right = result["target_metrics"]["right"]
    assert left["rom"] > right["rom"]
    assert left["rom"] >= 125.0
    assert 95.0 <= right["rom"] <= 115.0
    assert left["deficit_deg"] < right["deficit_deg"]
    assert result["symmetry"]["asymmetry_index"] > 10.0
    assert result["symmetry"]["score"] < 90.0


def test_shoulder_flexion_counts_only_reps_reaching_target_completion():
    frames = make_rehab_shoulder_flexion_frames(cycles=2)

    result = RehabAnalyzer(fps=10.0).analyze(frames, protocol="shoulder_flexion")

    left = result["target_metrics"]["left"]
    right = result["target_metrics"]["right"]
    assert left["reps"] == 2
    assert left["correct_reps"] == 2
    assert right["reps"] == 2
    assert right["correct_reps"] == 0
    assert result["total_correct_reps"] == 2
    assert len(result["rep_rom_history"]["left"]) == 2


def test_report_contains_all_tracked_joint_families():
    frames = make_rehab_shoulder_flexion_frames(cycles=1)

    result = RehabAnalyzer(fps=10.0).analyze(frames, protocol="shoulder_flexion")

    assert set(result["joint_metrics"]) == {
        "shoulder_flexion",
        "elbow_flexion",
        "hip_abduction",
        "knee_extension",
    }
    for metrics in result["joint_metrics"].values():
        assert set(metrics) >= {"left", "right", "asymmetry_index", "symmetry_score"}
