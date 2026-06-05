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


def test_shoulder_abduction_uses_distinct_target_from_flexion():
    frames = make_rehab_shoulder_flexion_frames(cycles=1)

    flex = RehabAnalyzer(fps=10.0).analyze(frames, protocol="shoulder_flexion")
    abd = RehabAnalyzer(fps=10.0).analyze(frames, protocol="shoulder_abduction")

    # Same 2D geometry, but the two protocols are no longer silent aliases:
    # abduction carries a stricter clinical target ROM.
    assert flex["target_rom"] == 150.0
    assert abd["target_rom"] == 160.0
    assert abd["target_metrics"]["left"]["target_rom"] == 160.0
    # Identical motion therefore leaves a larger remaining deficit under abduction.
    assert abd["target_metrics"]["left"]["deficit_deg"] >= flex["target_metrics"]["left"]["deficit_deg"]


def test_rep_duration_tracks_real_frames_through_dropped_poses():
    frames = make_rehab_shoulder_flexion_frames(cycles=1)
    # Interleave dropped-pose frames so the same motion spans twice as many frames.
    gappy = []
    for frame in frames:
        gappy.append(frame)
        gappy.append({})

    base = RehabAnalyzer(fps=10.0).analyze(frames, protocol="shoulder_flexion")
    spaced = RehabAnalyzer(fps=10.0).analyze(gappy, protocol="shoulder_flexion")

    base_reps = base["target_metrics"]["left"]["rep_details"]
    spaced_reps = spaced["target_metrics"]["left"]["rep_details"]
    assert base_reps and spaced_reps
    # Durations follow real elapsed frames, not compacted sample positions, so the
    # spread-out recording reports a proportionally longer repetition.
    assert spaced_reps[0]["duration_sec"] > base_reps[0]["duration_sec"] * 1.5


def test_report_contains_all_tracked_joint_families():
    frames = make_rehab_shoulder_flexion_frames(cycles=1)

    result = RehabAnalyzer(fps=10.0).analyze(frames, protocol="shoulder_flexion")

    assert set(result["joint_metrics"]) == {
        "shoulder",
        "elbow_flexion",
        "hip_abduction",
        "knee_extension",
    }
    for metrics in result["joint_metrics"].values():
        assert set(metrics) >= {"left", "right", "asymmetry_index", "symmetry_score"}
