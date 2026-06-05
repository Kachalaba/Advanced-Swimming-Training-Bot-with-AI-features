"""Tests for normalized postural coordinate calculations."""

from backend.app.services.posture import calculate_posture


def test_calculate_posture_reports_axis_angles_and_trunk_offset():
    landmarks = {
        "left_shoulder": {"x": 0.30, "y": 0.30},
        "right_shoulder": {"x": 0.70, "y": 0.34},
        "left_hip": {"x": 0.36, "y": 0.62},
        "right_hip": {"x": 0.66, "y": 0.60},
    }

    result = calculate_posture(landmarks, frame_center_x=0.5)

    assert result["available"] is True
    assert result["shoulder_angle_deg"] > 0
    assert result["pelvis_angle_deg"] < 0
    assert result["trunk_lean_deg"] != 0
    assert result["trunk_offset_pct"] != 0
    assert result["points"]["shoulder_mid"]["x"] == 0.5


def test_calculate_posture_reports_directional_severity():
    landmarks = {
        "left_shoulder": {"x": 0.30, "y": 0.28},
        "right_shoulder": {"x": 0.70, "y": 0.36},
        "left_hip": {"x": 0.34, "y": 0.60},
        "right_hip": {"x": 0.66, "y": 0.60},
    }

    result = calculate_posture(landmarks)

    assert result["shoulder"]["severity"] == "warning"
    assert result["shoulder"]["higher_side"] == "left"
    assert result["pelvis"]["severity"] == "good"


def test_calculate_posture_is_orientation_independent_for_level_line():
    # MediaPipe "left_*" lands on the right of a non-mirrored frame (left.x >
    # right.x). A perfectly level shoulder/pelvis line must still read ~0 deg and
    # stay "good" rather than folding to ~180 deg and tripping the warning.
    landmarks = {
        "left_shoulder": {"x": 0.70, "y": 0.30},
        "right_shoulder": {"x": 0.30, "y": 0.30},
        "left_hip": {"x": 0.66, "y": 0.62},
        "right_hip": {"x": 0.34, "y": 0.62},
    }

    result = calculate_posture(landmarks)

    assert abs(result["shoulder_angle_deg"]) <= 1.0
    assert result["shoulder"]["severity"] == "good"
    assert result["pelvis"]["severity"] == "good"


def test_calculate_posture_returns_unavailable_for_missing_landmarks():
    assert calculate_posture({}, frame_center_x=0.5) == {"available": False}
