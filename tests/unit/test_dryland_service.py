from backend.app.services.dryland import (
    dryland_quality,
    metric_ready,
    select_active_angles,
)


def test_metric_ready_squat_accepts_one_complete_side():
    keypoints = {
        "left_shoulder": (0, 0),
        "left_hip": (0, 1),
        "left_knee": (0, 2),
        "left_ankle": (0, 3),
    }

    assert metric_ready("squat", keypoints) is True


def test_metric_ready_lunge_requires_both_knees_and_ankles():
    incomplete = {
        "left_shoulder": (0, 0),
        "left_hip": (0, 1),
        "left_knee": (0, 2),
        "left_ankle": (0, 3),
    }
    complete = incomplete | {"right_knee": (1, 2), "right_ankle": (1, 3)}

    assert metric_ready("lunge", incomplete) is False
    assert metric_ready("lunge", complete) is True


def test_metric_ready_push_up_requires_arm_and_body_line():
    keypoints = {
        "right_shoulder": (0, 0),
        "right_elbow": (1, 0),
        "right_wrist": (2, 0),
        "right_hip": (0, 1),
        "right_ankle": (0, 2),
    }

    assert metric_ready("push_up", keypoints) is True


def test_dryland_quality_rejects_sparse_metric_ready_frames():
    result = dryland_quality(frames_total=100, frames_with_pose=60, metric_ready_frames=12)

    assert result["status"] == "fail"
    assert "Too few metric-ready frames" in result["warnings"][0]


def test_dryland_quality_warns_on_usable_but_low_pose_coverage():
    result = dryland_quality(frames_total=100, frames_with_pose=55, metric_ready_frames=30)

    assert result["status"] == "pass"
    assert result["pose_coverage"] == 55.0
    assert "below the preferred 70%" in result["warnings"][0]


def test_select_active_angles_outputs_expected_keys():
    frames = [
        {
            "left_hip": (0, 0),
            "left_knee": (0, 1),
            "left_ankle": (0, 2),
            "right_hip": (2, 0),
            "right_knee": (2, 1),
            "right_ankle": (2, 2),
        }
    ]

    angles = select_active_angles("squat", frames)

    assert set(angles[0]) == {"L.knee", "R.knee"}
