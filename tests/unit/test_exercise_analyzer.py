import math

import pytest

from video_analysis.exercise_analyzer import ExerciseAnalyzer


def _angle_frames(values, key="L.knee"):
    return [{key: value} if value is not None else {} for value in values]


def test_squat_counts_ready_effort_ready_reps():
    analyzer = ExerciseAnalyzer(fps=10)
    angles = [170, 166, 155, 130, 104, 92, 105, 135, 160, 170] * 2

    result = analyzer.analyze(_angle_frames(angles, "L.knee"), exercise_type="squat")

    assert result.exercise_type == "squat"
    assert result.tracked_joint == "knee"
    assert result.total_reps == 2
    assert result.avg_range_of_motion >= 70
    assert result.reps[0].start_frame == 0
    assert result.reps[0].effort_frame == 5
    assert result.reps[0].end_frame == 9
    assert result.reps[0].active_side == "left"


def test_push_up_uses_elbow_profile():
    analyzer = ExerciseAnalyzer(fps=10)
    elbow = [168, 160, 132, 96, 74, 68, 80, 110, 145, 166]

    result = analyzer.analyze(_angle_frames(elbow, "R.elbow"), exercise_type="push_up")

    assert result.tracked_joint == "elbow"
    assert result.total_reps == 1
    assert result.reps[0].active_side == "right"
    assert result.min_angle == pytest.approx(68, abs=1)


def test_lunge_uses_more_flexed_knee_and_side_label():
    analyzer = ExerciseAnalyzer(fps=10)
    frames = []
    left = [170, 165, 150, 124, 104, 94, 110, 138, 160, 170]
    right = [171, 168, 166, 160, 154, 150, 156, 164, 168, 170]
    for left_angle, right_angle in zip(left, right, strict=False):
        frames.append({"L.knee": left_angle, "R.knee": right_angle})

    result = analyzer.analyze(frames, exercise_type="lunge")

    assert result.total_reps == 1
    assert result.reps[0].active_side == "left"
    assert result.avg_range_of_motion >= 70


def test_partial_cycles_are_not_counted():
    analyzer = ExerciseAnalyzer(fps=10)
    angles = [120, 100, 88, 94, 112, 132, 152, 166]

    result = analyzer.analyze(_angle_frames(angles, "L.knee"), exercise_type="squat")

    assert result.total_reps == 0
    assert result.avg_tempo == 0


def test_long_missing_gap_does_not_create_rep():
    analyzer = ExerciseAnalyzer(fps=10)
    angles = [170, 164, 150, None, None, None, None, 92, 108, 138, 166, 170]

    result = analyzer.analyze(_angle_frames(angles, "L.knee"), exercise_type="squat")

    assert result.total_reps == 0
    assert all(math.isfinite(value) for value in result.angle_history)


def test_state_does_not_leak_between_clips():
    analyzer = ExerciseAnalyzer(fps=10)
    full_rep = [170, 166, 150, 120, 95, 90, 110, 138, 160, 170]

    first = analyzer.analyze(_angle_frames(full_rep, "L.knee"), exercise_type="squat")
    second = analyzer.analyze(_angle_frames([170, 166, 160], "L.knee"), exercise_type="squat")

    assert first.total_reps == 1
    assert second.total_reps == 0
    assert analyzer.get_rep_at_frame(20) == 0
