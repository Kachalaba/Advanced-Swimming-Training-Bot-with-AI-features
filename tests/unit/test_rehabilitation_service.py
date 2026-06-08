"""Tests for stateful web rehabilitation sessions."""

import time

import numpy as np

from backend.app.services.rehabilitation import LiveRehabRegistry, LiveRehabSession


class _FakePoseProcessor:
    def process_frame(self, frame, index, bbox=None):
        return frame, {
            "has_pose": True,
            "keypoints": {
                "left_shoulder": {"x": 30, "y": 30},
                "right_shoulder": {"x": 70, "y": 34},
                "left_hip": {"x": 36, "y": 62},
                "right_hip": {"x": 66, "y": 60},
                "left_elbow": {"x": 28, "y": 45},
                "right_elbow": {"x": 72, "y": 45},
                "left_wrist": {"x": 25, "y": 60},
                "right_wrist": {"x": 75, "y": 60},
                "left_knee": {"x": 38, "y": 78},
                "right_knee": {"x": 64, "y": 78},
                "left_ankle": {"x": 39, "y": 95},
                "right_ankle": {"x": 63, "y": 95},
            },
        }


class _FakeAnalyzer:
    def analyze(self, keypoints, protocol):
        return {
            "protocol": protocol,
            "total_correct_reps": len(keypoints),
            "completion_score": 82.0,
            "target_metrics": {
                "left": {"rom": 84.0},
                "right": {"rom": 81.0},
            },
            "symmetry": {"asymmetry_index": 3.6, "score": 96.4},
        }


class _FakeLevel:
    def __init__(self):
        self.last_bbox = None

    def calibrate(self, frame, athlete_bbox=None):
        self.last_bbox = athlete_bbox
        return {"angle_deg": 0.0, "confidence": 1.0, "status": "level", "relative": True}

    def measure(self, frame, athlete_bbox=None):
        self.last_bbox = athlete_bbox
        return {"angle_deg": 0.4, "confidence": 0.9, "status": "level", "relative": True}


def _session(max_frames=3):
    return LiveRehabSession(
        protocol="shoulder_flexion",
        fps=5.0,
        pose_processor=_FakePoseProcessor(),
        analyzer=_FakeAnalyzer(),
        camera_level=_FakeLevel(),
        max_frames=max_frames,
        analysis_interval=1,
    )


def test_live_session_keeps_bounded_keypoint_window():
    session = _session(max_frames=3)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    for _ in range(5):
        update = session.process_frame(frame)

    assert session.keypoint_count == 3
    assert update["report"]["protocol"] == "shoulder_flexion"
    assert update["report"]["total_correct_reps"] == 3


def test_live_session_normalizes_landmarks_and_calibrates():
    session = _session()

    update = session.process_frame(
        np.zeros((100, 100, 3), dtype=np.uint8),
        calibrate=True,
    )

    assert update["landmarks"]["left_shoulder"] == {"x": 0.3, "y": 0.3}
    assert update["posture"]["available"] is True
    assert update["camera_level"]["angle_deg"] == 0.0


def test_registry_creates_and_deletes_sessions():
    registry = LiveRehabRegistry(session_factory=lambda protocol, fps: _session())

    session = registry.create("shoulder_flexion", 5.0)

    assert registry.get(session.id) is session
    assert registry.delete(session.id) is True
    assert registry.get(session.id) is None


def test_registry_purges_idle_sessions():
    registry = LiveRehabRegistry(
        session_factory=lambda protocol, fps: _session(),
        ttl_seconds=10.0,
    )

    session = registry.create("shoulder_flexion", 5.0)
    session.last_active = time.monotonic() - 999.0

    assert registry.purge_expired() == 1
    assert registry.get(session.id) is None


def test_registry_evicts_oldest_when_at_capacity():
    registry = LiveRehabRegistry(
        session_factory=lambda protocol, fps: _session(),
        max_sessions=2,
    )

    first = registry.create("shoulder_flexion", 5.0)
    first.last_active = time.monotonic() - 100.0  # mark as least-recently active
    second = registry.create("shoulder_flexion", 5.0)
    third = registry.create("shoulder_flexion", 5.0)

    assert registry.active_count() == 2
    assert registry.get(first.id) is None
    assert registry.get(second.id) is second
    assert registry.get(third.id) is third


def test_live_session_excludes_detected_body_from_camera_level():
    level = _FakeLevel()
    session = LiveRehabSession(
        protocol="shoulder_flexion",
        fps=5.0,
        pose_processor=_FakePoseProcessor(),
        analyzer=_FakeAnalyzer(),
        camera_level=level,
        analysis_interval=1,
    )

    session.process_frame(np.zeros((100, 100, 3), dtype=np.uint8), calibrate=True)

    assert level.last_bbox is not None
    x1, y1, x2, y2 = level.last_bbox
    assert x1 < 25
    assert y1 < 30
    assert x2 > 75
    assert y2 > 95
