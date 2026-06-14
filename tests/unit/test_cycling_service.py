import threading
import time
from types import SimpleNamespace

import numpy as np

from backend.app.services import cycling
from backend.app.services.cycling import _extract_pose, _metric_ready, cycling_quality


def test_metric_ready_requires_one_complete_leg_and_torso_anchor():
    partial = {
        "left_hip": (1.0, 1.0),
        "left_knee": (1.0, 2.0),
    }
    complete = {
        **partial,
        "left_ankle": (1.0, 3.0),
        "left_shoulder": (1.0, 0.0),
    }

    assert _metric_ready(partial) is False
    assert _metric_ready(complete) is True


def test_cycling_quality_rejects_sparse_metric_ready_frames():
    rejected = cycling_quality(frames_total=100, frames_with_pose=19)
    accepted = cycling_quality(frames_total=100, frames_with_pose=70)

    assert rejected["status"] == "fail"
    assert rejected["pose_coverage"] == 19.0
    assert accepted["status"] == "pass"


def test_cached_pose_detector_calls_are_serialized(monkeypatch):
    class OverlapDetectingPose:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.guard = threading.Lock()

        def process(self, image):
            with self.guard:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.03)
            with self.guard:
                self.active -= 1
            return SimpleNamespace(pose_landmarks=None)

    detector = OverlapDetectingPose()
    monkeypatch.setattr(cycling, "get_pose_detector", lambda video_mode: detector)
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    barrier = threading.Barrier(3)

    def run_pose():
        barrier.wait()
        _extract_pose(frame, None, {})

    threads = [threading.Thread(target=run_pose) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert detector.max_active == 1
