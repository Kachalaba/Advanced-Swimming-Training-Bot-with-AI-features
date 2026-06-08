"""Unit tests for the stateful WebRTC rehabilitation processor."""

import pytest

# PyAV and streamlit-webrtc are heavy, optional dependencies. Skip gracefully
# (rather than aborting the whole unit-test session on a collection error) when
# they are absent — matching the cv2 importorskip convention in the other tests.
av = pytest.importorskip("av")
pytest.importorskip("streamlit_webrtc")

import numpy as np  # noqa: E402

from tests.fixtures.mock_keypoints import make_rehab_shoulder_flexion_frames  # noqa: E402
from video_analysis.live_rehab import LiveRehabProcessor  # noqa: E402


class _FakeVisualizer:
    def __init__(self, keypoints):
        self.keypoints = iter(keypoints)

    def process_frame(self, frame, index, bbox=None):
        return frame.copy(), {
            "has_pose": True,
            "keypoints": next(self.keypoints),
        }


class _FakeAnalyzer:
    def __init__(self):
        self.calls = []

    def analyze(self, keypoints, protocol):
        self.calls.append((list(keypoints), protocol))
        return {
            "protocol": protocol,
            "total_correct_reps": 1,
            "target_metrics": {
                "left": {"rom": 120.0},
                "right": {"rom": 110.0},
            },
            "symmetry": {"asymmetry_index": 8.3, "score": 91.7},
        }


def test_live_processor_analyzes_bounded_frame_window():
    visualizer = _FakeVisualizer(make_rehab_shoulder_flexion_frames(cycles=1))
    analyzer = _FakeAnalyzer()
    processor = LiveRehabProcessor(
        analyzer=analyzer,
        visualizer=visualizer,
        protocol="shoulder_flexion",
        labels={"collecting": "Collecting", "reps": "Reps", "asymmetry": "Asymmetry"},
        max_frames=3,
        analysis_interval=2,
    )

    for _ in range(4):
        frame = av.VideoFrame.from_ndarray(np.zeros((120, 160, 3), dtype=np.uint8), format="bgr24")
        output = processor.recv(frame)
        assert output.width == 160
        assert output.height == 120

    report = processor.get_report()
    assert report is not None
    assert report["protocol"] == "shoulder_flexion"
    assert len(analyzer.calls[-1][0]) == 3
    assert analyzer.calls[-1][1] == "shoulder_flexion"
