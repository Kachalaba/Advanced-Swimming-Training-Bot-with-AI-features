"""Pipeline contract tests for waterline-aware swimming analysis."""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from backend.app.services.swimming import ObservationBatch, _pose_landmarks, analyze_swimming_video
from video_analysis.waterline_analyzer import WaterlineEstimate


def _zone(issue_code=None, score=88.0):
    return {
        "available": True,
        "prerequisites_met": True,
        "confidence_inputs": {
            "landmark_visibility": 0.9,
            "temporal_continuity": 0.9,
            "waterline_clarity": 0.85,
            "identity_stability": 0.9,
            "cycle_coverage": 1.0,
        },
        "score": score,
        "issue_code": issue_code,
        "impact": 0.9 if issue_code else 0.0,
        "metrics": {"hip_drop_deg": 11.0} if issue_code == "hips_drop" else {},
    }


def _fake_batch(cycles: int, fps: float = 10.0) -> ObservationBatch:
    period_sec = 1.2
    frame_count = int((cycles + 1.25) * period_sec * fps)
    observations = []
    for frame_index in range(frame_count):
        t = frame_index / fps
        phase = 2.0 * math.pi * t / period_sec
        observations.append(
            {
                "frame_index": frame_index,
                "timestamp": t,
                "landmarks": {
                    "left_shoulder": {"x": 0.4, "y": 0.45, "visibility": 0.95},
                    "right_shoulder": {"x": 0.6, "y": 0.45, "visibility": 0.95},
                    "left_wrist": {
                        "x": 0.25,
                        "y": 0.45 + 0.25 * math.sin(phase),
                        "visibility": 0.95,
                    },
                    "right_wrist": {
                        "x": 0.75,
                        "y": 0.45 + 0.25 * math.sin(phase + math.pi),
                        "visibility": 0.95,
                    },
                },
                "tracking_confidence": 0.95,
                "waterline_confidence": 0.9,
                "waterline": WaterlineEstimate(
                    slope=0.01,
                    intercept=82.0 + math.sin(frame_index / 9.0),
                    confidence=0.9,
                    observed=frame_index % 5 != 0,
                ),
                "blur_quality": 0.9,
            }
        )
    cycle_metrics = []
    for index in range(cycles):
        peak = 0.3 + index * period_sec
        cycle_metrics.append(
            {
                "cycle_id": f"cycle-{index + 1}",
                "start_sec": peak,
                "peak_sec": peak + period_sec / 2,
                "end_sec": peak + period_sec,
                "zones": {
                    "body_position": _zone("hips_drop", 62.0),
                    "rotation": _zone(),
                    "catch": _zone(),
                    "breathing": _zone(),
                    "kick": _zone(),
                },
            }
        )
    return ObservationBatch(
        observations=observations,
        fps=fps,
        width=320,
        height=180,
        frames=[],
        quality_status="pass",
        quality_warnings=[],
        cycle_metrics=cycle_metrics,
    )


def _provider(cycles: int):
    batch = _fake_batch(cycles)

    def provide(video_path: Path, output_dir: Path, fps):
        return batch

    return provide


def _fake_renderer(batch, selected_cycles, technique, output_path):
    output_path.write_bytes(b"annotated")
    return output_path


def test_pipeline_emits_stable_stages_and_structured_result(tmp_path):
    events = list(
        analyze_swimming_video(
            video_path=tmp_path / "input.mp4",
            output_dir=tmp_path,
            observation_provider=_provider(cycles=4),
            render_video=_fake_renderer,
        )
    )

    assert [event.stage for event in events if event.type == "progress"] == [
        "quality_gate",
        "tracking",
        "waterline",
        "pose",
        "cycles",
        "technique",
        "coaching",
        "rendering",
        "completed",
    ]
    result = events[-1].to_dict()
    assert result["type"] == "result"
    assert result["analysis_type"] == "swimming_freestyle_side"
    assert result["coverage"]["total_zones"] == 5
    assert len(result["cycles"]) == 4
    assert result["primary_issue"]["issue_code"] == "hips_drop"
    assert result["video_path"] == "annotated.mp4"
    assert result["waterline_baseline"]["available"] is True
    assert abs(result["waterline_baseline"]["position_y_pct"] - 46.4) <= 0.2
    assert result["waterline_baseline"]["observed_coverage_pct"] >= 75.0
    assert result["waterline_baseline"]["confidence_pct"] == 90.0
    assert result["waterline_baseline"]["drift_pct"] < 2.0


def test_pipeline_returns_reshoot_error_for_one_cycle(tmp_path):
    events = list(
        analyze_swimming_video(
            video_path=tmp_path / "input.mp4",
            output_dir=tmp_path,
            observation_provider=_provider(cycles=1),
            render_video=_fake_renderer,
        )
    )

    error = events[-1].to_dict()
    assert error["type"] == "error"
    assert error["code"] == "insufficient_cycles"
    assert "complete cycles" in error["message"]
    assert error["reshoot_guidance"]


def test_pipeline_preserves_partial_quality_warnings(tmp_path):
    batch = _fake_batch(cycles=3)
    batch.quality_status = "partial"
    batch.quality_warnings = ["Feet leave the frame"]

    events = list(
        analyze_swimming_video(
            video_path=tmp_path / "input.mp4",
            output_dir=tmp_path,
            observation_provider=lambda video_path, output_dir, fps: batch,
            render_video=_fake_renderer,
        )
    )

    result = events[-1].to_dict()
    assert result["quality"]["status"] == "partial"
    assert result["quality"]["warnings"] == ["Feet leave the frame"]


def test_shared_pose_detector_calls_are_serialized():
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
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    waterline = WaterlineEstimate(slope=0.0, intercept=40.0, confidence=0.9, observed=True)
    barrier = threading.Barrier(3)

    def run_pose():
        barrier.wait()
        _pose_landmarks(frame, (10, 10, 110, 70), waterline, detector)

    threads = [threading.Thread(target=run_pose) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert detector.max_active == 1
