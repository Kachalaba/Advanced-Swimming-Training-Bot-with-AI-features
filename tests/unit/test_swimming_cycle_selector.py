"""Tests for selecting the clearest complete freestyle cycles."""

from __future__ import annotations

import math

from video_analysis.swimming_cycle_selector import SwimmingCycleSelector


def _synthetic_freestyle_landmarks(cycles: int, fps: float = 10.0):
    period_sec = 1.2
    frame_count = int((cycles + 1.25) * period_sec * fps)
    frames = []
    for frame_index in range(frame_count):
        t = frame_index / fps
        phase = 2.0 * math.pi * t / period_sec
        frames.append(
            {
                "frame_index": frame_index,
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
                "blur_quality": 0.9,
            }
        )
    return frames


def test_selects_best_non_overlapping_complete_cycles():
    frames = _synthetic_freestyle_landmarks(cycles=6)

    cycles = SwimmingCycleSelector(fps=10.0).select(frames, limit=5)

    assert len(cycles) == 5
    assert all(cycle.complete for cycle in cycles)
    assert all(cycles[index].end_frame <= cycles[index + 1].start_frame for index in range(4))
    assert all(cycle.quality >= 0.78 for cycle in cycles)


def test_cycle_quality_penalizes_tracking_and_waterline_gaps():
    clean = _synthetic_freestyle_landmarks(cycles=3)
    degraded = _synthetic_freestyle_landmarks(cycles=3)
    for frame in degraded[12:24]:
        frame["tracking_confidence"] = 0.2
        frame["waterline_confidence"] = 0.2
        frame["landmarks"].pop("left_wrist")

    clean_cycles = SwimmingCycleSelector(fps=10.0).select(clean)
    degraded_cycles = SwimmingCycleSelector(fps=10.0).select(degraded)

    clean_match = min(clean_cycles, key=lambda cycle: abs(cycle.start_frame - 15))
    degraded_match = min(degraded_cycles, key=lambda cycle: abs(cycle.start_frame - 15))
    assert clean_match.quality > degraded_match.quality


def test_fewer_than_two_reliable_cycles_remains_insufficient():
    frames = _synthetic_freestyle_landmarks(cycles=1)

    result = SwimmingCycleSelector(fps=10.0).select(frames)

    assert len(result) < 2


def test_cycle_timestamps_are_derived_from_source_fps():
    frames = _synthetic_freestyle_landmarks(cycles=2, fps=20.0)

    cycle = SwimmingCycleSelector(fps=20.0).select(frames)[0]

    assert cycle.start_sec == cycle.start_frame / 20.0
    assert cycle.peak_sec == cycle.peak_frame / 20.0
    assert cycle.end_sec == cycle.end_frame / 20.0


def test_bridged_landmarks_cannot_establish_a_stroke_cycle():
    frames = _synthetic_freestyle_landmarks(cycles=3)
    for frame in frames:
        for landmark in frame["landmarks"].values():
            landmark["state"] = "bridged"

    result = SwimmingCycleSelector(fps=10.0).select(frames)

    assert result == []
