"""
Reusable keypoint fixtures for unit tests.

All fixtures return lists of per-frame dicts with MediaPipe-style
keypoint data (integer landmark index → {x, y, visibility}).
"""

from __future__ import annotations

import math
from typing import Dict, List


def _lm(x: float, y: float, vis: float = 0.9) -> Dict:
    return {"x": x, "y": y, "visibility": vis}


def make_empty_frames(n: int = 30) -> List[Dict]:
    """Return *n* frames with no landmarks."""
    return [{} for _ in range(n)]


def make_static_standing_frames(n: int = 60) -> List[Dict]:
    """Return frames representing a person standing still.

    Landmarks placed at anatomically plausible normalized coordinates.
    """
    base = {
        0: _lm(0.50, 0.05),  # nose
        7: _lm(0.46, 0.05),  # left_ear
        8: _lm(0.54, 0.05),  # right_ear
        11: _lm(0.40, 0.25),  # left_shoulder
        12: _lm(0.60, 0.25),  # right_shoulder
        13: _lm(0.35, 0.42),  # left_elbow
        14: _lm(0.65, 0.42),  # right_elbow
        15: _lm(0.32, 0.57),  # left_wrist
        16: _lm(0.68, 0.57),  # right_wrist
        23: _lm(0.43, 0.55),  # left_hip
        24: _lm(0.57, 0.55),  # right_hip
        25: _lm(0.43, 0.73),  # left_knee
        26: _lm(0.57, 0.73),  # right_knee
        27: _lm(0.43, 0.90),  # left_ankle
        28: _lm(0.57, 0.90),  # right_ankle
    }
    return [dict(base) for _ in range(n)]


def make_running_frames(n: int = 90, fps: float = 30.0) -> List[Dict]:
    """Return frames simulating a running gait cycle.

    Alternates left/right ankle positions to trigger step detection.
    """
    frames = []
    for i in range(n):
        t = i / fps
        freq = 2.8  # ~168 steps/min
        phase = 2 * math.pi * freq * t
        left_ankle_y = 0.90 + 0.05 * math.sin(phase)
        right_ankle_y = 0.90 + 0.05 * math.sin(phase + math.pi)
        frames.append(
            {
                0: _lm(0.50, 0.05),
                11: _lm(0.40, 0.25),
                12: _lm(0.60, 0.25),
                13: _lm(0.35, 0.42),
                14: _lm(0.65, 0.42),
                15: _lm(0.32, 0.57),
                16: _lm(0.68, 0.57),
                23: _lm(0.43, 0.55),
                24: _lm(0.57, 0.55),
                25: _lm(0.43, 0.73),
                26: _lm(0.57, 0.73),
                27: _lm(0.43, left_ankle_y),
                28: _lm(0.57, right_ankle_y),
            }
        )
    return frames


def make_cycling_frames(n: int = 90, fps: float = 30.0) -> List[Dict]:
    """Return frames simulating a pedal stroke cycle (~90 RPM)."""
    frames = []
    for i in range(n):
        t = i / fps
        freq = 1.5  # 90 RPM = 1.5 Hz
        phase = 2 * math.pi * freq * t
        left_knee_y = 0.65 + 0.15 * math.sin(phase)
        right_knee_y = 0.65 + 0.15 * math.sin(phase + math.pi)
        left_ankle_y = 0.80 + 0.12 * math.sin(phase + 0.2)
        right_ankle_y = 0.80 + 0.12 * math.sin(phase + math.pi + 0.2)
        frames.append(
            {
                11: _lm(0.40, 0.30),
                12: _lm(0.60, 0.30),
                23: _lm(0.43, 0.50),
                24: _lm(0.57, 0.50),
                25: _lm(0.43, left_knee_y),
                26: _lm(0.57, right_knee_y),
                27: _lm(0.43, left_ankle_y),
                28: _lm(0.57, right_ankle_y),
            }
        )
    return frames
