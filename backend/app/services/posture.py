"""Pure postural coordinate calculations from normalized pose landmarks."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping

Point = Mapping[str, float]

GOOD_AXIS_DEG = 2.0
WARNING_AXIS_DEG = 7.0
GOOD_TRUNK_DEG = 2.0
WARNING_TRUNK_DEG = 6.0
GOOD_OFFSET_PCT = 2.0
WARNING_OFFSET_PCT = 5.0


def _midpoint(first: Point, second: Point) -> Dict[str, float]:
    return {
        "x": round((float(first["x"]) + float(second["x"])) / 2.0, 4),
        "y": round((float(first["y"]) + float(second["y"])) / 2.0, 4),
    }


def _axis_angle(left: Point, right: Point) -> float:
    return round(
        math.degrees(
            math.atan2(
                float(right["y"]) - float(left["y"]),
                float(right["x"]) - float(left["x"]),
            )
        ),
        1,
    )


def _severity(value: float, good_limit: float, warning_limit: float) -> str:
    magnitude = abs(value)
    if magnitude <= good_limit:
        return "good"
    if magnitude <= warning_limit:
        return "moderate"
    return "warning"


def _higher_side(left: Point, right: Point) -> str:
    difference = float(right["y"]) - float(left["y"])
    if abs(difference) < 0.002:
        return "level"
    return "left" if difference > 0 else "right"


def calculate_posture(
    landmarks: Mapping[str, Point],
    frame_center_x: float = 0.5,
) -> Dict[str, Any]:
    """Return postural axes and deviations from normalized image landmarks."""
    required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if any(name not in landmarks for name in required):
        return {"available": False}

    left_shoulder = landmarks["left_shoulder"]
    right_shoulder = landmarks["right_shoulder"]
    left_hip = landmarks["left_hip"]
    right_hip = landmarks["right_hip"]
    shoulder_mid = _midpoint(left_shoulder, right_shoulder)
    hip_mid = _midpoint(left_hip, right_hip)

    shoulder_angle = _axis_angle(left_shoulder, right_shoulder)
    pelvis_angle = _axis_angle(left_hip, right_hip)
    trunk_dx = hip_mid["x"] - shoulder_mid["x"]
    trunk_dy = hip_mid["y"] - shoulder_mid["y"]
    trunk_lean = round(math.degrees(math.atan2(trunk_dx, max(abs(trunk_dy), 1e-6))), 1)
    trunk_center_x = (shoulder_mid["x"] + hip_mid["x"]) / 2.0
    trunk_offset = round((trunk_center_x - frame_center_x) * 100.0, 1)
    points = {
        name: {"x": round(float(point["x"]), 4), "y": round(float(point["y"]), 4)}
        for name, point in landmarks.items()
        if "x" in point and "y" in point
    }
    points.update({"shoulder_mid": shoulder_mid, "hip_mid": hip_mid})

    return {
        "available": True,
        "points": points,
        "shoulder_angle_deg": shoulder_angle,
        "pelvis_angle_deg": pelvis_angle,
        "trunk_lean_deg": trunk_lean,
        "trunk_offset_pct": trunk_offset,
        "shoulder": {
            "severity": _severity(shoulder_angle, GOOD_AXIS_DEG, WARNING_AXIS_DEG),
            "higher_side": _higher_side(left_shoulder, right_shoulder),
        },
        "pelvis": {
            "severity": _severity(pelvis_angle, GOOD_AXIS_DEG, WARNING_AXIS_DEG),
            "higher_side": _higher_side(left_hip, right_hip),
        },
        "trunk": {
            "severity": max(
                _severity(trunk_lean, GOOD_TRUNK_DEG, WARNING_TRUNK_DEG),
                _severity(trunk_offset, GOOD_OFFSET_PCT, WARNING_OFFSET_PCT),
                key=("good", "moderate", "warning").index,
            ),
            "lean_direction": "right" if trunk_lean > 0 else "left" if trunk_lean < 0 else "center",
            "offset_direction": "right" if trunk_offset > 0 else "left" if trunk_offset < 0 else "center",
        },
    }
