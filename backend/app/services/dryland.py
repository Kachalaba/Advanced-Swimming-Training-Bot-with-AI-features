"""Dryland exercise analysis pipeline for the FastAPI product surface."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Union

import cv2
import numpy as np

from video_analysis.base_analyzer import BaseAnalyzer, get_pose_detector, get_pose_processing_lock
from video_analysis.constants import (
    DRYLAND_MIN_POSE_COVERAGE_PCT,
    DRYLAND_MIN_VALID_FRAMES,
    DRYLAND_SUPPORTED_EXERCISES,
    DRYLAND_WARN_POSE_COVERAGE_PCT,
)
from video_analysis.exercise_analyzer import ExerciseAnalyzer
from video_analysis.frame_extractor import extract_frames_from_video
from video_analysis.swimmer_detector import detect_swimmer_in_frames

from .video_encoding import finalize_browser_video, open_intermediate_writer

logger = logging.getLogger(__name__)

MIN_LANDMARK_VISIBILITY = 0.4
BBOX_PADDING_PCT = 18
EMA_ALPHA = 0.65
YOLO_CONFIDENCE = 0.25
_POSE_PROCESSING_LOCK = get_pose_processing_lock()
_ANGLE_HELPER = BaseAnalyzer()

LANDMARK_MAP = {
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}


@dataclass
class ProgressEvent:
    pct: int
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": "progress", "pct": self.pct, "label": self.label}


@dataclass
class ResultEvent:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"type": "result", **self.payload}


@dataclass
class ErrorEvent:
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": "error", "message": self.message}


Event = Union[ProgressEvent, ResultEvent, ErrorEvent]


def dryland_quality(frames_total: int, frames_with_pose: int, metric_ready_frames: int) -> dict[str, Any]:
    """Return an explicit confidence gate for Dryland repetition metrics."""

    pose_coverage = round(frames_with_pose / frames_total * 100, 1) if frames_total else 0.0
    warnings: list[str] = []
    if metric_ready_frames < DRYLAND_MIN_VALID_FRAMES:
        warnings.append("Too few metric-ready frames for a reliable dryland result.")
    if pose_coverage < DRYLAND_MIN_POSE_COVERAGE_PCT:
        warnings.append("Pose coverage is too low for dryland analysis.")
    elif pose_coverage < DRYLAND_WARN_POSE_COVERAGE_PCT:
        warnings.append("Pose coverage is usable but below the preferred 70%.")

    status = (
        "fail"
        if metric_ready_frames < DRYLAND_MIN_VALID_FRAMES or pose_coverage < DRYLAND_MIN_POSE_COVERAGE_PCT
        else "pass"
    )
    return {
        "status": status,
        "pose_coverage": pose_coverage,
        "metric_ready_frames": metric_ready_frames,
        "minimum_required_frames": DRYLAND_MIN_VALID_FRAMES,
        "warnings": warnings,
    }


def metric_ready(exercise_type: str, keypoints: dict[str, tuple[float, float]]) -> bool:
    """Require exercise-specific evidence before scoring a frame."""

    if exercise_type not in DRYLAND_SUPPORTED_EXERCISES:
        return False

    left_side = all(name in keypoints for name in ("left_shoulder", "left_hip", "left_knee", "left_ankle"))
    right_side = all(name in keypoints for name in ("right_shoulder", "right_hip", "right_knee", "right_ankle"))
    if exercise_type == "squat":
        return left_side or right_side

    if exercise_type == "lunge":
        torso = ("left_shoulder" in keypoints and "left_hip" in keypoints) or (
            "right_shoulder" in keypoints and "right_hip" in keypoints
        )
        return torso and all(name in keypoints for name in ("left_knee", "left_ankle", "right_knee", "right_ankle"))

    if exercise_type == "push_up":
        left_push = all(
            name in keypoints for name in ("left_shoulder", "left_elbow", "left_wrist", "left_hip", "left_ankle")
        )
        right_push = all(
            name in keypoints for name in ("right_shoulder", "right_elbow", "right_wrist", "right_hip", "right_ankle")
        )
        return left_push or right_push

    return False


def select_active_angles(exercise_type: str, keypoints_list: list[dict[str, tuple[float, float]]]) -> list[dict[str, float]]:
    """Convert frame keypoints into the angle dicts consumed by ExerciseAnalyzer."""

    frames: list[dict[str, float]] = []
    for keypoints in keypoints_list:
        frame_angles: dict[str, float] = {}
        if exercise_type in {"squat", "lunge"}:
            for prefix, side in (("L", "left"), ("R", "right")):
                names = (f"{side}_hip", f"{side}_knee", f"{side}_ankle")
                if all(name in keypoints for name in names):
                    frame_angles[f"{prefix}.knee"] = _ANGLE_HELPER._calculate_angle(
                        keypoints[names[0]],
                        keypoints[names[1]],
                        keypoints[names[2]],
                    )
        if exercise_type == "push_up":
            for prefix, side in (("L", "left"), ("R", "right")):
                names = (f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
                if all(name in keypoints for name in names):
                    frame_angles[f"{prefix}.elbow"] = _ANGLE_HELPER._calculate_angle(
                        keypoints[names[0]],
                        keypoints[names[1]],
                        keypoints[names[2]],
                    )
        frames.append(frame_angles)
    return frames


def _frame_path(frame_info: Any) -> str:
    return str(Path(frame_info["path"] if isinstance(frame_info, dict) else frame_info))


def _crop_bounds(
    bbox: list[float] | tuple[float, ...] | None,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    if not bbox:
        return 0, 0, width, height
    x1, y1, x2, y2 = (int(value) for value in bbox[:4])
    pad_x = int(max(1, x2 - x1) * BBOX_PADDING_PCT / 100)
    pad_y = int(max(1, y2 - y1) * BBOX_PADDING_PCT / 100)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def _extract_pose(
    frame: np.ndarray,
    bbox: list[float] | tuple[float, ...] | None,
    ema_state: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _crop_bounds(bbox, width, height)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return {}

    pose = get_pose_detector(video_mode=True)
    with _POSE_PROCESSING_LOCK:
        result = pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    if not result.pose_landmarks:
        return {}

    crop_h, crop_w = crop.shape[:2]
    keypoints: dict[str, tuple[float, float]] = {}
    for index, name in LANDMARK_MAP.items():
        landmark = result.pose_landmarks.landmark[index]
        if landmark.visibility < MIN_LANDMARK_VISIBILITY:
            continue
        x = x1 + landmark.x * crop_w
        y = y1 + landmark.y * crop_h
        previous = ema_state.get(name)
        if previous is not None:
            x = EMA_ALPHA * x + (1 - EMA_ALPHA) * previous[0]
            y = EMA_ALPHA * y + (1 - EMA_ALPHA) * previous[1]
        ema_state[name] = (x, y)
        keypoints[name] = (x, y)
    return keypoints


def _draw_annotation(
    frame: np.ndarray,
    keypoints: dict[str, tuple[float, float]],
    stats: Any,
    frame_idx: int,
) -> np.ndarray:
    output = frame.copy()
    for x, y in keypoints.values():
        cv2.circle(output, (int(x), int(y)), 4, (139, 92, 246), -1)

    rep_number = 0
    for rep in stats.reps:
        if rep.start_frame <= frame_idx <= rep.end_frame:
            rep_number = rep.rep_number
            break

    cv2.rectangle(output, (24, 24), (390, 116), (15, 23, 42), -1)
    cv2.putText(
        output,
        f"Dryland reps: {rep_number}/{stats.total_reps}",
        (42, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        output,
        f"ROM: {stats.avg_range_of_motion:.1f} deg",
        (42, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (196, 181, 253),
        2,
    )
    return output


def analyze_dryland_video(
    video_path: Path,
    output_dir: Path,
    exercise_type: str,
    fps: float = 15.0,
) -> Iterator[Event]:
    """Analyze a fixed side-view Dryland clip and produce browser-ready evidence."""

    if exercise_type not in DRYLAND_SUPPORTED_EXERCISES:
        yield ErrorEvent(f"Unsupported dryland exercise: {exercise_type}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    detections_dir = output_dir / "detections"

    yield ProgressEvent(2, "Extracting dryland frames")
    frame_result = extract_frames_from_video(str(video_path), output_dir=str(frames_dir), fps=fps)
    frames = frame_result["frames"]
    if not frames:
        yield ErrorEvent("No frames could be extracted from this video.")
        return

    first_frame = cv2.imread(_frame_path(frames[0]))
    if first_frame is None:
        yield ErrorEvent("The first video frame could not be decoded.")
        return
    height, width = first_frame.shape[:2]

    yield ProgressEvent(15, "Locking onto the athlete")
    detection_result = detect_swimmer_in_frames(
        frames,
        output_dir=str(detections_dir),
        draw_boxes=False,
        enable_tracking=True,
        confidence_threshold=YOLO_CONFIDENCE,
    )
    detections = detection_result.get("detections", [])

    keypoints_list: list[dict[str, tuple[float, float]]] = []
    ema_state: dict[str, tuple[float, float]] = {}
    previous_bbox: list[float] | tuple[float, ...] | None = None
    frames_with_pose = 0
    metric_ready_frames = 0
    total = len(frames)

    yield ProgressEvent(28, "Measuring dryland movement")
    for index, frame_info in enumerate(frames):
        frame = cv2.imread(_frame_path(frame_info))
        if frame is None:
            keypoints_list.append({})
            continue

        detection = detections[index] if index < len(detections) else {}
        bbox = detection.get("bbox") if isinstance(detection, dict) else None
        if bbox:
            previous_bbox = bbox

        keypoints = _extract_pose(frame, bbox or previous_bbox, ema_state)
        if keypoints:
            frames_with_pose += 1
        if metric_ready(exercise_type, keypoints):
            metric_ready_frames += 1
            keypoints_list.append(keypoints)
        else:
            keypoints_list.append({})

        if index % 20 == 0:
            yield ProgressEvent(
                28 + int(34 * index / max(1, total)),
                f"Pose frame {index + 1}/{total}",
            )

    quality = dryland_quality(total, frames_with_pose, metric_ready_frames)
    if quality["status"] == "fail":
        guidance = "Record a fixed side view with the selected exercise, full body, hands or feet, and the active joint visible."
        yield ErrorEvent(f"{quality['warnings'][0]} {guidance}")
        return

    yield ProgressEvent(66, "Detecting full repetitions")
    analyzer = ExerciseAnalyzer(fps=fps)
    stats = analyzer.analyze(select_active_angles(exercise_type, keypoints_list), exercise_type=exercise_type, fps=fps)
    if stats.total_reps == 0:
        yield ErrorEvent(
            "No complete repetition was confirmed. Start in the ready position, move through the effort phase, and return to ready."
        )
        return

    yield ProgressEvent(76, "Encoding annotated dryland evidence")
    annotated_path = output_dir / "annotated.mp4"
    try:
        writer, intermediate_path = open_intermediate_writer(annotated_path, fps=fps, frame_size=(width, height))
    except RuntimeError as exc:
        yield ErrorEvent(str(exc))
        return

    for index, (frame_info, keypoints) in enumerate(zip(frames, keypoints_list)):
        frame = cv2.imread(_frame_path(frame_info))
        if frame is None:
            continue
        writer.write(_draw_annotation(frame, keypoints, stats, index))
    writer.release()

    try:
        finalize_browser_video(intermediate_path, annotated_path)
    except RuntimeError as exc:
        yield ErrorEvent(str(exc))
        return

    yield ProgressEvent(100, "Dryland analysis complete")
    yield ResultEvent(
        {
            "exercise_type": exercise_type,
            "analysis": _safe(asdict(stats)),
            "quality": quality,
            "frames_total": total,
            "frames_with_pose": frames_with_pose,
            "video_path": str(annotated_path.relative_to(output_dir)),
        }
    )


def _safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    return str(value)
