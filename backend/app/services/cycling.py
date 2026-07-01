"""Side-view cycling analysis pipeline for the FastAPI product surface."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Union

import cv2
import numpy as np

from video_analysis.base_analyzer import get_pose_detector, get_pose_processing_lock
from video_analysis.cycling_analyzer import CyclingAnalysis, CyclingAnalyzer
from video_analysis.frame_extractor import extract_frames_from_video
from video_analysis.swimmer_detector import detect_swimmer_in_frames

from .video_encoding import finalize_browser_video, open_intermediate_writer

logger = logging.getLogger(__name__)

MIN_LANDMARK_VISIBILITY = 0.4
MIN_POSE_FRAMES = 20
MIN_POSE_COVERAGE = 35.0
BBOX_PADDING_PCT = 18
EMA_ALPHA = 0.65
YOLO_CONFIDENCE = 0.25
_POSE_PROCESSING_LOCK = get_pose_processing_lock()

LANDMARK_MAP = {
    0: "nose",
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


def cycling_quality(frames_total: int, frames_with_pose: int) -> dict[str, Any]:
    """Return an explicit confidence gate for side-view bike-fit metrics."""

    coverage = round(frames_with_pose / frames_total * 100, 1) if frames_total > 0 else 0.0
    warnings = []
    if frames_with_pose < MIN_POSE_FRAMES:
        warnings.append("Too few frames contain a complete cycling pose.")
    if coverage < MIN_POSE_COVERAGE:
        warnings.append("Pose coverage is too low for reliable bike-fit metrics.")
    elif coverage < 70:
        warnings.append("Pose coverage is usable but below the preferred 70%.")

    return {
        "status": "fail" if frames_with_pose < MIN_POSE_FRAMES or coverage < MIN_POSE_COVERAGE else "pass",
        "pose_coverage": coverage,
        "warnings": warnings,
    }


def _metric_ready(keypoints: dict[str, tuple[float, float]]) -> bool:
    """Require one complete side plus a torso anchor before scoring a frame."""

    left_ready = all(name in keypoints for name in ("left_shoulder", "left_hip", "left_knee", "left_ankle"))
    right_ready = all(
        name in keypoints
        for name in (
            "right_shoulder",
            "right_hip",
            "right_knee",
            "right_ankle",
        )
    )
    return left_ready or right_ready


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


def analyze_cycling_video(
    video_path: Path,
    output_dir: Path,
    fps: float = 30.0,
) -> Iterator[Event]:
    """Analyze a side-view cycling clip and produce browser-ready evidence."""

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    detections_dir = output_dir / "detections"

    yield ProgressEvent(2, "Extracting cycling frames")
    frame_result = extract_frames_from_video(
        str(video_path),
        output_dir=str(frames_dir),
        fps=fps,
    )
    frames = frame_result["frames"]
    if not frames:
        yield ErrorEvent("No frames could be extracted from this video.")
        return

    first_frame = cv2.imread(_frame_path(frames[0]))
    if first_frame is None:
        yield ErrorEvent("The first video frame could not be decoded.")
        return
    height, width = first_frame.shape[:2]

    yield ProgressEvent(15, "Locking onto the cyclist")
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
    total = len(frames)

    yield ProgressEvent(28, "Measuring cycling posture")
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
        if _metric_ready(keypoints):
            frames_with_pose += 1
            keypoints_list.append(keypoints)
        else:
            keypoints_list.append({})

        if index % 20 == 0:
            pct = 28 + int(34 * index / max(1, total))
            yield ProgressEvent(
                pct,
                f"Pose frame {index + 1}/{total}",
            )

    quality = cycling_quality(total, frames_with_pose)
    if quality["status"] == "fail":
        guidance = "Record 15–30 seconds from a fixed side view with the full rider, " "crank and both wheels visible."
        yield ErrorEvent(f"{quality['warnings'][0]} {guidance}")
        return

    yield ProgressEvent(66, "Computing bike-fit metrics")
    analyzer = CyclingAnalyzer(fps=fps)
    analysis = analyzer.analyze(keypoints_list, fps=fps)

    yield ProgressEvent(76, "Encoding annotated evidence")
    annotated_path = output_dir / "annotated.mp4"
    try:
        writer, intermediate_path = open_intermediate_writer(
            annotated_path,
            fps=fps,
            frame_size=(width, height),
        )
    except RuntimeError as exc:
        yield ErrorEvent(str(exc))
        return

    for frame_info, keypoints in zip(frames, keypoints_list, strict=False):
        frame = cv2.imread(_frame_path(frame_info))
        if frame is None:
            continue
        writer.write(analyzer.draw_overlay(frame, keypoints, analysis))
    writer.release()

    try:
        finalize_browser_video(intermediate_path, annotated_path)
    except RuntimeError as exc:
        yield ErrorEvent(str(exc))
        return

    payload = {
        "analysis": _serialize_analysis(analysis),
        "frames_total": total,
        "frames_with_pose": frames_with_pose,
        "quality": quality,
        "video_path": str(annotated_path.relative_to(output_dir)),
    }
    yield ProgressEvent(100, "Cycling analysis complete")
    yield ResultEvent(payload)


def _serialize_analysis(analysis: CyclingAnalysis) -> dict[str, Any]:
    return {key: _safe(value) for key, value in asdict(analysis).items()}


def _safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if hasattr(value, "value"):
        return value.value
    return str(value)
