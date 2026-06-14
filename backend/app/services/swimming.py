"""Waterline-aware side-view freestyle analysis pipeline."""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_analysis.base_analyzer import get_pose_detector, get_pose_processing_lock  # noqa: E402
from video_analysis.constants import (  # noqa: E402
    SWIM_CONFIDENCE_HIGH,
    SWIM_CONFIDENCE_MEDIUM,
    SWIM_MIN_PARTIAL_CYCLE_COUNT,
)
from video_analysis.frame_extractor import extract_frames_from_video  # noqa: E402
from video_analysis.swimmer_detector import detect_swimmer_in_frames  # noqa: E402
from video_analysis.swimming_cycle_selector import StrokeCycle, SwimmingCycleSelector  # noqa: E402
from video_analysis.swimming_technique_analyzer import SwimmingTechniqueAnalyzer  # noqa: E402
from video_analysis.waterline_analyzer import WaterlineAnalyzer, WaterlineEstimate  # noqa: E402

from .video_encoding import finalize_browser_video, open_intermediate_writer

logger = logging.getLogger(__name__)
_POSE_PROCESSING_LOCK = get_pose_processing_lock()

LANDMARK_NAMES = {
    0: "nose",
    7: "left_ear",
    8: "right_ear",
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

SKELETON_CONNECTIONS = (
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)


@dataclass
class ObservationBatch:
    observations: List[Dict[str, Any]]
    fps: float
    width: int
    height: int
    frames: List[np.ndarray]
    quality_status: str
    quality_warnings: List[str]
    cycle_metrics: Optional[List[Dict[str, Any]]] = None
    rejection: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    pct: int
    label: str
    type: str = field(default="progress", init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "stage": self.stage,
            "pct": self.pct,
            "label": self.label,
        }


@dataclass(frozen=True)
class ResultEvent:
    payload: Dict[str, Any]
    type: str = field(default="result", init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, **self.payload}


@dataclass(frozen=True)
class ErrorEvent:
    code: str
    message: str
    reshoot_guidance: str = ""
    type: str = field(default="error", init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "code": self.code,
            "message": self.message,
            "reshoot_guidance": self.reshoot_guidance,
        }


Event = Union[ProgressEvent, ResultEvent, ErrorEvent]
ObservationProvider = Callable[[Path, Path, Optional[float]], ObservationBatch]
VideoRenderer = Callable[
    [ObservationBatch, Sequence[StrokeCycle], Mapping[str, Any], Path],
    Path,
]


def analyze_swimming_video(
    video_path: Path,
    output_dir: Path,
    fps: Optional[float] = None,
    *,
    observation_provider: Optional[ObservationProvider] = None,
    render_video: Optional[VideoRenderer] = None,
) -> Iterator[Event]:
    """Run the complete side-view freestyle pipeline and yield transport-neutral events."""
    output_dir.mkdir(parents=True, exist_ok=True)
    provider = observation_provider or _collect_observations
    renderer = render_video or _render_annotated_video

    yield ProgressEvent("quality_gate", 5, "Checking video quality")
    yield ProgressEvent("tracking", 18, "Locking on to the primary swimmer")
    yield ProgressEvent("waterline", 32, "Estimating the waterline")
    yield ProgressEvent("pose", 48, "Tracking reliable body segments")

    try:
        batch = provider(video_path, output_dir, fps)
    except (ValueError, RuntimeError) as exc:
        yield ErrorEvent(
            code="video_processing_failed",
            message=str(exc),
            reshoot_guidance="Use a stable side-on clip in MP4 or MOV format with the full swimmer visible.",
        )
        return

    if batch.rejection:
        yield ErrorEvent(
            code=str(batch.rejection.get("code", "quality_rejected")),
            message=str(batch.rejection.get("message", "Video quality is insufficient for analysis.")),
            reshoot_guidance=str(batch.rejection.get("reshoot_guidance", "")),
        )
        return

    yield ProgressEvent("cycles", 62, "Selecting the clearest complete stroke cycles")
    selector = SwimmingCycleSelector(fps=batch.fps)
    selected_cycles = selector.select(batch.observations)
    if len(selected_cycles) < SWIM_MIN_PARTIAL_CYCLE_COUNT:
        yield ErrorEvent(
            code="insufficient_cycles",
            message="Fewer than two reliable complete cycles were found.",
            reshoot_guidance=(
                "Record at least four complete freestyle cycles from the side, "
                "keep the full swimmer in frame, and hold the phone steady."
            ),
        )
        return

    yield ProgressEvent("technique", 74, "Evaluating five technique zones")
    cycle_metrics = batch.cycle_metrics or _derive_cycle_metrics(batch.observations, selected_cycles)
    technique = SwimmingTechniqueAnalyzer().build_result(cycle_metrics)

    yield ProgressEvent("coaching", 84, "Building the corrective drill and mini-set")
    yield ProgressEvent("rendering", 91, "Rendering confidence-aware video evidence")
    annotated_path = output_dir / "annotated.mp4"
    try:
        rendered_path = renderer(batch, selected_cycles, technique, annotated_path)
    except RuntimeError as exc:
        yield ErrorEvent(code="render_failed", message=str(exc))
        return

    payload = {
        "analysis_type": "swimming_freestyle_side",
        "contract_version": "1.0",
        "quality": {
            "status": batch.quality_status,
            "warnings": list(batch.quality_warnings),
        },
        "waterline_baseline": _waterline_baseline(
            batch.observations,
            batch.width,
            batch.height,
        ),
        "coverage": technique["coverage"],
        "overall_score": technique["overall_score"],
        "cycles": [dict(cycle.to_dict()) for cycle in selected_cycles],
        "zones": technique["zones"],
        "primary_issue": technique["primary_issue"],
        "prescription": technique["prescription"],
        "frames_total": len(batch.observations),
        "frames_with_pose": sum(bool(item.get("landmarks")) for item in batch.observations),
        "video_path": str(rendered_path.relative_to(output_dir)),
    }
    yield ProgressEvent("completed", 100, "Analysis complete")
    yield ResultEvent(payload)


def _waterline_baseline(
    observations: Sequence[Mapping[str, Any]],
    frame_width: int,
    frame_height: int,
) -> Dict[str, Any]:
    """Summarize the temporal water surface used by the hybrid pose model."""

    estimates = [item.get("waterline") for item in observations if isinstance(item.get("waterline"), WaterlineEstimate)]
    usable = [estimate for estimate in estimates if estimate.confidence >= SWIM_CONFIDENCE_MEDIUM]
    total = len(observations)
    if not usable or frame_width <= 0 or frame_height <= 0:
        return {
            "available": False,
            "position_y_pct": None,
            "slope_pct": None,
            "confidence_pct": 0.0,
            "observed_coverage_pct": 0.0,
            "usable_coverage_pct": 0.0,
            "drift_pct": None,
        }

    centre_x = frame_width / 2.0
    positions = np.array(
        [estimate.y_at(centre_x) / frame_height * 100 for estimate in usable],
        dtype=float,
    )
    slopes = np.array(
        [estimate.slope * frame_width / frame_height * 100 for estimate in usable],
        dtype=float,
    )
    confidence = mean(estimate.confidence for estimate in usable) * 100
    observed = sum(estimate.observed for estimate in estimates)
    lower, upper = np.percentile(positions, [10, 90])

    return {
        "available": True,
        "position_y_pct": round(float(np.median(positions)), 1),
        "slope_pct": round(float(np.median(slopes)), 1),
        "confidence_pct": round(confidence, 1),
        "observed_coverage_pct": round(observed / max(total, 1) * 100, 1),
        "usable_coverage_pct": round(len(usable) / max(total, 1) * 100, 1),
        "drift_pct": round(float(upper - lower) / 2.0, 1),
    }


def _collect_observations(video_path: Path, output_dir: Path, fps: Optional[float]) -> ObservationBatch:
    source_fps, width, height = _probe_video(video_path)
    analysis_fps = max(8.0, min(float(fps or source_fps or 15.0), 20.0))
    frames_result = extract_frames_from_video(
        str(video_path),
        output_dir=str(output_dir / "frames"),
        fps=analysis_fps,
    )
    frame_infos = list(frames_result["frames"])
    if not frame_infos:
        raise ValueError("No readable frames were extracted from the video.")

    detection_result = detect_swimmer_in_frames(
        frame_infos,
        output_dir=str(output_dir / "detections"),
        draw_boxes=False,
        enable_tracking=True,
        confidence_threshold=0.25,
    )
    detections = list(detection_result["detections"])
    waterline_analyzer = WaterlineAnalyzer()
    pose_detector = get_pose_detector(video_mode=True)

    observations: List[Dict[str, Any]] = []
    frames: List[np.ndarray] = []
    previous_bbox: Optional[Tuple[int, int, int, int]] = None
    previous_landmarks: Dict[str, Dict[str, Any]] = {}
    lost_bbox_frames = 0
    lost_pose_frames = 0

    for index, frame_info in enumerate(frame_infos):
        frame = cv2.imread(str(frame_info["path"]))
        if frame is None:
            continue
        frames.append(frame)
        detection = detections[index] if index < len(detections) else {}
        bbox = _valid_bbox(detection.get("bbox"), width, height)
        if bbox is not None:
            previous_bbox = bbox
            lost_bbox_frames = 0
        elif previous_bbox is not None and lost_bbox_frames < 8:
            lost_bbox_frames += 1
            bbox = previous_bbox
        else:
            lost_bbox_frames += 1

        waterline = waterline_analyzer.analyze(frame, bbox)
        landmarks = _pose_landmarks(frame, bbox, waterline, pose_detector)
        if landmarks:
            previous_landmarks = landmarks
            lost_pose_frames = 0
        elif previous_landmarks and lost_pose_frames < 3:
            lost_pose_frames += 1
            landmarks = {
                name: {
                    **value,
                    "visibility": round(float(value["visibility"]) * (0.75**lost_pose_frames), 3),
                    "state": "bridged",
                }
                for name, value in previous_landmarks.items()
            }
        else:
            lost_pose_frames += 1

        blur_quality = _blur_quality(frame)
        brightness_quality = _brightness_quality(frame)
        tracking_confidence = float(detection.get("confidence") or 0.0)
        if bbox is not None and not detection.get("bbox"):
            tracking_confidence = max(0.25, 0.65 - 0.05 * lost_bbox_frames)

        observations.append(
            {
                "frame_index": index,
                "timestamp": float(frame_info.get("timestamp", index / analysis_fps)),
                "video_frame": int(frame_info.get("video_frame", index)),
                "frame_path": str(frame_info["path"]),
                "bbox": bbox,
                "landmarks": landmarks,
                "tracking_confidence": round(tracking_confidence, 3),
                "waterline": waterline,
                "waterline_confidence": waterline.confidence,
                "blur_quality": blur_quality,
                "brightness_quality": brightness_quality,
            }
        )

    return _quality_gate(
        observations=observations,
        frames=frames,
        fps=analysis_fps,
        width=width,
        height=height,
    )


def _probe_video(video_path: Path) -> Tuple[float, int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Could not open the uploaded video.")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    if width <= 0 or height <= 0:
        raise ValueError("Could not read video dimensions.")
    return fps, width, height


def _quality_gate(
    *,
    observations: List[Dict[str, Any]],
    frames: List[np.ndarray],
    fps: float,
    width: int,
    height: int,
) -> ObservationBatch:
    if not observations:
        return ObservationBatch(
            observations=[],
            fps=fps,
            width=width,
            height=height,
            frames=frames,
            quality_status="reject",
            quality_warnings=[],
            rejection={
                "code": "no_frames",
                "message": "No usable video frames were found.",
                "reshoot_guidance": "Export the clip as MP4 or MOV and try again.",
            },
        )

    tracking_coverage = mean(item["tracking_confidence"] >= 0.25 for item in observations)
    pose_coverage = mean(bool(item["landmarks"]) for item in observations)
    average_blur = mean(item["blur_quality"] for item in observations)
    average_brightness = mean(item["brightness_quality"] for item in observations)
    waterline_coverage = mean(item["waterline_confidence"] >= SWIM_CONFIDENCE_MEDIUM for item in observations)

    if tracking_coverage < 0.35 or pose_coverage < 0.2:
        return ObservationBatch(
            observations=observations,
            fps=fps,
            width=width,
            height=height,
            frames=frames,
            quality_status="reject",
            quality_warnings=[],
            rejection={
                "code": "swimmer_not_stable",
                "message": "The swimmer could not be tracked reliably.",
                "reshoot_guidance": (
                    "Move closer, keep the swimmer centred, and record from the side "
                    "without other people crossing the frame."
                ),
            },
        )
    if average_blur < 0.2 or average_brightness < 0.2:
        return ObservationBatch(
            observations=observations,
            fps=fps,
            width=width,
            height=height,
            frames=frames,
            quality_status="reject",
            quality_warnings=[],
            rejection={
                "code": "image_quality_low",
                "message": "The clip is too dark or blurred for reliable joint tracking.",
                "reshoot_guidance": "Use brighter lighting and keep the phone stable.",
            },
        )

    warnings = []
    if tracking_coverage < 0.7:
        warnings.append("Swimmer tracking is intermittent in part of the clip")
    if pose_coverage < 0.65:
        warnings.append("Some body segments are hidden by water or occlusion")
    if waterline_coverage < 0.6:
        warnings.append("Waterline confidence is limited in part of the clip")
    if _feet_coverage(observations) < 0.55:
        warnings.append("Feet leave the frame or are not visible consistently")
    return ObservationBatch(
        observations=observations,
        fps=fps,
        width=width,
        height=height,
        frames=frames,
        quality_status="partial" if warnings else "pass",
        quality_warnings=warnings,
    )


def _valid_bbox(
    raw_bbox: Any,
    frame_width: int,
    frame_height: int,
) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) < 4:
        return None
    x1, y1, x2, y2 = (int(value) for value in raw_bbox[:4])
    x1, x2 = max(0, x1), min(frame_width, x2)
    y1, y2 = max(0, y1), min(frame_height, y2)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    return x1, y1, x2, y2


def _pose_landmarks(
    frame: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
    waterline: WaterlineEstimate,
    pose_detector: Any,
) -> Dict[str, Dict[str, Any]]:
    if bbox is None:
        return {}
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x = int((x2 - x1) * 0.18)
    pad_y = int((y2 - y1) * 0.25)
    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    cx2, cy2 = min(frame_width, x2 + pad_x), min(frame_height, y2 + pad_y)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return {}

    enhanced = _enhance_water_zones(crop, waterline, cx1, cy1)
    with _POSE_PROCESSING_LOCK:
        result = pose_detector.process(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    if not result.pose_landmarks:
        return {}

    crop_height, crop_width = crop.shape[:2]
    landmarks: Dict[str, Dict[str, Any]] = {}
    for index, name in LANDMARK_NAMES.items():
        landmark = result.pose_landmarks.landmark[index]
        px = cx1 + landmark.x * crop_width
        py = cy1 + landmark.y * crop_height
        line_y = waterline.y_at(px)
        distance_ratio = abs(py - line_y) / max(frame_height, 1)
        proximity_penalty = 0.65 if distance_ratio < 0.04 else 1.0
        visibility = max(0.0, min(1.0, float(landmark.visibility) * proximity_penalty))
        if visibility < 0.25:
            continue
        if distance_ratio < 0.04:
            water_zone = "waterline"
        else:
            water_zone = "above" if py < line_y else "below"
        landmarks[name] = {
            "x": px / frame_width,
            "y": py / frame_height,
            "px": px,
            "py": py,
            "visibility": round(visibility, 3),
            "state": "observed",
            "water_zone": water_zone,
        }
    return landmarks


def _enhance_water_zones(
    crop: np.ndarray,
    waterline: WaterlineEstimate,
    offset_x: int,
    offset_y: int,
) -> np.ndarray:
    enhanced = crop.copy()
    height, width = crop.shape[:2]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    luminance, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    underwater = cv2.cvtColor(cv2.merge((clahe.apply(luminance), channel_a, channel_b)), cv2.COLOR_LAB2BGR)
    yy, xx = np.indices((height, width))
    line_y = waterline.slope * (xx + offset_x) + waterline.intercept - offset_y
    mask = yy >= line_y
    enhanced[mask] = underwater[mask]
    return enhanced


def _blur_quality(frame: np.ndarray) -> float:
    variance = float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    return round(max(0.0, min(1.0, variance / 180.0)), 3)


def _brightness_quality(frame: np.ndarray) -> float:
    brightness = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
    distance = abs(brightness - 130.0)
    return round(max(0.0, min(1.0, 1.0 - distance / 130.0)), 3)


def _feet_coverage(observations: Sequence[Mapping[str, Any]]) -> float:
    if not observations:
        return 0.0
    visible = 0
    for item in observations:
        landmarks = item.get("landmarks", {})
        if "left_ankle" in landmarks or "right_ankle" in landmarks:
            visible += 1
    return visible / len(observations)


def _derive_cycle_metrics(
    observations: Sequence[Mapping[str, Any]],
    cycles: Sequence[StrokeCycle],
) -> List[Dict[str, Any]]:
    return [_derive_single_cycle(observations, cycle) for cycle in cycles]


def _derive_single_cycle(
    observations: Sequence[Mapping[str, Any]],
    cycle: StrokeCycle,
) -> Dict[str, Any]:
    segment = list(observations[cycle.start_frame : cycle.end_frame + 1])
    base_confidence = {
        "temporal_continuity": _pose_coverage(segment),
        "waterline_clarity": _metadata_mean(segment, "waterline_confidence"),
        "identity_stability": _metadata_mean(segment, "tracking_confidence"),
        "cycle_coverage": cycle.quality,
    }

    body = _body_zone(segment, base_confidence)
    rotation = _rotation_zone(segment, base_confidence)
    catch = _catch_zone(segment, base_confidence)
    breathing = _breathing_zone(segment, base_confidence)
    kick = _kick_zone(segment, base_confidence)
    return {
        "cycle_id": cycle.id,
        "start_sec": cycle.start_sec,
        "peak_sec": cycle.peak_sec,
        "end_sec": cycle.end_sec,
        "zones": {
            "body_position": body,
            "rotation": rotation,
            "catch": catch,
            "breathing": breathing,
            "kick": kick,
        },
    }


def _body_zone(segment: Sequence[Mapping[str, Any]], base: Mapping[str, float]) -> Dict[str, Any]:
    required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_ankle", "right_ankle")
    samples = []
    visibility = []
    for item in segment:
        points = _points(item, required)
        if points is None:
            continue
        shoulder = _midpoint(points["left_shoulder"], points["right_shoulder"])
        hip = _midpoint(points["left_hip"], points["right_hip"])
        ankle = _midpoint(points["left_ankle"], points["right_ankle"])
        body_angle = _horizontal_angle(shoulder, ankle)
        hip_drop = _distance_to_line(hip, shoulder, ankle)
        samples.append((body_angle, hip_drop))
        visibility.append(_visibility(item, required))
    if not samples:
        return _unavailable_zone(base)
    body_angle = median(sample[0] for sample in samples)
    hip_drop = median(sample[1] for sample in samples)
    issue = "hips_drop" if hip_drop > 0.045 or body_angle > 10.0 else None
    score = max(0.0, 100.0 - hip_drop * 650.0 - max(0.0, body_angle - 5.0) * 1.5)
    return _zone_payload(
        base,
        visibility=mean(visibility),
        score=score,
        issue_code=issue,
        impact=0.95,
        metrics={"body_line_deg": body_angle, "hip_drop_ratio": hip_drop},
    )


def _rotation_zone(segment: Sequence[Mapping[str, Any]], base: Mapping[str, float]) -> Dict[str, Any]:
    required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    samples = []
    visibility = []
    for item in segment:
        points = _points(item, required)
        if points is None:
            continue
        shoulder_roll = abs(points["left_shoulder"][1] - points["right_shoulder"][1])
        hip_roll = abs(points["left_hip"][1] - points["right_hip"][1])
        samples.append((shoulder_roll, hip_roll))
        visibility.append(_visibility(item, required))
    if not samples:
        return _unavailable_zone(base)
    timing_proxy = abs(mean(sample[0] for sample in samples) - mean(sample[1] for sample in samples))
    issue = "rotation_asymmetry" if timing_proxy > 0.07 else None
    return _zone_payload(
        base,
        visibility=mean(visibility),
        score=max(0.0, 100.0 - timing_proxy * 500.0),
        issue_code=issue,
        impact=0.75,
        metrics={"shoulder_hip_rotation_offset": timing_proxy},
    )


def _catch_zone(segment: Sequence[Mapping[str, Any]], base: Mapping[str, float]) -> Dict[str, Any]:
    required = ("left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist")
    angles = []
    crossovers = []
    visibility = []
    for item in segment:
        points = _points(item, required)
        if points is None:
            continue
        angles.extend(
            (
                _angle(points["left_shoulder"], points["left_elbow"], points["left_wrist"]),
                _angle(points["right_shoulder"], points["right_elbow"], points["right_wrist"]),
            )
        )
        shoulder_mid_x = (points["left_shoulder"][0] + points["right_shoulder"][0]) / 2.0
        crossovers.append(
            min(
                abs(points["left_wrist"][0] - shoulder_mid_x),
                abs(points["right_wrist"][0] - shoulder_mid_x),
            )
        )
        visibility.append(_visibility(item, required))
    if not angles:
        return _unavailable_zone(base)
    elbow_angle = mean(angles)
    crossover_proxy = mean(crossovers)
    issue = "crossover_entry" if crossover_proxy < 0.025 else None
    score = max(0.0, 100.0 - abs(elbow_angle - 105.0) * 0.8 - max(0.0, 0.04 - crossover_proxy) * 400.0)
    return _zone_payload(
        base,
        visibility=mean(visibility),
        score=score,
        issue_code=issue,
        impact=0.85,
        metrics={"catch_elbow_deg": elbow_angle, "entry_crossover_proxy": crossover_proxy},
    )


def _breathing_zone(segment: Sequence[Mapping[str, Any]], base: Mapping[str, float]) -> Dict[str, Any]:
    required = ("nose", "left_shoulder", "right_shoulder")
    offsets = []
    visibility = []
    for item in segment:
        points = _points(item, required)
        if points is None:
            continue
        shoulder = _midpoint(points["left_shoulder"], points["right_shoulder"])
        offsets.append(abs(points["nose"][1] - shoulder[1]))
        visibility.append(_visibility(item, required))
    if len(offsets) < max(2, len(segment) // 4):
        return _unavailable_zone(base)
    late_offset = mean(offsets[-max(1, len(offsets) // 5) :])
    issue = "late_head_return" if late_offset > 0.16 else None
    return _zone_payload(
        base,
        visibility=mean(visibility),
        score=max(0.0, 100.0 - max(0.0, late_offset - 0.08) * 300.0),
        issue_code=issue,
        impact=0.7,
        metrics={"head_return_offset": late_offset},
    )


def _kick_zone(segment: Sequence[Mapping[str, Any]], base: Mapping[str, float]) -> Dict[str, Any]:
    required = ("left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle")
    knee_angles = []
    ankle_y = []
    visibility = []
    for item in segment:
        points = _points(item, required)
        if points is None:
            continue
        knee_angles.extend(
            (
                _angle(points["left_hip"], points["left_knee"], points["left_ankle"]),
                _angle(points["right_hip"], points["right_knee"], points["right_ankle"]),
            )
        )
        ankle_y.extend((points["left_ankle"][1], points["right_ankle"][1]))
        visibility.append(_visibility(item, required))
    if not knee_angles:
        return _unavailable_zone(base)
    knee_angle = mean(knee_angles)
    amplitude = max(ankle_y) - min(ankle_y)
    issue = "knee_driven_kick" if knee_angle < 135.0 else None
    return _zone_payload(
        base,
        visibility=mean(visibility),
        score=max(0.0, 100.0 - max(0.0, 150.0 - knee_angle) * 1.2),
        issue_code=issue,
        impact=0.65,
        metrics={"knee_angle_deg": knee_angle, "ankle_amplitude": amplitude},
    )


def _zone_payload(
    base: Mapping[str, float],
    *,
    visibility: float,
    score: float,
    issue_code: Optional[str],
    impact: float,
    metrics: Mapping[str, float],
) -> Dict[str, Any]:
    return {
        "available": True,
        "prerequisites_met": True,
        "confidence_inputs": {
            "landmark_visibility": max(0.0, min(1.0, visibility)),
            **base,
        },
        "score": round(max(0.0, min(100.0, score)), 1),
        "issue_code": issue_code,
        "impact": impact if issue_code else 0.0,
        "metrics": {key: round(float(value), 3) for key, value in metrics.items()},
    }


def _unavailable_zone(base: Mapping[str, float]) -> Dict[str, Any]:
    return {
        "available": False,
        "prerequisites_met": False,
        "confidence_inputs": {"landmark_visibility": 0.0, **base},
        "score": 0.0,
        "issue_code": None,
        "impact": 0.0,
        "metrics": {},
    }


def _points(
    observation: Mapping[str, Any],
    names: Sequence[str],
) -> Optional[Dict[str, Tuple[float, float]]]:
    landmarks = observation.get("landmarks", {})
    result = {}
    for name in names:
        landmark = landmarks.get(name)
        if not isinstance(landmark, Mapping) or landmark.get("state") == "bridged":
            return None
        if float(landmark.get("visibility", 0.0)) < SWIM_CONFIDENCE_MEDIUM:
            return None
        result[name] = (float(landmark["x"]), float(landmark["y"]))
    return result


def _visibility(observation: Mapping[str, Any], names: Sequence[str]) -> float:
    landmarks = observation.get("landmarks", {})
    return mean(float(landmarks[name]["visibility"]) for name in names)


def _pose_coverage(segment: Sequence[Mapping[str, Any]]) -> float:
    if not segment:
        return 0.0
    return mean(bool(item.get("landmarks")) for item in segment)


def _metadata_mean(segment: Sequence[Mapping[str, Any]], key: str) -> float:
    if not segment:
        return 0.0
    return mean(max(0.0, min(1.0, float(item.get(key, 0.0)))) for item in segment)


def _midpoint(first: Tuple[float, float], second: Tuple[float, float]) -> Tuple[float, float]:
    return (first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0


def _horizontal_angle(first: Tuple[float, float], second: Tuple[float, float]) -> float:
    angle = abs(math.degrees(math.atan2(second[1] - first[1], second[0] - first[0])))
    return min(angle, abs(180.0 - angle))


def _distance_to_line(
    point: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
) -> float:
    dx = line_end[0] - line_start[0]
    dy = line_end[1] - line_start[1]
    denominator = math.hypot(dx, dy)
    if denominator == 0:
        return 0.0
    numerator = abs(dy * point[0] - dx * point[1] + line_end[0] * line_start[1] - line_end[1] * line_start[0])
    return numerator / denominator


def _angle(
    first: Tuple[float, float],
    centre: Tuple[float, float],
    third: Tuple[float, float],
) -> float:
    vector_a = (first[0] - centre[0], first[1] - centre[1])
    vector_b = (third[0] - centre[0], third[1] - centre[1])
    denominator = math.hypot(*vector_a) * math.hypot(*vector_b)
    if denominator == 0:
        return 0.0
    cosine = max(-1.0, min(1.0, (vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]) / denominator))
    return math.degrees(math.acos(cosine))


def _render_annotated_video(
    batch: ObservationBatch,
    selected_cycles: Sequence[StrokeCycle],
    technique: Mapping[str, Any],
    output_path: Path,
) -> Path:
    if not batch.frames:
        raise RuntimeError("No frames are available for annotated video rendering.")
    writer, intermediate_path = open_intermediate_writer(
        output_path,
        fps=batch.fps,
        frame_size=(batch.width, batch.height),
    )
    cycle_frames = {
        frame_index for cycle in selected_cycles for frame_index in range(cycle.start_frame, cycle.end_frame + 1)
    }
    primary_title = ""
    if technique.get("primary_issue"):
        primary_title = str(technique["primary_issue"].get("title", ""))

    try:
        for index, source in enumerate(batch.frames):
            frame = source.copy()
            observation = batch.observations[index] if index < len(batch.observations) else {}
            _draw_waterline(frame, observation.get("waterline"))
            _draw_strict_skeleton(frame, observation.get("landmarks", {}))
            if index in cycle_frames:
                cv2.putText(
                    frame,
                    "Selected cycle",
                    (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (211, 211, 34),
                    2,
                    cv2.LINE_AA,
                )
            if primary_title:
                cv2.putText(
                    frame,
                    primary_title[:72],
                    (16, batch.height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (230, 230, 230),
                    1,
                    cv2.LINE_AA,
                )
            writer.write(frame)
    finally:
        writer.release()
    finalize_browser_video(intermediate_path, output_path)
    return output_path


def _draw_waterline(frame: np.ndarray, estimate: Any) -> None:
    if not isinstance(estimate, WaterlineEstimate) or estimate.confidence < SWIM_CONFIDENCE_MEDIUM:
        return
    height, width = frame.shape[:2]
    color = (211, 211, 34) if estimate.confidence >= SWIM_CONFIDENCE_HIGH else (0, 191, 255)
    first = (0, int(max(0, min(height - 1, estimate.y_at(0)))))
    second = (width - 1, int(max(0, min(height - 1, estimate.y_at(width - 1)))))
    cv2.line(frame, first, second, color, 1, cv2.LINE_AA)


def _draw_strict_skeleton(frame: np.ndarray, landmarks: Mapping[str, Any]) -> None:
    for start_name, end_name in SKELETON_CONNECTIONS:
        start = landmarks.get(start_name)
        end = landmarks.get(end_name)
        if not _renderable_landmark(start) or not _renderable_landmark(end):
            continue
        confidence = min(float(start["visibility"]), float(end["visibility"]))
        color = (211, 211, 34) if confidence >= SWIM_CONFIDENCE_HIGH else (0, 191, 255)
        cv2.line(
            frame,
            (int(start["px"]), int(start["py"])),
            (int(end["px"]), int(end["py"])),
            color,
            2,
            cv2.LINE_AA,
        )


def _renderable_landmark(landmark: Any) -> bool:
    return (
        isinstance(landmark, Mapping)
        and landmark.get("state") == "observed"
        and float(landmark.get("visibility", 0.0)) >= SWIM_CONFIDENCE_MEDIUM
        and "px" in landmark
        and "py" in landmark
    )
