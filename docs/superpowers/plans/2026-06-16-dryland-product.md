# Dryland Product Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the production Dryland web workflow for squat, lunge, and push-up analysis across the Next.js app, FastAPI API, CV service, persistence, overview history, and verification gates.

**Architecture:** Keep `ExerciseAnalyzer` as the dryland domain analyzer and make it profile-driven instead of auto-inferring a joint from arbitrary input keys. Add a FastAPI dryland pipeline that mirrors the running/cycling upload, SSE, annotated-video, save, and overview contract. Replace the placeholder Dryland landing page with a real dark-theme SPRINT AI workflow that requires explicit exercise selection before upload.

**Tech Stack:** Python 3, FastAPI, pytest, OpenCV, MediaPipe via shared `BaseAnalyzer` helpers, raw SQLite `AthleteDatabase`, Next.js 16, React 19, TypeScript, Vitest, Testing Library.

---

## File Structure

- Modify `video_analysis/constants.py` to store Dryland thresholds, supported exercise IDs, minimum quality gates, and interpolation limits.
- Modify `video_analysis/exercise_analyzer.py` to add explicit exercise profiles, a ready-effort-ready state machine, side selection, bounded interpolation, `tracked_joint`, and per-rep `effort_frame`.
- Create `tests/unit/test_exercise_analyzer.py` for synthetic angle-series coverage without OpenCV or MediaPipe.
- Create `backend/app/services/dryland.py` for video extraction, person lock-on, MediaPipe pose extraction, exercise-specific metric readiness, quality gates, annotation rendering, and result events.
- Create `tests/unit/test_dryland_service.py` for metric-ready gates, quality gates, active angle extraction, and service failure modes.
- Modify `backend/app/api/analysis.py` to expose `/api/analysis/dryland` upload, status, SSE events, annotated video, and save routes.
- Modify `backend/app/services/jobs.py` to document and accept `dryland` jobs with the existing registry.
- Create `tests/unit/test_dryland_api.py` for valid upload, unsupported exercise rejection, job kind isolation, video route, and idempotent save.
- Modify `video_analysis/athlete_database.py` so Dryland saves persist exercise type, rep count, tempo, stability, score, summary, and the raw result payload.
- Modify `video_analysis/sport_sessions.py` so saved Dryland sessions show meaningful overview metrics and evidence-based insights.
- Extend `tests/unit/test_athlete_database.py`, `tests/unit/test_sport_sessions.py`, and `tests/unit/test_sport_overview_api.py` for Dryland persistence/history.
- Modify `frontend/lib/analysis.ts` to add Dryland upload/save/SSE/video helpers and typed result payload fields.
- Extend `frontend/lib/analysis.test.ts` for Dryland API URLs, `exercise_type` form data, and SSE completion behavior.
- Create `frontend/components/sports/DrylandUploader.tsx` for exercise selection, capture guidance, upload state, progress, result summary, video link, per-rep table, athlete save, and rejection guidance.
- Modify `frontend/app/dryland/page.tsx` to replace the planned placeholder with the production landing page and overview data.
- Create `frontend/app/dryland/[jobId]/page.tsx` for direct result review, progress recovery, annotated video, save, and rejected-state rendering.
- Extend `frontend/app/dryland/page.test.tsx` and create `frontend/app/dryland/[jobId]/page.test.tsx` for user-visible workflow behavior.
- Modify `frontend/components/layout/TopNav.tsx` only if Dryland routing is missing from the current navigation.
- Modify `README.md` and `docs/UX.md` where they describe sport workflows and Dryland status.

## Task 1: Profile-Driven Exercise Analyzer

**Files:**
- Modify: `video_analysis/constants.py`
- Modify: `video_analysis/exercise_analyzer.py`
- Create: `tests/unit/test_exercise_analyzer.py`

- [x] **Step 1: Write failing analyzer tests**

```python
# tests/unit/test_exercise_analyzer.py
import math

import pytest

from video_analysis.exercise_analyzer import ExerciseAnalyzer


def _angle_frames(values, key="L.knee"):
    return [{key: value} if value is not None else {} for value in values]


def test_squat_counts_ready_effort_ready_reps():
    analyzer = ExerciseAnalyzer(fps=10)
    angles = [170, 166, 155, 130, 104, 92, 105, 135, 160, 170] * 2

    result = analyzer.analyze(_angle_frames(angles, "L.knee"), exercise_type="squat")

    assert result.exercise_type == "squat"
    assert result.tracked_joint == "knee"
    assert result.total_reps == 2
    assert result.avg_range_of_motion >= 70
    assert result.reps[0].start_frame == 0
    assert result.reps[0].effort_frame == 5
    assert result.reps[0].end_frame == 9
    assert result.reps[0].active_side == "left"


def test_push_up_uses_elbow_profile():
    analyzer = ExerciseAnalyzer(fps=10)
    elbow = [168, 160, 132, 96, 74, 68, 80, 110, 145, 166]

    result = analyzer.analyze(_angle_frames(elbow, "R.elbow"), exercise_type="push_up")

    assert result.tracked_joint == "elbow"
    assert result.total_reps == 1
    assert result.reps[0].active_side == "right"
    assert result.min_angle == pytest.approx(68, abs=1)


def test_lunge_uses_more_flexed_knee_and_side_label():
    analyzer = ExerciseAnalyzer(fps=10)
    frames = []
    left = [170, 165, 150, 124, 104, 94, 110, 138, 160, 170]
    right = [171, 168, 166, 160, 154, 150, 156, 164, 168, 170]
    for left_angle, right_angle in zip(left, right):
        frames.append({"L.knee": left_angle, "R.knee": right_angle})

    result = analyzer.analyze(frames, exercise_type="lunge")

    assert result.total_reps == 1
    assert result.reps[0].active_side == "left"
    assert result.avg_range_of_motion >= 70


def test_partial_cycles_are_not_counted():
    analyzer = ExerciseAnalyzer(fps=10)
    angles = [120, 100, 88, 94, 112, 132, 152, 166]

    result = analyzer.analyze(_angle_frames(angles, "L.knee"), exercise_type="squat")

    assert result.total_reps == 0
    assert result.avg_tempo == 0


def test_long_missing_gap_does_not_create_rep():
    analyzer = ExerciseAnalyzer(fps=10)
    angles = [170, 164, 150, None, None, None, None, 92, 108, 138, 166, 170]

    result = analyzer.analyze(_angle_frames(angles, "L.knee"), exercise_type="squat")

    assert result.total_reps == 0
    assert all(math.isfinite(value) for value in result.angle_history)


def test_state_does_not_leak_between_clips():
    analyzer = ExerciseAnalyzer(fps=10)
    full_rep = [170, 166, 150, 120, 95, 90, 110, 138, 160, 170]

    first = analyzer.analyze(_angle_frames(full_rep, "L.knee"), exercise_type="squat")
    second = analyzer.analyze(_angle_frames([170, 166, 160], "L.knee"), exercise_type="squat")

    assert first.total_reps == 1
    assert second.total_reps == 0
    assert analyzer.get_rep_at_frame(20) == 0
```

- [x] **Step 2: Run analyzer tests to verify failure**

Run: `pytest tests/unit/test_exercise_analyzer.py -v`

Expected: FAIL because `ExerciseStats.tracked_joint` and `RepData.effort_frame` do not exist and the old valley-peak-valley detector cannot count ready-effort-ready cycles.

- [x] **Step 3: Add Dryland constants**

```python
# video_analysis/constants.py
DRYLAND_SUPPORTED_EXERCISES = ("squat", "lunge", "push_up")
DRYLAND_MIN_VALID_FRAMES = 20
DRYLAND_MIN_POSE_COVERAGE_PCT = 35.0
DRYLAND_WARN_POSE_COVERAGE_PCT = 70.0
DRYLAND_MAX_INTERPOLATION_GAP_FRAMES = 3
DRYLAND_MIN_REP_DURATION_SEC = 0.45
DRYLAND_MAX_REP_DURATION_SEC = 8.0

DRYLAND_EXERCISE_PROFILES = {
    "squat": {
        "tracked_joint": "knee",
        "angle_keys": ("L.knee", "R.knee"),
        "ready_threshold": 160.0,
        "effort_threshold": 105.0,
        "min_rom": 35.0,
    },
    "lunge": {
        "tracked_joint": "knee",
        "angle_keys": ("L.knee", "R.knee"),
        "ready_threshold": 158.0,
        "effort_threshold": 110.0,
        "min_rom": 30.0,
    },
    "push_up": {
        "tracked_joint": "elbow",
        "angle_keys": ("L.elbow", "R.elbow"),
        "ready_threshold": 150.0,
        "effort_threshold": 90.0,
        "min_rom": 35.0,
    },
}
```

- [x] **Step 4: Implement analyzer dataclasses and profile lookup**

```python
# video_analysis/exercise_analyzer.py
from video_analysis.constants import (
    DRYLAND_EXERCISE_PROFILES,
    DRYLAND_MAX_INTERPOLATION_GAP_FRAMES,
    DRYLAND_MAX_REP_DURATION_SEC,
    DRYLAND_MIN_REP_DURATION_SEC,
)


@dataclass
class RepData:
    rep_number: int
    start_frame: int
    effort_frame: int
    end_frame: int
    duration_sec: float
    min_angle: float
    max_angle: float
    range_of_motion: float
    active_side: str = ""


@dataclass
class ExerciseStats:
    exercise_type: str
    tracked_joint: str
    total_reps: int
    avg_tempo: float
    avg_range_of_motion: float
    stability_score: float
    min_angle: float
    max_angle: float
    reps: list[RepData] = field(default_factory=list)
    angle_history: list[float] = field(default_factory=list)
```

- [x] **Step 5: Implement bounded extraction and cycle detection**

```python
# video_analysis/exercise_analyzer.py
def analyze(self, angles_list: list[dict], exercise_type: str = "squat", fps: float | None = None) -> ExerciseStats:
    self.fps = float(fps or self.fps)
    self.angle_history = []
    self.reps = []
    profile = self._profile(exercise_type)
    angles, sides, valid = self._extract_profile_angles(angles_list, profile)
    self.angle_history = angles
    if sum(valid) < 3:
        return self._empty_stats(exercise_type, profile["tracked_joint"], angles)
    smoothed = self._smooth(angles, window=3)
    self.reps = self._detect_reps(smoothed, sides, valid, profile)
    return self._calc_stats(exercise_type, profile["tracked_joint"], smoothed)


def _profile(self, exercise_type: str) -> dict:
    try:
        return DRYLAND_EXERCISE_PROFILES[exercise_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported dryland exercise: {exercise_type}") from exc


def _interpolate_short_gaps(self, values: list[float], valid: list[bool]) -> list[float]:
    output = values[:]
    index = 0
    while index < len(output):
        if valid[index]:
            index += 1
            continue
        start = index
        while index < len(output) and not valid[index]:
            index += 1
        end = index - 1
        gap = end - start + 1
        if start == 0 or index >= len(output) or gap > DRYLAND_MAX_INTERPOLATION_GAP_FRAMES:
            continue
        left = output[start - 1]
        right = output[index]
        step = (right - left) / (gap + 1)
        for offset in range(gap):
            output[start + offset] = left + step * (offset + 1)
            valid[start + offset] = True
    fallback = next((value for value, ok in zip(output, valid) if ok), 0.0)
    return [value if ok else fallback for value, ok in zip(output, valid)]
```

- [x] **Step 6: Run analyzer tests to verify pass**

Run: `pytest tests/unit/test_exercise_analyzer.py -v`

Expected: PASS.

- [x] **Step 7: Commit analyzer**

```bash
git add video_analysis/constants.py video_analysis/exercise_analyzer.py tests/unit/test_exercise_analyzer.py
git commit -m "feat: add profile-driven dryland analyzer"
```

## Task 2: Dryland Video Service

**Files:**
- Create: `backend/app/services/dryland.py`
- Create: `tests/unit/test_dryland_service.py`

- [x] **Step 1: Write failing service tests**

```python
# tests/unit/test_dryland_service.py
from backend.app.services.dryland import (
    dryland_quality,
    metric_ready,
    select_active_angles,
)


def test_metric_ready_squat_accepts_one_complete_side():
    keypoints = {
        "left_shoulder": (0, 0),
        "left_hip": (0, 1),
        "left_knee": (0, 2),
        "left_ankle": (0, 3),
    }

    assert metric_ready("squat", keypoints) is True


def test_metric_ready_lunge_requires_both_knees_and_ankles():
    incomplete = {
        "left_shoulder": (0, 0),
        "left_hip": (0, 1),
        "left_knee": (0, 2),
        "left_ankle": (0, 3),
    }
    complete = incomplete | {"right_knee": (1, 2), "right_ankle": (1, 3)}

    assert metric_ready("lunge", incomplete) is False
    assert metric_ready("lunge", complete) is True


def test_metric_ready_push_up_requires_arm_and_body_line():
    keypoints = {
        "right_shoulder": (0, 0),
        "right_elbow": (1, 0),
        "right_wrist": (2, 0),
        "right_hip": (0, 1),
        "right_ankle": (0, 2),
    }

    assert metric_ready("push_up", keypoints) is True


def test_dryland_quality_rejects_sparse_metric_ready_frames():
    result = dryland_quality(frames_total=100, frames_with_pose=60, metric_ready_frames=12)

    assert result["status"] == "fail"
    assert "Too few metric-ready frames" in result["warnings"][0]


def test_select_active_angles_outputs_expected_keys():
    frames = [
        {
            "left_hip": (0, 0),
            "left_knee": (0, 1),
            "left_ankle": (0, 2),
            "right_hip": (2, 0),
            "right_knee": (2, 1),
            "right_ankle": (2, 2),
        }
    ]

    angles = select_active_angles("squat", frames)

    assert set(angles[0]) == {"L.knee", "R.knee"}
```

- [x] **Step 2: Run service tests to verify failure**

Run: `pytest tests/unit/test_dryland_service.py -v`

Expected: FAIL because `backend.app.services.dryland` does not exist.

- [x] **Step 3: Implement service skeleton and quality gates**

```python
# backend/app/services/dryland.py
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
_POSE_PROCESSING_LOCK = get_pose_processing_lock()
_ANGLE_HELPER = BaseAnalyzer()


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
    pose_coverage = round(frames_with_pose / frames_total * 100, 1) if frames_total else 0.0
    warnings: list[str] = []
    if metric_ready_frames < DRYLAND_MIN_VALID_FRAMES:
        warnings.append("Too few metric-ready frames for a reliable dryland result.")
    if pose_coverage < DRYLAND_MIN_POSE_COVERAGE_PCT:
        warnings.append("Pose coverage is too low for dryland analysis.")
    elif pose_coverage < DRYLAND_WARN_POSE_COVERAGE_PCT:
        warnings.append("Pose coverage is usable but below the preferred 70%.")
    status = "fail" if metric_ready_frames < DRYLAND_MIN_VALID_FRAMES or pose_coverage < DRYLAND_MIN_POSE_COVERAGE_PCT else "pass"
    return {
        "status": status,
        "pose_coverage": pose_coverage,
        "metric_ready_frames": metric_ready_frames,
        "minimum_required_frames": DRYLAND_MIN_VALID_FRAMES,
        "warnings": warnings,
    }
```

- [x] **Step 4: Implement metric readiness and active angle helpers**

```python
# backend/app/services/dryland.py
def metric_ready(exercise_type: str, keypoints: dict[str, tuple[float, float]]) -> bool:
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
        left_push = all(name in keypoints for name in ("left_shoulder", "left_elbow", "left_wrist", "left_hip", "left_ankle"))
        right_push = all(name in keypoints for name in ("right_shoulder", "right_elbow", "right_wrist", "right_hip", "right_ankle"))
        return left_push or right_push
    return False


def select_active_angles(exercise_type: str, keypoints_list: list[dict[str, tuple[float, float]]]) -> list[dict[str, float]]:
    frames: list[dict[str, float]] = []
    for keypoints in keypoints_list:
        frame_angles: dict[str, float] = {}
        if exercise_type in {"squat", "lunge"}:
            for prefix, side in (("L", "left"), ("R", "right")):
                names = (f"{side}_hip", f"{side}_knee", f"{side}_ankle")
                if all(name in keypoints for name in names):
                    frame_angles[f"{prefix}.knee"] = _ANGLE_HELPER._calculate_angle(*(keypoints[name] for name in names))
        if exercise_type == "push_up":
            for prefix, side in (("L", "left"), ("R", "right")):
                names = (f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
                if all(name in keypoints for name in names):
                    frame_angles[f"{prefix}.elbow"] = _ANGLE_HELPER._calculate_angle(*(keypoints[name] for name in names))
        frames.append(frame_angles)
    return frames
```

- [x] **Step 5: Implement `analyze_dryland_video()` generator**

```python
# backend/app/services/dryland.py
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
MIN_LANDMARK_VISIBILITY = 0.4
BBOX_PADDING_PCT = 18
EMA_ALPHA = 0.65
YOLO_CONFIDENCE = 0.25


def _frame_path(frame_info: Any) -> str:
    return str(Path(frame_info["path"] if isinstance(frame_info, dict) else frame_info))


def _crop_bounds(bbox: list[float] | tuple[float, ...] | None, width: int, height: int) -> tuple[int, int, int, int]:
    if not bbox:
        return 0, 0, width, height
    x1, y1, x2, y2 = (int(value) for value in bbox[:4])
    pad_x = int(max(1, x2 - x1) * BBOX_PADDING_PCT / 100)
    pad_y = int(max(1, y2 - y1) * BBOX_PADDING_PCT / 100)
    return max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)


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
    cv2.rectangle(output, (24, 24), (380, 112), (15, 23, 42), -1)
    cv2.putText(output, f"Dryland reps: {rep_number}/{stats.total_reps}", (42, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(output, f"ROM: {stats.avg_range_of_motion:.1f} deg", (42, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (196, 181, 253), 2)
    return output


def analyze_dryland_video(video_path: Path, output_dir: Path, exercise_type: str, fps: float = 15.0) -> Iterator[Event]:
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
            yield ProgressEvent(28 + int(34 * index / max(1, total)), f"Pose frame {index + 1}/{total}")

    quality = dryland_quality(total, frames_with_pose, metric_ready_frames)
    if quality["status"] == "fail":
        guidance = "Record a fixed side view with the selected exercise, full body, hands or feet, and the active joint visible."
        yield ErrorEvent(f"{quality['warnings'][0]} {guidance}")
        return

    yield ProgressEvent(66, "Detecting full repetitions")
    analyzer = ExerciseAnalyzer(fps=fps)
    stats = analyzer.analyze(select_active_angles(exercise_type, keypoints_list), exercise_type=exercise_type, fps=fps)
    if stats.total_reps == 0:
        yield ErrorEvent("No complete repetition was confirmed. Start in the ready position, move through the effort phase, and return to ready.")
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

    yield ResultEvent(
        {
            "exercise_type": exercise_type,
            "analysis": asdict(stats),
            "quality": quality,
            "frames_total": total,
            "frames_with_pose": frames_with_pose,
            "video_path": "annotated.mp4",
        }
    )
```

- [x] **Step 7: Run service tests to verify pass**

Run: `pytest tests/unit/test_dryland_service.py -v`

Expected: PASS.

- [x] **Step 8: Commit service**

```bash
git add backend/app/services/dryland.py tests/unit/test_dryland_service.py
git commit -m "feat: add dryland analysis service"
```

## Task 3: Dryland API Routes

**Files:**
- Modify: `backend/app/api/analysis.py`
- Modify: `backend/app/services/jobs.py`
- Create: `tests/unit/test_dryland_api.py`

- [x] **Step 1: Write failing API tests**

```python
# tests/unit/test_dryland_api.py
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.jobs import registry


client = TestClient(app)


def test_dryland_upload_rejects_unsupported_exercise(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    response = client.post(
        "/api/analysis/dryland",
        files={"video": ("clip.mp4", video.read_bytes(), "video/mp4")},
        data={"exercise_type": "plank", "fps": "15"},
    )

    assert response.status_code == 400
    assert "Unsupported dryland exercise" in response.json()["detail"]


def test_dryland_job_routes_reject_wrong_kind():
    job = registry.create(kind="running")

    response = client.get(f"/api/analysis/dryland/{job.id}")

    assert response.status_code == 404


def test_dryland_save_requires_done_job():
    job = registry.create(kind="dryland")

    response = client.post(f"/api/analysis/dryland/{job.id}/save", json={"athlete_name": "Nikita"})

    assert response.status_code == 409
```

- [x] **Step 2: Run API tests to verify failure**

Run: `pytest tests/unit/test_dryland_api.py -v`

Expected: FAIL because `/api/analysis/dryland` routes are missing.

- [x] **Step 3: Add Dryland router helpers**

```python
# backend/app/api/analysis.py
from video_analysis.constants import DRYLAND_SUPPORTED_EXERCISES
from ..services import dryland as dryland_pipeline


def _run_dryland_pipeline_in_thread(job: Job, video_path: Path, exercise_type: str, fps: float) -> None:
    try:
        job.status = "running"
        for event in dryland_pipeline.analyze_dryland_video(video_path=video_path, output_dir=job.workspace, exercise_type=exercise_type, fps=fps):
            payload = event.to_dict()
            job.push_event(payload)
            if payload["type"] == "result":
                job.result = payload
                job.status = "done"
            elif payload["type"] == "error":
                job.error = payload["message"]
                job.status = "error"
        if job.status == "running":
            job.status = "done"
    except Exception as exc:
        logger.exception("Dryland job %s failed", job.id)
        job.error = str(exc)
        job.push_event({"type": "error", "message": str(exc)})
        job.status = "error"


def _get_dryland_job(job_id: str) -> Job:
    job = registry.get(job_id)
    if job is None or job.kind != "dryland":
        raise HTTPException(status_code=404, detail="Job not found")
    return job
```

- [x] **Step 4: Add Dryland endpoints**

```python
# backend/app/api/analysis.py
@router.post("/dryland")
async def upload_dryland(video: UploadFile = File(...), exercise_type: str = Form(...), fps: float = Form(15.0)) -> dict[str, str]:
    if exercise_type not in DRYLAND_SUPPORTED_EXERCISES:
        raise HTTPException(status_code=400, detail=f"Unsupported dryland exercise: {exercise_type}")
    suffix = validate_video_upload(filename=video.filename, content_type=video.content_type, declared_size=getattr(video, "size", None))
    job = registry.create(kind="dryland")
    video_path = job.workspace / f"input{suffix}"
    try:
        with video_path.open("wb") as handle:
            copy_upload_with_limit(video.file, handle)
    except UploadValidationError as exc:
        shutil.rmtree(job.workspace, ignore_errors=True)
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    Thread(target=_run_dryland_pipeline_in_thread, args=(job, video_path, exercise_type, fps), daemon=True).start()
    return {"job_id": job.id}


@router.get("/dryland/{job_id}")
async def get_dryland_job_status(job_id: str) -> dict:
    job = _get_dryland_job(job_id)
    return {"id": job.id, "status": job.status, "result": job.result, "error": job.error, "event_count": len(job.events), "saved_session_id": job.saved_session_id}


@router.get("/dryland/{job_id}/events")
async def stream_dryland_events(job_id: str):
    job = _get_dryland_job(job_id)
    return StreamingResponse(stream_job_events(job), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/dryland/{job_id}/video")
async def stream_dryland_video(job_id: str):
    job = _get_dryland_job(job_id)
    video_path = job.workspace / "annotated.mp4"
    if not video_path.exists():
        raise HTTPException(404, "Annotated video not ready")
    return FileResponse(video_path, media_type="video/mp4")


@router.post("/dryland/{job_id}/save")
async def save_dryland_job(job_id: str, request: SaveAnalysisRequest) -> dict[str, int]:
    job = _get_dryland_job(job_id)
    if job.status != "done" or not job.result:
        raise HTTPException(status_code=409, detail="Dryland analysis is not ready")
    session_id = await asyncio.to_thread(_save_completed_sport_job, job, request, "dryland")
    return {"session_id": session_id}
```

- [x] **Step 5: Run API tests**

Run: `pytest tests/unit/test_dryland_api.py -v`

Expected: PASS.

- [x] **Step 6: Commit API**

```bash
git add backend/app/api/analysis.py backend/app/services/jobs.py tests/unit/test_dryland_api.py
git commit -m "feat: expose dryland analysis api"
```

## Task 4: Dryland Persistence and Overview

**Files:**
- Modify: `video_analysis/athlete_database.py`
- Modify: `video_analysis/sport_sessions.py`
- Modify: `tests/unit/test_athlete_database.py`
- Modify: `tests/unit/test_sport_sessions.py`
- Modify: `tests/unit/test_sport_overview_api.py`

- [ ] **Step 1: Write persistence and overview tests**

```python
# tests/unit/test_sport_sessions.py
from video_analysis.athlete_database import TrainingSession
from video_analysis.sport_sessions import normalize_sport_session


def test_dryland_session_normalizes_reps_tempo_and_quality():
    session = TrainingSession(
        id=7,
        session_type="dryland",
        full_analysis='{"dryland_analysis":{"exercise_type":"squat","analysis":{"total_reps":4,"avg_tempo":2.1,"avg_range_of_motion":72,"stability_score":88},"quality":{"pose_coverage":81,"warnings":[]}}}',
    )

    result = normalize_sport_session(session)

    assert result["metrics"]["total_reps"]["value"] == 4
    assert result["metrics"]["avg_tempo"]["unit"] == "sec"
    assert result["score"] == 88
    assert result["quality"]["pose_coverage"] == 81
    assert "Squat" in result["summary"]
```

- [ ] **Step 2: Run focused tests to verify failure**

Run: `pytest tests/unit/test_sport_sessions.py::test_dryland_session_normalizes_reps_tempo_and_quality -v`

Expected: FAIL because Dryland normalization is not implemented.

- [ ] **Step 3: Implement Dryland normalization**

```python
# video_analysis/sport_sessions.py
def _dryland_session(session: TrainingSession, stored: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _base_session(session)
    result = stored.get("dryland_analysis", stored)
    if not isinstance(result, dict):
        return normalized
    analysis = result.get("analysis", result)
    if not isinstance(analysis, dict):
        return normalized
    exercise = str(result.get("exercise_type") or analysis.get("exercise_type") or session.exercise_type or "dryland")
    pairs = [
        _metric("total_reps", "Reps", analysis.get("total_reps"), "", higher_is_better=None),
        _metric("avg_tempo", "Average tempo", analysis.get("avg_tempo"), "sec", higher_is_better=None),
        _metric("avg_range_of_motion", "Average ROM", analysis.get("avg_range_of_motion"), "deg", higher_is_better=True),
        _metric("stability_score", "Movement consistency", analysis.get("stability_score"), "/100", higher_is_better=True),
    ]
    normalized["metrics"] = {key: value for pair in pairs if pair for key, value in [pair]}
    normalized["score"] = _number(analysis.get("stability_score"))
    quality = result.get("quality") if isinstance(result.get("quality"), dict) else {}
    normalized["quality"] = quality or _pose_quality(result)
    reps = _number(analysis.get("total_reps"))
    tempo = _number(analysis.get("avg_tempo"))
    summary_parts = [exercise.replace("_", "-").title()]
    if reps is not None:
        summary_parts.append(f"{reps:.0f} reps")
    if tempo is not None:
        summary_parts.append(f"{tempo:.1f}s tempo")
    normalized["summary"] = " · ".join(summary_parts)
    normalized["insights"] = [
        {
            "code": "dryland_evidence",
            "level": "success" if normalized["score"] is not None and normalized["score"] >= 80 else "info",
            "title": "Confirmed full repetitions",
            "detail": "Review the annotated video and per-rep evidence before changing exercise load.",
        }
    ]
    return normalized
```

- [ ] **Step 4: Route Dryland in normalizer**

```python
# video_analysis/sport_sessions.py
def normalize_sport_session(session: TrainingSession) -> Dict[str, Any]:
    stored = _stored_payload(session)
    if session.session_type == "running":
        return _running_session(session, stored)
    if session.session_type == "swimming":
        return _swimming_session(session, stored)
    if session.session_type == "cycling":
        return _cycling_session(session, stored)
    if session.session_type == "dryland":
        return _dryland_session(session, stored)
    return _base_session(session)
```

- [ ] **Step 5: Persist Dryland top-level session fields**

```python
# video_analysis/athlete_database.py
elif session_type == "dryland":
    result = analysis.get("dryland_analysis", analysis)
    exercise_stats = result.get("analysis", result) if isinstance(result, dict) else {}
    session.exercise_type = str(result.get("exercise_type") or exercise_stats.get("exercise_type") or "")
    session.reps = int(exercise_stats.get("total_reps") or 0)
    session.avg_tempo = float(exercise_stats.get("avg_tempo") or 0)
    session.stability_score = float(exercise_stats.get("stability_score") or 0)
    session.ai_score = float(exercise_stats.get("stability_score") or 0)
    session.ai_summary = f"{session.exercise_type.replace('_', '-').title()} · {session.reps} reps · {session.avg_tempo:.1f}s tempo"
```

- [ ] **Step 6: Run persistence tests**

Run: `pytest tests/unit/test_athlete_database.py tests/unit/test_sport_sessions.py tests/unit/test_sport_overview_api.py -v`

Expected: PASS.

- [ ] **Step 7: Commit persistence**

```bash
git add video_analysis/athlete_database.py video_analysis/sport_sessions.py tests/unit/test_athlete_database.py tests/unit/test_sport_sessions.py tests/unit/test_sport_overview_api.py
git commit -m "feat: persist dryland sport sessions"
```

## Task 5: Frontend API Client

**Files:**
- Modify: `frontend/lib/analysis.ts`
- Modify: `frontend/lib/analysis.test.ts`

- [ ] **Step 1: Write failing frontend API tests**

```ts
// frontend/lib/analysis.test.ts
import { uploadDrylandVideo, drylandAnnotatedVideoUrl } from "./analysis";

it("uploads dryland videos with the selected exercise", async () => {
  const fetchMock = vi.spyOn(global, "fetch").mockResolvedValueOnce(
    new Response(JSON.stringify({ job_id: "job-1" }), { status: 200 }),
  );
  const file = new File(["video"], "squat.mp4", { type: "video/mp4" });

  await expect(uploadDrylandVideo(file, "squat", 15)).resolves.toEqual({ jobId: "job-1" });

  const [, init] = fetchMock.mock.calls[0];
  const body = init?.body as FormData;
  expect(fetchMock.mock.calls[0][0]).toBe("http://localhost:8000/api/analysis/dryland");
  expect(body.get("exercise_type")).toBe("squat");
  expect(body.get("fps")).toBe("15");
});

it("builds dryland annotated video URLs", () => {
  expect(drylandAnnotatedVideoUrl("job-1")).toBe("http://localhost:8000/api/analysis/dryland/job-1/video");
});
```

- [ ] **Step 2: Run frontend API tests to verify failure**

Run: `cd frontend && npm test -- analysis.test.ts`

Expected: FAIL because Dryland helpers are missing.

- [ ] **Step 3: Implement Dryland helpers and types**

```ts
// frontend/lib/analysis.ts
export type DrylandExerciseType = "squat" | "lunge" | "push_up";

type AnalysisSport = "running" | "cycling" | "dryland";

export type DrylandResultEvent = ResultEvent & {
  exercise_type: DrylandExerciseType;
  analysis: {
    tracked_joint: "knee" | "elbow";
    total_reps: number;
    avg_tempo: number;
    avg_range_of_motion: number;
    stability_score: number;
    min_angle: number;
    max_angle: number;
    reps: Array<{
      rep_number: number;
      start_frame: number;
      effort_frame: number;
      end_frame: number;
      duration_sec: number;
      min_angle: number;
      max_angle: number;
      range_of_motion: number;
      active_side: string;
    }>;
  };
  quality: {
    status: "pass" | "fail";
    pose_coverage: number;
    metric_ready_frames: number;
    minimum_required_frames: number;
    warnings?: string[];
  };
};

export async function uploadDrylandVideo(file: File, exerciseType: DrylandExerciseType, fps = 15) {
  const fd = new FormData();
  fd.append("video", file);
  fd.append("exercise_type", exerciseType);
  fd.append("fps", String(fps));
  const res = await fetch(`${BACKEND_URL}/api/analysis/dryland`, { method: "POST", body: fd });
  if (!res.ok) {
    throw new Error(`Upload failed: ${res.status}`);
  }
  const data = (await res.json()) as { job_id: string };
  return { jobId: data.job_id };
}

export function subscribeDrylandAnalysis(jobId: string, onEvent: (event: AnalysisEvent) => void) {
  return subscribeSportAnalysis("dryland", jobId, onEvent);
}

export function saveDrylandAnalysis(jobId: string, input: { athleteId?: number; athleteName?: string }) {
  return saveSportAnalysis("dryland", jobId, input);
}

export function drylandAnnotatedVideoUrl(jobId: string): string {
  return sportAnnotatedVideoUrl("dryland", jobId);
}
```

- [ ] **Step 4: Run frontend API tests**

Run: `cd frontend && npm test -- analysis.test.ts`

Expected: PASS.

- [ ] **Step 5: Commit frontend client**

```bash
git add frontend/lib/analysis.ts frontend/lib/analysis.test.ts
git commit -m "feat: add dryland frontend api client"
```

## Task 6: Dryland Landing and Uploader

**Files:**
- Create: `frontend/components/sports/DrylandUploader.tsx`
- Modify: `frontend/app/dryland/page.tsx`
- Modify: `frontend/app/dryland/page.test.tsx`

- [ ] **Step 1: Write failing landing tests**

```tsx
// frontend/app/dryland/page.test.tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import DrylandPage from "./page";

it("renders exercise selection before upload", async () => {
  render(<DrylandPage />);

  expect(screen.getByRole("heading", { name: /dryland/i })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /squat/i })).toHaveAttribute("aria-pressed", "true");
  expect(screen.getByText(/fixed side view/i)).toBeInTheDocument();

  await userEvent.click(screen.getByRole("button", { name: /push-up/i }));

  expect(screen.getByText(/camera at torso height/i)).toBeInTheDocument();
});

it("does not show a planned workflow badge", () => {
  render(<DrylandPage />);

  expect(screen.queryByText(/web workflow planned/i)).not.toBeInTheDocument();
});
```

- [ ] **Step 2: Run landing tests to verify failure**

Run: `cd frontend && npm test -- dryland/page.test.tsx`

Expected: FAIL because the current page still says the web workflow is planned.

- [ ] **Step 3: Implement `DrylandUploader`**

```tsx
// frontend/components/sports/DrylandUploader.tsx
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

import { uploadDrylandVideo, type DrylandExerciseType } from "@/lib/analysis";
import { FileDropZone } from "@/components/ui/FileDropZone";

const EXERCISES: Record<DrylandExerciseType, { label: string; guidance: string }> = {
  squat: { label: "Squat", guidance: "Fixed side view. Keep shoulder, hip, knee, ankle and both feet in frame." },
  lunge: { label: "Lunge", guidance: "Fixed side view. Record one repeated side and keep both feet visible." },
  push_up: { label: "Push-up", guidance: "Camera at torso height. Keep hands, shoulders, hips and feet visible." },
};

export function DrylandUploader() {
  const router = useRouter();
  const [exercise, setExercise] = useState<DrylandExerciseType>("squat");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function handleFile(file: File) {
    setBusy(true);
    setError("");
    try {
      const { jobId } = await uploadDrylandVideo(file, exercise, 15);
      router.push(`/dryland/${jobId}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="space-y-5">
      <div className="grid gap-3 sm:grid-cols-3">
        {Object.entries(EXERCISES).map(([id, item]) => (
          <button
            key={id}
            type="button"
            aria-pressed={exercise === id}
            onClick={() => setExercise(id as DrylandExerciseType)}
            className="rounded-3xl border border-white/10 bg-white/[0.04] p-4 text-left text-white data-[active=true]:border-violet-300 data-[active=true]:bg-violet-400/15"
            data-active={exercise === id}
          >
            {item.label}
          </button>
        ))}
      </div>
      <p className="rounded-2xl border border-violet-300/20 bg-violet-500/10 p-4 text-sm text-violet-100">
        {EXERCISES[exercise].guidance}
      </p>
      <FileDropZone disabled={busy} onFile={handleFile} accept="video/*" label="Upload dryland video" />
      {error ? <p className="text-sm text-rose-300">{error}</p> : null}
    </section>
  );
}
```

- [ ] **Step 4: Wire landing page to real workflow**

```tsx
// frontend/app/dryland/page.tsx
import { DrylandUploader } from "@/components/sports/DrylandUploader";
import { getSportOverview } from "@/lib/sportOverview";

export default async function DrylandPage() {
  const overview = await getSportOverview("dryland");
  return (
    <SportLanding
      title="Dryland · Strength evidence"
      subtitle="Exercise-specific repetition evidence for squats, lunges and push-ups with annotated video, tempo, ROM and movement consistency."
      badges={[
        { icon: Dumbbell, label: "Squat · Lunge · Push-up" },
        { variant: "success", label: "Quality-gated analysis" },
        { variant: "info", label: "Fixed side view" },
      ]}
      hint="Choose the exercise first, then upload a fixed side-view clip. SPRINT only scores confirmed full repetitions."
      accentRgb="139,92,246"
      metrics={[
        { label: "Input", value: "Side view", icon: Camera, accent: true, hint: "Full body visible" },
        { label: "Evidence", value: "Full reps", icon: Dumbbell, hint: "Ready → effort → ready" },
        { label: "Measurement", value: "Tempo + ROM", icon: Activity, hint: "Per-rep evidence table" },
        { label: "History", value: `${overview.sessions.length}`, icon: Target, hint: "Saved dryland sessions" },
      ]}
      sessions={overview.sessions}
      insights={overview.insights}
      uploadAvailable
      uploader={<DrylandUploader />}
    />
  );
}
```

- [ ] **Step 5: Run landing tests**

Run: `cd frontend && npm test -- dryland/page.test.tsx`

Expected: PASS.

- [ ] **Step 6: Commit landing**

```bash
git add frontend/components/sports/DrylandUploader.tsx frontend/app/dryland/page.tsx frontend/app/dryland/page.test.tsx
git commit -m "feat: connect dryland upload workspace"
```

## Task 7: Dryland Result Page

**Files:**
- Create: `frontend/app/dryland/[jobId]/page.tsx`
- Create: `frontend/app/dryland/[jobId]/page.test.tsx`

- [ ] **Step 1: Write failing result-page tests**

```tsx
// frontend/app/dryland/[jobId]/page.test.tsx
import { render, screen, waitFor } from "@testing-library/react";

import DrylandResultPage from "./page";

it("renders progress and final dryland result", async () => {
  render(<DrylandResultPage params={Promise.resolve({ jobId: "job-1" })} />);

  expect(screen.getByText(/preparing dryland analysis/i)).toBeInTheDocument();

  await waitFor(() => {
    expect(screen.getByText(/annotated evidence/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run result-page tests to verify failure**

Run: `cd frontend && npm test -- 'dryland/[jobId]/page.test.tsx'`

Expected: FAIL because the route does not exist.

- [ ] **Step 3: Implement result page**

```tsx
// frontend/app/dryland/[jobId]/page.tsx
"use client";

import { useEffect, useState } from "react";

import {
  drylandAnnotatedVideoUrl,
  saveDrylandAnalysis,
  subscribeDrylandAnalysis,
  type AnalysisEvent,
  type DrylandResultEvent,
} from "@/lib/analysis";

export default function DrylandResultPage({ params }: { params: Promise<{ jobId: string }> }) {
  const [jobId, setJobId] = useState("");
  const [event, setEvent] = useState<AnalysisEvent | null>(null);
  const [athleteName, setAthleteName] = useState("Athlete");
  const [savedId, setSavedId] = useState<number | null>(null);

  useEffect(() => {
    params.then(({ jobId: id }) => setJobId(id));
  }, [params]);

  useEffect(() => {
    if (!jobId) return;
    return subscribeDrylandAnalysis(jobId, setEvent);
  }, [jobId]);

  if (!jobId || !event || event.type === "progress") {
    const pct = event?.type === "progress" ? event.pct : 0;
    const label = event?.type === "progress" ? event.label : "Preparing dryland analysis";
    return <main className="min-h-screen bg-slate-950 p-8 text-white">{label} · {pct}%</main>;
  }

  if (event.type === "error") {
    return <main className="min-h-screen bg-slate-950 p-8 text-white"><h1>Dryland analysis needs a reshoot</h1><p>{event.message}</p></main>;
  }

  const result = event as DrylandResultEvent;
  return (
    <main className="min-h-screen bg-slate-950 p-8 text-white">
      <h1>Annotated evidence</h1>
      <video src={drylandAnnotatedVideoUrl(jobId)} controls className="mt-4 w-full rounded-3xl" />
      <dl>
        <dt>Reps</dt><dd>{result.analysis.total_reps}</dd>
        <dt>Tempo</dt><dd>{result.analysis.avg_tempo}s</dd>
        <dt>ROM</dt><dd>{result.analysis.avg_range_of_motion}°</dd>
      </dl>
      <input value={athleteName} onChange={(e) => setAthleteName(e.target.value)} aria-label="Athlete name" />
      <button type="button" onClick={async () => setSavedId((await saveDrylandAnalysis(jobId, { athleteName })).sessionId)}>
        Save to athlete history
      </button>
      {savedId ? <p>Saved session #{savedId}</p> : null}
    </main>
  );
}
```

- [ ] **Step 4: Run result-page tests**

Run: `cd frontend && npm test -- 'dryland/[jobId]/page.test.tsx'`

Expected: PASS.

- [ ] **Step 5: Commit result page**

```bash
git add frontend/app/dryland/[jobId]/page.tsx frontend/app/dryland/[jobId]/page.test.tsx
git commit -m "feat: add dryland result page"
```

## Task 8: Documentation, Full Gates, Browser QA, and Merge Prep

**Files:**
- Modify: `README.md`
- Modify: `docs/UX.md`
- Inspect: `frontend/components/layout/TopNav.tsx`

- [ ] **Step 1: Update product docs**

```markdown
Dryland analysis now supports explicit squat, lunge, and push-up upload workflows in the FastAPI + Next.js product. The workflow is quality-gated, renders annotated evidence video, persists athlete sessions, and shows saved metrics in sport overview history.
```

- [ ] **Step 2: Run backend focused tests**

Run: `pytest tests/unit/test_exercise_analyzer.py tests/unit/test_dryland_service.py tests/unit/test_dryland_api.py tests/unit/test_athlete_database.py tests/unit/test_sport_sessions.py tests/unit/test_sport_overview_api.py -v`

Expected: PASS.

- [ ] **Step 3: Run frontend focused tests**

Run: `cd frontend && npm test -- analysis.test.ts dryland/page.test.tsx 'dryland/[jobId]/page.test.tsx'`

Expected: PASS.

- [ ] **Step 4: Run lint and type gates**

Run: `PATH="$PWD/.venv/bin:$PATH" make lint`

Expected: PASS.

Run: `cd frontend && npm run typecheck`

Expected: PASS.

- [ ] **Step 5: Run production build**

Run: `cd frontend && npm run build`

Expected: PASS.

- [ ] **Step 6: Run browser QA on the local product surface**

Open: `http://127.0.0.1:3000/dryland`

Verify:
- the page uses the existing dark SPRINT AI visual system;
- no “planned workflow” label is visible;
- squat is selected by default;
- selecting lunge changes guidance to “both feet visible”;
- selecting push-up changes guidance to “camera at torso height”;
- upload is disabled only while upload is in progress;
- saved Dryland sessions appear in the overview after a successful save.

- [ ] **Step 7: Commit docs and verification fixes**

```bash
git add README.md docs/UX.md frontend/components/layout/TopNav.tsx
git commit -m "docs: mark dryland workflow production ready"
```

- [ ] **Step 8: Prepare PR or main merge**

Run: `git status --short`

Expected: no uncommitted changes in the worktree.

Run: `git log --oneline origin/main..HEAD`

Expected: dryland design, analyzer, service, API, persistence, frontend, result page, and docs commits are listed.

## Self-Review

- Spec coverage: the plan covers explicit exercise selection, capture guidance, upload validation, person lock-on, cached MediaPipe pose inference, metric-ready frame gates, full-cycle state machine, rep count, tempo, ROM, consistency, per-rep evidence, annotated MP4, quality evidence, athlete save/history, honest rejected states, desktop/mobile dark UI, and documentation.
- Deferred scope remains out of implementation: auto classification, plank/isometric holds, loaded force/power estimation, group analysis, live camera, exercise programs, cloud/auth.
- Placeholder scan: this plan contains concrete paths, commands, expected failures, expected passes, and implementation snippets for every code-producing task.
- Type consistency: exercise IDs are `squat`, `lunge`, `push_up`; result payload uses `exercise_type`, `analysis`, `quality`, `frames_total`, `frames_with_pose`, and `video_path`; per-rep fields use `rep_number`, `start_frame`, `effort_frame`, `end_frame`, `duration_sec`, `min_angle`, `max_angle`, `range_of_motion`, and `active_side`.
