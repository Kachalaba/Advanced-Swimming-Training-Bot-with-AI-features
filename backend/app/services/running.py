"""Running video analysis pipeline.

Pure-Python port of the Streamlit running pipeline (pages/running.py).
Yields progress events so the API layer can stream them to the client
via SSE without coupling the analyzer to a specific transport.

Preserves the lock-on tracking and guard-relaxation fixes that were
proven on real footage:
  - YOLO confidence 0.25 (catch distant runners)
  - MIN_LANDMARK_VISIBILITY 0.3
  - Per-frame target re-acquisition when track id churns
  - Displacement guard floored at 80px
  - Pose detection / tracking confidence 0.2 / 0.4
  - Face landmarks excluded from spatial containment guard
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Union

import cv2
import numpy as np

# The video_analysis package lives at the repo root. Backend runs from
# /srv inside docker; mount the repo root and add it to the path so we
# can reuse the existing analyzers without copying them.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_analysis.frame_extractor import extract_frames_from_video  # noqa: E402
from video_analysis.running_analyzer import RunningAnalysis  # noqa: E402
from video_analysis.running_analyzer import RunningAnalyzer
from video_analysis.swimmer_detector import detect_swimmer_in_frames  # noqa: E402

from .video_encoding import finalize_browser_video, open_intermediate_writer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────
MIN_LANDMARK_VISIBILITY = 0.3
MIN_CROP_SIZE = 256
EMA_ALPHA = 0.72
BBOX_PADDING_PCT = 20
MAX_LOST_FRAMES = 10
YOLO_CONF = 0.25

FULL_LANDMARK_MAP = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_toe",
    32: "right_toe",
}

FACE_LANDMARKS = {
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
}

# Skeleton drawing
_C_LEFT = (255, 120, 60)
_C_RIGHT = (60, 180, 255)
_C_TORSO = (200, 200, 200)
_LEFT_CONN = [
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_toe"),
]
_RIGHT_CONN = [
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_toe"),
]
_TORSO_CONN = [
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
]


# ─────────────────────────────────────────────────────────────────────
# Events
# ─────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────
# Helpers (ported from pages/running.py — no Streamlit deps)
# ─────────────────────────────────────────────────────────────────────
def _calc_iou(b1, b2) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def _centroid_dist(b1, b2) -> float:
    c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
    c2 = ((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


def _bbox_area(b) -> float:
    return (b[2] - b[0]) * (b[3] - b[1])


def _match_tracks(current_bboxes, prev_bboxes, frame_w, frame_h, iou_thr=0.2):
    matches: dict[int, int] = {}
    used: set[int] = set()
    sorted_curr = sorted(
        enumerate(current_bboxes),
        key=lambda x: (x[1][0] + x[1][2]) / 2,
    )
    for ci, cb in sorted_curr:
        best_iou, best_id = 0.0, None
        for tid, pb in prev_bboxes.items():
            if tid in used:
                continue
            iou = _calc_iou(cb, pb)
            if iou > best_iou and iou >= iou_thr:
                best_iou, best_id = iou, tid
        if best_id is None:
            best_dist = float("inf")
            max_dist = max(frame_w, frame_h) * 0.3
            for tid, pb in prev_bboxes.items():
                if tid in used:
                    continue
                d = _centroid_dist(cb, pb)
                if d < best_dist and d < max_dist:
                    best_dist, best_id = d, tid
        if best_id is not None:
            matches[ci] = best_id
            used.add(best_id)
    return matches


def _dim(c, alpha):
    return tuple(int(v * alpha) for v in c)


def _draw_skeleton(frame, kps):
    if not kps:
        return
    alpha = 0.85
    thickness = 2
    radius = 2
    groups = [
        (_LEFT_CONN, _dim(_C_LEFT, alpha)),
        (_RIGHT_CONN, _dim(_C_RIGHT, alpha)),
        (_TORSO_CONN, _dim(_C_TORSO, alpha)),
    ]
    for connections, color in groups:
        for a, b in connections:
            if a in kps and b in kps:
                pa = (int(kps[a][0]), int(kps[a][1]))
                pb = (int(kps[b][0]), int(kps[b][1]))
                cv2.line(frame, pa, pb, color, thickness, cv2.LINE_AA)
    for name, (x, y) in kps.items():
        pt = (int(x), int(y))
        if "left" in name:
            jc = _dim(_C_LEFT, alpha)
        elif "right" in name:
            jc = _dim(_C_RIGHT, alpha)
        else:
            jc = _dim(_C_TORSO, alpha)
        cv2.circle(frame, pt, radius, jc, -1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────
def analyze_running_video(
    video_path: Path,
    output_dir: Path,
    fps: float = 30.0,
) -> Iterator[Event]:
    """Run full pipeline and yield progress + final result events."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    detections_dir = output_dir / "detections"

    yield ProgressEvent(2, "Extracting frames")
    frame_result = extract_frames_from_video(
        str(video_path),
        output_dir=str(frames_dir),
        fps=fps,
    )
    frames = frame_result["frames"]
    if not frames:
        yield ErrorEvent("No frames extracted from video")
        return

    yield ProgressEvent(15, f"Extracted {len(frames)} frames")

    yield ProgressEvent(18, "Detecting runner with YOLO")
    detection_result = detect_swimmer_in_frames(
        frames,
        output_dir=str(detections_dir),
        draw_boxes=False,
        enable_tracking=True,
        confidence_threshold=YOLO_CONF,
    )
    yield ProgressEvent(30, "Detection complete")

    # Read first frame to learn video dims
    first_path = str(Path(frames[0]["path"] if isinstance(frames[0], dict) else frames[0]))
    first_frame = cv2.imread(first_path)
    if first_frame is None:
        yield ErrorEvent("Could not read first frame")
        return
    h, w = first_frame.shape[:2]

    # Initialize MediaPipe pose with relaxed confidences
    import mediapipe as _mp

    pose_hq = _mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.2,
        min_tracking_confidence=0.4,
    )

    ema_state: dict[str, tuple[float, float]] = {}
    prev_keypoints: dict[int, dict[str, tuple[float, float]]] = {}
    lost_frame_count: dict[int, int] = {}
    prev_bboxes: dict[int, tuple] = {}
    track_id_counter = 0
    target_track_id: int | None = None
    target_bbox: tuple | None = None
    keypoints_list: list[dict[str, tuple[float, float]]] = []
    annotated_frames: list[Any] = []
    frames_with_pose = 0

    def ema(name: str, x: float, y: float) -> tuple[float, float]:
        if name not in ema_state:
            ema_state[name] = (x, y)
        else:
            px, py = ema_state[name]
            ema_state[name] = (
                EMA_ALPHA * x + (1 - EMA_ALPHA) * px,
                EMA_ALPHA * y + (1 - EMA_ALPHA) * py,
            )
        return ema_state[name]

    yield ProgressEvent(32, "Pose estimation with lock-on tracking")
    n = len(frames)
    for i, frame_info in enumerate(frames):
        frame_path = str(Path(frame_info["path"] if isinstance(frame_info, dict) else frame_info))
        frame = cv2.imread(frame_path)
        if frame is None:
            keypoints_list.append({})
            annotated_frames.append(None)
            continue

        annotated = frame.copy()
        current_bboxes: list = []

        # Step A: pull YOLO detections for this frame
        if i < len(detection_result["detections"]):
            det = detection_result["detections"][i]
            bboxes = det.get("all_boxes") or ([det.get("bbox")] if det.get("bbox") else [])
            current_bboxes = sorted(
                [b for b in bboxes if b],
                key=lambda b: (b[0] + b[2]) / 2,
            )

        if current_bboxes:
            if prev_bboxes:
                matches = _match_tracks(current_bboxes, prev_bboxes, w, h)
            else:
                matches = {idx: idx for idx in range(len(current_bboxes))}
                track_id_counter = len(current_bboxes)

            if i == 0 and target_track_id is None:
                # Pick largest bbox as the target on the first frame
                target_track_id = max(
                    range(len(current_bboxes)),
                    key=lambda k: _bbox_area(current_bboxes[k]),
                )

            new_prev: dict[int, tuple] = {}
            target_matched = False
            for ci, bb in enumerate(current_bboxes):
                tid = matches.get(ci, track_id_counter)
                if tid not in matches.values():
                    track_id_counter += 1
                new_prev[tid] = bb
                if tid == target_track_id:
                    target_bbox = bb
                    target_matched = True

            # Lock-on re-acquisition
            if not target_matched and target_bbox is not None and current_bboxes:
                prev_cx = (target_bbox[0] + target_bbox[2]) / 2
                prev_cy = (target_bbox[1] + target_bbox[3]) / 2
                best_idx = min(
                    range(len(current_bboxes)),
                    key=lambda k: (
                        ((current_bboxes[k][0] + current_bboxes[k][2]) / 2 - prev_cx) ** 2
                        + ((current_bboxes[k][1] + current_bboxes[k][3]) / 2 - prev_cy) ** 2
                    ),
                )
                for tid, bb in new_prev.items():
                    if bb is current_bboxes[best_idx]:
                        target_track_id = tid
                        break
                target_bbox = current_bboxes[best_idx]

            prev_bboxes = new_prev

        kps: dict[str, tuple[float, float]] = {}
        if target_bbox:
            tx1, ty1, tx2, ty2 = (int(c) for c in target_bbox[:4])
            bw = max(1, tx2 - tx1)
            bh = max(1, ty2 - ty1)
            pad_x = int(bw * BBOX_PADDING_PCT / 100)
            pad_y = int(bh * BBOX_PADDING_PCT / 100)
            cx1 = max(0, tx1 - pad_x)
            cy1 = max(0, ty1 - pad_y)
            cx2 = min(w, tx2 + pad_x)
            cy2 = min(h, ty2 + pad_y)
            crop = frame[cy1:cy2, cx1:cx2]
            crop_h, crop_w = crop.shape[:2]
            if crop_w >= 4 and crop_h >= 4:
                scale = 1.0
                min_dim = min(crop_w, crop_h)
                if min_dim < MIN_CROP_SIZE:
                    scale = MIN_CROP_SIZE / min_dim
                    crop = cv2.resize(
                        crop,
                        (int(crop_w * scale), int(crop_h * scale)),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                    crop_h, crop_w = crop.shape[:2]

                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                mp_result = pose_hq.process(rgb)

                if mp_result.pose_landmarks:
                    raw: dict[str, tuple[float, float]] = {}
                    for idx, name in FULL_LANDMARK_MAP.items():
                        lm = mp_result.pose_landmarks.landmark[idx]
                        if lm.visibility >= MIN_LANDMARK_VISIBILITY:
                            ax = cx1 + (lm.x * crop_w) / scale
                            ay = cy1 + (lm.y * crop_h) / scale
                            raw[name] = (ax, ay)

                    if raw:
                        pose_ok = True

                        # Spatial guard (skip face)
                        tol_x = (cx2 - cx1) * 0.3
                        tol_y = (cy2 - cy1) * 0.3
                        for name, (ax, ay) in raw.items():
                            if name in FACE_LANDMARKS:
                                continue
                            if not (cx1 - tol_x <= ax <= cx2 + tol_x and cy1 - tol_y <= ay <= cy2 + tol_y):
                                pose_ok = False
                                break

                        # Size guard
                        if pose_ok:
                            pts = list(raw.values())
                            sx = max(p[0] for p in pts) - min(p[0] for p in pts)
                            sy = max(p[1] for p in pts) - min(p[1] for p in pts)
                            if sy > (cy2 - cy1) * 1.5 or sx > (cx2 - cx1) * 2.0:
                                pose_ok = False

                        # Displacement guard (floored at 80px)
                        tid = target_track_id if target_track_id is not None else 0
                        if pose_ok and tid in prev_keypoints:
                            person_diag = math.hypot(bw, bh)
                            max_jump = max(person_diag, 80.0)
                            for anchor in ("left_hip", "right_hip"):
                                if anchor in raw and anchor in prev_keypoints[tid]:
                                    dx = raw[anchor][0] - prev_keypoints[tid][anchor][0]
                                    dy = raw[anchor][1] - prev_keypoints[tid][anchor][1]
                                    if math.hypot(dx, dy) > max_jump:
                                        pose_ok = False
                                        break

                        if pose_ok:
                            for name, (x, y) in raw.items():
                                kps[name] = ema(name, x, y)
                            frames_with_pose += 1

        # Step E: short fallback if detection lost
        tid = target_track_id if target_track_id is not None else 0
        if kps:
            prev_keypoints[tid] = kps
            lost_frame_count[tid] = 0
        elif tid in prev_keypoints:
            lc = lost_frame_count.get(tid, 0) + 1
            lost_frame_count[tid] = lc
            if lc <= MAX_LOST_FRAMES:
                kps = prev_keypoints[tid]

        if kps:
            _draw_skeleton(annotated, kps)

        keypoints_list.append(kps if kps else {})
        annotated_frames.append(annotated)

        if i % 20 == 0:
            pct = 32 + int(33 * (i / max(1, n)))
            yield ProgressEvent(pct, f"Pose: frame {i + 1}/{n}")

    pose_hq.close()

    yield ProgressEvent(68, "Computing running metrics")
    analyzer = RunningAnalyzer(fps=fps)
    analysis: RunningAnalysis = analyzer.analyze(keypoints_list, fps=fps)

    yield ProgressEvent(75, "Encoding annotated video")
    annotated_path = output_dir / "annotated.mp4"
    try:
        writer, intermediate_path = open_intermediate_writer(
            annotated_path,
            fps=fps,
            frame_size=(w, h),
        )
    except RuntimeError as exc:
        yield ErrorEvent(str(exc))
        return

    cadence_label = f"Cadence: {analysis.cadence:.0f} spm"
    strike_label = f"Foot strike: {analysis.foot_strike_type}"
    for af in annotated_frames:
        if af is None:
            continue
        cv2.putText(
            af,
            cadence_label,
            (10, h - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            af,
            strike_label,
            (10, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        writer.write(af)
    writer.release()
    try:
        finalize_browser_video(intermediate_path, annotated_path)
    except RuntimeError as exc:
        yield ErrorEvent(str(exc))
        return

    yield ProgressEvent(95, "Finalizing")
    payload = {
        "analysis": _serialize_analysis(analysis),
        "frames_total": len(frames),
        "frames_with_pose": frames_with_pose,
        "video_path": str(annotated_path.relative_to(output_dir)),
    }
    yield ProgressEvent(100, "Done")
    yield ResultEvent(payload)


def _serialize_analysis(a: RunningAnalysis) -> dict[str, Any]:
    """Convert analyzer result to plain JSON-friendly dict."""
    try:
        d = asdict(a)
    except TypeError:
        # Fallback for non-dataclass results
        d = {k: getattr(a, k) for k in dir(a) if not k.startswith("_") and not callable(getattr(a, k))}
    # Coerce non-serializable types
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [_safe(x) for x in v]
        elif isinstance(v, dict):
            out[k] = {kk: _safe(vv) for kk, vv in v.items()}
        else:
            out[k] = _safe(v)
    out["vertical_oscillation_px"] = out.get("avg_vertical_osc", 0)
    return out


def _safe(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]
    if isinstance(v, dict):
        return {k: _safe(vv) for k, vv in v.items()}
    if hasattr(v, "value"):  # Enum
        return v.value
    if hasattr(v, "__dict__"):
        return {k: _safe(vv) for k, vv in vars(v).items()}
    return str(v)
