"""Stateful live and uploaded-video rehabilitation processing."""

from __future__ import annotations

import threading
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Optional

import cv2
import numpy as np

from backend.app.services.camera_level import CameraLevelEstimator
from backend.app.services.posture import calculate_posture
from video_analysis.biomechanics_visualizer import BiomechanicsVisualizer
from video_analysis.frame_extractor import extract_frames_from_video
from video_analysis.rehab_analyzer import RehabAnalyzer

MAX_LIVE_WINDOW_FRAMES = 180
DEFAULT_ANALYSIS_INTERVAL = 3
_POSE_PROCESSING_LOCK = threading.Lock()


def _normalize_keypoints(
    keypoints: Dict[str, Dict[str, float]],
    width: int,
    height: int,
) -> Dict[str, Dict[str, float]]:
    if width <= 0 or height <= 0:
        return {}
    return {
        name: {
            "x": round(float(point["x"]) / float(width), 4),
            "y": round(float(point["y"]) / float(height), 4),
        }
        for name, point in keypoints.items()
        if "x" in point and "y" in point
    }


def _keypoint_bbox(
    keypoints: Dict[str, Dict[str, float]],
    width: int,
    height: int,
) -> Optional[tuple]:
    if not keypoints:
        return None
    xs = [float(point["x"]) for point in keypoints.values() if "x" in point and "y" in point]
    ys = [float(point["y"]) for point in keypoints.values() if "x" in point and "y" in point]
    if not xs or not ys:
        return None
    pad_x = width * 0.08
    pad_y = height * 0.08
    return (
        max(0, int(min(xs) - pad_x)),
        max(0, int(min(ys) - pad_y)),
        min(width, int(max(xs) + pad_x)),
        min(height, int(max(ys) + pad_y)),
    )


class LiveRehabSession:
    """Own mutable analysis state for one browser camera session."""

    def __init__(
        self,
        protocol: str,
        fps: float,
        pose_processor: Optional[Any] = None,
        analyzer: Optional[Any] = None,
        camera_level: Optional[Any] = None,
        max_frames: int = MAX_LIVE_WINDOW_FRAMES,
        analysis_interval: int = DEFAULT_ANALYSIS_INTERVAL,
    ) -> None:
        self.id = uuid.uuid4().hex[:16]
        self.protocol = protocol
        self.fps = fps if fps > 0 else 5.0
        self.pose_processor = pose_processor or BiomechanicsVisualizer(trajectory_length=20)
        self.analyzer = analyzer or RehabAnalyzer(fps=self.fps)
        self.camera_level = camera_level or CameraLevelEstimator()
        self.keypoint_frames: Deque[Dict[str, Dict[str, float]]] = deque(maxlen=max_frames)
        self.analysis_interval = max(1, analysis_interval)
        self.frame_count = 0
        self.latest_update: Optional[Dict[str, Any]] = None
        self._report: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()

    @property
    def keypoint_count(self) -> int:
        return len(self.keypoint_frames)

    def process_frame(self, frame: np.ndarray, calibrate: bool = False) -> Dict[str, Any]:
        """Analyze one frame and return the latest live UI payload."""
        with self._lock:
            self.frame_count += 1
            with _POSE_PROCESSING_LOCK:
                _, pose_data = self.pose_processor.process_frame(frame, self.frame_count)
            height, width = frame.shape[:2]
            raw_keypoints = pose_data.get("keypoints", {}) if pose_data.get("has_pose") else {}
            normalized = _normalize_keypoints(raw_keypoints, width, height)
            if raw_keypoints:
                self.keypoint_frames.append(raw_keypoints)
            if self.keypoint_frames and (self._report is None or self.frame_count % self.analysis_interval == 0):
                self._report = self.analyzer.analyze(
                    list(self.keypoint_frames),
                    protocol=self.protocol,
                )

            posture = calculate_posture(normalized)
            athlete_bbox = _keypoint_bbox(raw_keypoints, width, height)
            level = (
                self.camera_level.calibrate(frame, athlete_bbox=athlete_bbox)
                if calibrate
                else self.camera_level.measure(frame, athlete_bbox=athlete_bbox)
            )
            self.latest_update = {
                "session_id": self.id,
                "sequence": self.frame_count,
                "pose_detected": bool(raw_keypoints),
                "landmarks": normalized,
                "posture": posture,
                "camera_level": level,
                "report": self._report,
            }
            return self.latest_update

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            if self.latest_update is not None:
                return dict(self.latest_update)
            return {
                "session_id": self.id,
                "sequence": self.frame_count,
                "pose_detected": False,
                "landmarks": {},
                "posture": {"available": False},
                "camera_level": {
                    "angle_deg": None,
                    "confidence": 0.0,
                    "status": "uncalibrated",
                    "relative": True,
                },
                "report": None,
            }


class LiveRehabRegistry:
    """Thread-safe in-memory registry for local live rehabilitation sessions."""

    def __init__(
        self,
        session_factory: Optional[Callable[[str, float], LiveRehabSession]] = None,
    ) -> None:
        self._session_factory = session_factory or (lambda protocol, fps: LiveRehabSession(protocol=protocol, fps=fps))
        self._sessions: Dict[str, LiveRehabSession] = {}
        self._lock = threading.Lock()

    def create(self, protocol: str, fps: float) -> LiveRehabSession:
        session = self._session_factory(protocol, fps)
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Optional[LiveRehabSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None


def analyze_rehabilitation_video(
    video_path: Path,
    output_dir: Path,
    protocol: str,
    fps: float,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    """Run the existing rehabilitation pipeline for an uploaded video."""

    def progress(pct: int, label: str) -> None:
        if on_progress:
            on_progress(pct, label)

    output_dir.mkdir(parents=True, exist_ok=True)
    progress(5, "Extracting frames")
    frame_result = extract_frames_from_video(
        str(video_path),
        output_dir=str(output_dir / "frames"),
        fps=fps,
    )
    frames = frame_result.get("frames", [])
    if not frames:
        raise ValueError("No frames extracted from video")

    visualizer = BiomechanicsVisualizer(trajectory_length=30)
    analyzer = RehabAnalyzer(fps=fps)
    keypoint_frames = []
    annotated_frames = []
    frame_size = None

    for index, frame_info in enumerate(frames):
        frame_path = Path(frame_info["path"] if isinstance(frame_info, dict) else frame_info)
        frame = cv2.imread(str(frame_path))
        if frame is None:
            keypoint_frames.append({})
            continue
        if frame_size is None:
            frame_size = (frame.shape[1], frame.shape[0])
        with _POSE_PROCESSING_LOCK:
            annotated, pose_data = visualizer.process_frame(frame, index)
        annotated_frames.append(annotated)
        keypoint_frames.append(pose_data.get("keypoints", {}) if pose_data.get("has_pose") else {})
        if index % 10 == 0:
            progress(15 + int(55 * index / max(1, len(frames))), "Detecting posture")

    progress(72, "Calculating ROM and symmetry")
    report = analyzer.analyze(keypoint_frames, protocol=protocol)
    annotated_path = output_dir / "annotated.mp4"
    if frame_size and annotated_frames:
        writer = None
        for codec in ("avc1", "mp4v"):
            candidate = cv2.VideoWriter(
                str(annotated_path),
                cv2.VideoWriter_fourcc(*codec),
                fps,
                frame_size,
            )
            if candidate.isOpened():
                writer = candidate
                break
            candidate.release()
        if writer is not None:
            for annotated in annotated_frames:
                writer.write(annotated)
            writer.release()
    progress(100, "Done")
    return {
        "report": report,
        "frames_total": len(frames),
        "frames_with_pose": sum(1 for item in keypoint_frames if item),
        "video_path": "annotated.mp4" if annotated_path.exists() else None,
    }


registry = LiveRehabRegistry()
