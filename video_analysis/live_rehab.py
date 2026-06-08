"""Stateful WebRTC frame processor for live rehabilitation analysis."""

from __future__ import annotations

import copy
import logging
import threading
from collections import deque
from typing import Any, Deque, Dict, Mapping, Optional, Protocol

import av
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase

from video_analysis.constants import REHAB_LIVE_ANALYSIS_INTERVAL_FRAMES, REHAB_LIVE_WINDOW_FRAMES

logger = logging.getLogger(__name__)


class _RehabAnalyzerProtocol(Protocol):
    def analyze(self, keypoints_list: list[Dict], protocol: str) -> Dict[str, Any]: ...


class _VisualizerProtocol(Protocol):
    def process_frame(
        self,
        frame: np.ndarray,
        idx: int = 0,
        bbox: Any = None,
    ) -> tuple[np.ndarray, Dict[str, Any]]: ...


class LiveRehabProcessor(VideoProcessorBase):
    """Analyze a bounded rolling window and annotate the live camera feed."""

    def __init__(
        self,
        analyzer: _RehabAnalyzerProtocol,
        visualizer: _VisualizerProtocol,
        protocol: str,
        labels: Mapping[str, str],
        max_frames: int = REHAB_LIVE_WINDOW_FRAMES,
        analysis_interval: int = REHAB_LIVE_ANALYSIS_INTERVAL_FRAMES,
    ) -> None:
        self.analyzer = analyzer
        self.visualizer = visualizer
        self.protocol = protocol
        self.labels = dict(labels)
        self.keypoint_frames: Deque[Dict] = deque(maxlen=max_frames)
        self.analysis_interval = max(1, analysis_interval)
        self.frame_count = 0
        self._report: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process one WebRTC frame and return an annotated frame."""
        image = frame.to_ndarray(format="bgr24")
        with self._lock:
            self.frame_count += 1
            try:
                annotated, pose_data = self.visualizer.process_frame(
                    image,
                    self.frame_count,
                )
                if pose_data.get("has_pose") and pose_data.get("keypoints"):
                    self.keypoint_frames.append(pose_data["keypoints"])
                if self.keypoint_frames and self.frame_count % self.analysis_interval == 0:
                    self._report = self.analyzer.analyze(
                        list(self.keypoint_frames),
                        protocol=self.protocol,
                    )
            except Exception:
                logger.exception("Live rehabilitation frame processing failed")
                annotated = image

            self._draw_overlay(annotated)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def get_report(self) -> Optional[Dict[str, Any]]:
        """Return a thread-safe snapshot of the latest rolling report."""
        with self._lock:
            return copy.deepcopy(self._report)

    def _draw_overlay(self, frame: np.ndarray) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (360, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        if not self._report:
            lines = [self.labels.get("collecting", "")]
        else:
            target = self._report["target_metrics"]
            lines = [
                (
                    f'{self.labels.get("rom", "")}: '
                    f'L {target["left"]["rom"]:.0f} / '
                    f'R {target["right"]["rom"]:.0f} deg'
                ),
                (f'{self.labels.get("reps", "")}: ' f'{self._report["total_correct_reps"]}'),
                (f'{self.labels.get("asymmetry", "")}: ' f'{self._report["symmetry"]["asymmetry_index"]:.1f}%'),
            ]

        for index, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (20, 42 + index * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 180),
                2,
                cv2.LINE_AA,
            )
