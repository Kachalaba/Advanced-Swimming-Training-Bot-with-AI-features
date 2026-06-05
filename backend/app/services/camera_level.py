"""Relative optical camera-roll estimation for calibrated live sessions."""

from __future__ import annotations

import math
import threading
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

LEVEL_TOLERANCE_DEG = 0.7
MIN_CONFIDENCE = 0.35
MIN_MATCHES = 8


class CameraLevelEstimator:
    """Estimate roll relative to a user-selected reference frame."""

    def __init__(self, max_features: int = 900) -> None:
        self._orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=8)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._reference_keypoints: Sequence[Any] = ()
        self._reference_descriptors: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _mask(
        frame: np.ndarray,
        athlete_bbox: Optional[Tuple[int, int, int, int]],
    ) -> Optional[np.ndarray]:
        if athlete_bbox is None:
            return None
        mask = np.full(frame.shape[:2], 255, dtype=np.uint8)
        x1, y1, x2, y2 = athlete_bbox
        cv2.rectangle(mask, (max(0, x1), max(0, y1)), (max(0, x2), max(0, y2)), 0, -1)
        return mask

    @staticmethod
    def _result(
        angle_deg: Optional[float],
        confidence: float,
        status: str,
    ) -> Dict[str, Any]:
        direction = "level"
        if angle_deg is not None and abs(angle_deg) > LEVEL_TOLERANCE_DEG:
            direction = "right_edge_high" if angle_deg > 0 else "left_edge_high"
        return {
            "angle_deg": None if angle_deg is None else round(float(angle_deg), 1),
            "confidence": round(float(max(0.0, min(1.0, confidence))), 2),
            "status": status,
            "direction": direction,
            "relative": True,
        }

    def calibrate(
        self,
        frame: np.ndarray,
        athlete_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, Any]:
        """Store the current frame as the zero-roll reference."""
        gray = self._gray(frame)
        keypoints, descriptors = self._orb.detectAndCompute(gray, self._mask(gray, athlete_bbox))
        confidence = min(1.0, len(keypoints) / 40.0) if keypoints else 0.0
        with self._lock:
            self._reference_keypoints = keypoints or ()
            self._reference_descriptors = descriptors
        if descriptors is None or len(keypoints) < MIN_MATCHES:
            return self._result(None, confidence, "recalibrate")
        return self._result(0.0, confidence, "level")

    def measure(
        self,
        frame: np.ndarray,
        athlete_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, Any]:
        """Measure roll relative to the latest valid calibration frame."""
        with self._lock:
            reference_keypoints = tuple(self._reference_keypoints)
            reference_descriptors = self._reference_descriptors
        if reference_descriptors is None or not reference_keypoints:
            return self._result(None, 0.0, "uncalibrated")

        gray = self._gray(frame)
        current_keypoints, current_descriptors = self._orb.detectAndCompute(
            gray,
            self._mask(gray, athlete_bbox),
        )
        if current_descriptors is None or len(current_keypoints) < MIN_MATCHES:
            return self._result(None, 0.0, "recalibrate")

        pairs = self._matcher.knnMatch(reference_descriptors, current_descriptors, k=2)
        matches = [first for first, second in pairs if first.distance < 0.75 * second.distance]
        if len(matches) < MIN_MATCHES:
            return self._result(None, len(matches) / float(MIN_MATCHES), "recalibrate")

        source = np.float32([reference_keypoints[match.queryIdx].pt for match in matches])
        target = np.float32([current_keypoints[match.trainIdx].pt for match in matches])
        matrix, inliers = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        if matrix is None or inliers is None:
            return self._result(None, 0.0, "recalibrate")

        inlier_ratio = float(inliers.ravel().sum()) / float(len(matches))
        match_coverage = min(1.0, len(matches) / 30.0)
        confidence = inlier_ratio * match_coverage
        if confidence < MIN_CONFIDENCE:
            return self._result(None, confidence, "recalibrate")

        angle = math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))
        status = "level" if abs(angle) <= LEVEL_TOLERANCE_DEG else "adjust"
        return self._result(angle, confidence, status)
