"""Waterline estimation with temporal smoothing and explicit confidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from video_analysis.base_analyzer import BaseAnalyzer
from video_analysis.constants import (
    SWIM_WATERLINE_BRIDGE_DECAY,
    SWIM_WATERLINE_MAX_JUMP_RATIO,
    SWIM_WATERLINE_MAX_SLOPE,
    SWIM_WATERLINE_MIN_EDGE_STRENGTH,
)


@dataclass(frozen=True)
class WaterlineEstimate:
    slope: float
    intercept: float
    confidence: float
    observed: bool

    def y_at(self, x: float) -> float:
        return self.slope * x + self.intercept

    def to_dict(self, *, frame_width: int, frame_height: int) -> Dict[str, Any]:
        return {
            "slope_norm": round(self.slope * frame_width / max(frame_height, 1), 4),
            "intercept_norm": round(self.intercept / max(frame_height, 1), 4),
            "confidence": self.confidence,
            "observed": self.observed,
        }


class WaterlineAnalyzer(BaseAnalyzer):
    """Estimate a deck-side pool waterline without pretending every frame observes it."""

    def __init__(self) -> None:
        super().__init__()
        self._previous: Optional[WaterlineEstimate] = None

    def analyze(
        self,
        frame: np.ndarray,
        swimmer_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> WaterlineEstimate:
        if frame is None or frame.size == 0:
            return self._bridge_previous()
        observed = self._estimate_frame(frame, swimmer_bbox)
        if observed is None:
            return self._bridge_previous()
        stable = self._reject_jump(observed, frame.shape[0], frame.shape[1])
        smoothed = self._smooth_estimate(stable)
        self._previous = smoothed
        return smoothed

    def _estimate_frame(
        self,
        frame: np.ndarray,
        swimmer_bbox: Optional[Tuple[int, int, int, int]],
    ) -> Optional[WaterlineEstimate]:
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        gradient = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3))

        row_scores = gradient.mean(axis=1)
        start = max(2, int(height * 0.05))
        end = min(height - 2, int(height * 0.95))
        if swimmer_bbox is not None:
            _, y1, _, y2 = swimmer_bbox
            margin = max(8, int((y2 - y1) * 0.2))
            start = max(start, y1 - margin)
            end = min(end, y2 + margin)
        if end <= start:
            return None

        local_index = int(np.argmax(row_scores[start:end]))
        candidate_y = start + local_index
        edge_strength = float(row_scores[candidate_y])
        if edge_strength < SWIM_WATERLINE_MIN_EDGE_STRENGTH:
            return None

        row = gradient[candidate_y]
        support_threshold = max(SWIM_WATERLINE_MIN_EDGE_STRENGTH, edge_strength * 0.45)
        support = float(np.count_nonzero(row >= support_threshold)) / max(width, 1)
        contrast = self._vertical_contrast(blurred, candidate_y)
        slope, intercept, line_support = self._hough_refinement(blurred, candidate_y)

        edge_score = min(1.0, edge_strength / 80.0)
        support_score = min(1.0, max(support, line_support) / 0.45)
        contrast_score = min(1.0, contrast / 70.0)
        confidence = 0.45 * edge_score + 0.35 * support_score + 0.20 * contrast_score
        return WaterlineEstimate(
            slope=slope,
            intercept=intercept,
            confidence=round(max(0.0, min(1.0, confidence)), 2),
            observed=True,
        )

    @staticmethod
    def _vertical_contrast(gray: np.ndarray, y: int) -> float:
        height = gray.shape[0]
        upper = gray[max(0, y - 6) : max(1, y - 2)]
        lower = gray[min(height - 1, y + 2) : min(height, y + 6)]
        if upper.size == 0 or lower.size == 0:
            return 0.0
        return abs(float(upper.mean()) - float(lower.mean()))

    @staticmethod
    def _hough_refinement(gray: np.ndarray, candidate_y: int) -> Tuple[float, float, float]:
        height, width = gray.shape
        edges = cv2.Canny(gray, 40, 120)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(20, width // 6),
            minLineLength=max(30, width // 3),
            maxLineGap=max(8, width // 20),
        )
        candidates: List[Tuple[float, float, float]] = []
        if lines is not None:
            for raw in lines[:, 0]:
                x1, y1, x2, y2 = (int(value) for value in raw)
                dx = x2 - x1
                if dx == 0:
                    continue
                slope = (y2 - y1) / dx
                if abs(slope) > SWIM_WATERLINE_MAX_SLOPE:
                    continue
                intercept = y1 - slope * x1
                centre_y = slope * (width / 2.0) + intercept
                if abs(centre_y - candidate_y) > max(10, height * 0.08):
                    continue
                length = float(np.hypot(dx, y2 - y1))
                candidates.append((length, slope, intercept))
        if not candidates:
            return 0.0, float(candidate_y), 0.0
        length, slope, intercept = max(candidates, key=lambda item: item[0])
        return float(slope), float(intercept), min(1.0, length / max(width, 1))

    def _reject_jump(
        self,
        estimate: WaterlineEstimate,
        frame_height: int,
        frame_width: int,
    ) -> WaterlineEstimate:
        if self._previous is None:
            return estimate
        centre_x = frame_width / 2.0
        previous_y = self._previous.y_at(centre_x)
        observed_y = estimate.y_at(centre_x)
        max_jump = max(8.0, frame_height * SWIM_WATERLINE_MAX_JUMP_RATIO)
        delta = observed_y - previous_y
        if abs(delta) <= max_jump:
            return estimate
        guarded_y = previous_y + np.sign(delta) * max_jump
        intercept = guarded_y - estimate.slope * centre_x
        return WaterlineEstimate(
            slope=estimate.slope,
            intercept=float(intercept),
            confidence=round(estimate.confidence * 0.65, 2),
            observed=True,
        )

    def _smooth_estimate(self, estimate: WaterlineEstimate) -> WaterlineEstimate:
        if self._previous is None:
            return estimate
        slope = self._ema("waterline_slope", estimate.slope)
        intercept = self._ema("waterline_intercept", estimate.intercept)
        confidence = self._ema("waterline_confidence", estimate.confidence)
        return WaterlineEstimate(
            slope=float(slope),
            intercept=float(intercept),
            confidence=round(max(0.0, min(1.0, confidence)), 2),
            observed=estimate.observed,
        )

    def _bridge_previous(self) -> WaterlineEstimate:
        if self._previous is None:
            return WaterlineEstimate(slope=0.0, intercept=0.0, confidence=0.0, observed=False)
        bridged = WaterlineEstimate(
            slope=self._previous.slope,
            intercept=self._previous.intercept,
            confidence=round(self._previous.confidence * SWIM_WATERLINE_BRIDGE_DECAY, 2),
            observed=False,
        )
        self._previous = bridged
        return bridged
