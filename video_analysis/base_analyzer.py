"""
Shared base class for all sport-specific biomechanics analyzers.

Consolidates utilities duplicated across stroke_analyzer, running_analyzer,
and cycling_analyzer: keypoint extraction, angle calculation, smoothing,
and EMA-based temporal filtering.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from video_analysis.constants import (
    EMA_ALPHA,
    MEDIAPIPE_POSE_CONFIG,
    MEDIAPIPE_POSE_VIDEO_CONFIG,
    SMOOTHING_WINDOW_SIZE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level MediaPipe Pose cache
# Avoids re-loading heavy model weights on every analyzer instantiation.
# ---------------------------------------------------------------------------
_pose_cache: dict = {}


def get_pose_detector(video_mode: bool = False):
    """Return a cached MediaPipe Pose instance.

    Args:
        video_mode: Use streaming (static_image_mode=False) vs single-image mode.
    """
    key = "video" if video_mode else "image"
    if key not in _pose_cache:
        try:
            import mediapipe as mp  # lazy import — heavy dependency

            config = MEDIAPIPE_POSE_VIDEO_CONFIG if video_mode else MEDIAPIPE_POSE_CONFIG
            _pose_cache[key] = mp.solutions.pose.Pose(**config)
            logger.info("MediaPipe Pose loaded (mode=%s)", key)
        except Exception as exc:
            logger.error("Failed to initialize MediaPipe Pose: %s", exc)
            raise
    return _pose_cache[key]


Point2D = Tuple[float, float]


class BaseAnalyzer:
    """Shared utilities mixin for sport biomechanics analyzers.

    Provides:
    - ``_get_point``        — unified keypoint extraction (list/tuple/object formats)
    - ``_calculate_angle``  — 3-point joint angle (degrees)
    - ``_angle_2d``         — signed vector angle from horizontal
    - ``_smooth``           — sliding-window mean filter
    - ``_ema``              — per-key exponential moving average
    - ``_apply_ema_to_dict``— batch EMA over a metrics dict
    - ``_score_in_range``   — 0-100 score based on distance from optimal range
    """

    def __init__(self, ema_alpha: float = EMA_ALPHA) -> None:
        self._ema_alpha = ema_alpha
        self._ema_state: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Keypoint helpers
    # ------------------------------------------------------------------

    def _get_point(
        self,
        kps: Dict,
        name: str,
        alt_names: Optional[Dict[str, List[Union[str, int]]]] = None,
    ) -> Optional[Point2D]:
        """Extract (x, y) from a keypoint dict, tolerating several storage formats.

        Supported formats:
        - ``{"left_knee": (x, y)}``
        - ``{"left_knee": [x, y, z, vis]}``
        - ``{"left_knee": obj}`` where obj has ``.x`` / ``.y`` attributes
        - Integer index keys (MediaPipe raw)
        """
        if name in kps:
            p = kps[name]
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                return float(p[0]), float(p[1])
            if hasattr(p, "x") and hasattr(p, "y"):
                return float(p.x), float(p.y)

        if alt_names and name in alt_names:
            for alias in alt_names[name]:
                if alias in kps:
                    p = kps[alias]
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        return float(p[0]), float(p[1])
                    if hasattr(p, "x") and hasattr(p, "y"):
                        return float(p.x), float(p.y)

        return None

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_angle(
        p1: Point2D,
        p2: Point2D,
        p3: Point2D,
    ) -> float:
        """Return the angle (degrees) at *p2* in the p1-p2-p3 triplet.

        Returns 0 if any input point is None or vectors are degenerate.
        """
        if p1 is None or p2 is None or p3 is None:
            return 0.0
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        mag1 = math.hypot(*v1)
        mag2 = math.hypot(*v2)
        if mag1 * mag2 == 0:
            return 0.0
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_a))

    @staticmethod
    def _angle_2d(p1: Point2D, p2: Point2D) -> float:
        """Return the signed angle (degrees) of the vector p1→p2 from horizontal."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.degrees(math.atan2(dy, dx))

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------

    @staticmethod
    def _smooth(values: List[float], window: int = SMOOTHING_WINDOW_SIZE) -> List[float]:
        """Apply a uniform sliding-window mean filter to *values*."""
        if len(values) < window:
            return list(values)
        kernel = np.ones(window) / window
        return np.convolve(values, kernel, mode="same").tolist()

    def _ema(self, key: str, value: float) -> float:
        """Return the EMA-smoothed value for *key*, updating internal state."""
        prev = self._ema_state.get(key, value)
        smoothed = self._ema_alpha * value + (1.0 - self._ema_alpha) * prev
        self._ema_state[key] = smoothed
        return smoothed

    def _apply_ema_to_dict(self, metrics: Dict[str, float], prefix: str = "") -> Dict[str, float]:
        """Apply EMA smoothing to all numeric values in *metrics*.

        Non-numeric values are passed through unchanged.
        """
        result: Dict = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                result[k] = self._ema(f"{prefix}{k}", float(v))
            else:
                result[k] = v
        return result

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_in_range(
        value: float,
        optimal_min: float,
        optimal_max: float,
        penalty_per_unit: float = 2.0,
        max_score: float = 100.0,
    ) -> float:
        """Return a 0-100 score based on how close *value* is to [optimal_min, optimal_max]."""
        if optimal_min <= value <= optimal_max:
            return max_score
        deviation = min(abs(value - optimal_min), abs(value - optimal_max))
        return max(0.0, max_score - deviation * penalty_per_unit)
