"""Synthetic-frame tests for temporal waterline estimation."""

from __future__ import annotations

import cv2
import numpy as np

from video_analysis.waterline_analyzer import WaterlineAnalyzer


def _frame_with_waterline(y: int, width: int = 320, height: int = 180):
    frame = np.full((height, width, 3), (55, 45, 35), dtype=np.uint8)
    frame[y:, :] = (130, 85, 35)
    cv2.line(frame, (0, y), (width - 1, y), (230, 230, 230), 2)
    return frame


def test_detects_horizontal_waterline():
    estimate = WaterlineAnalyzer().analyze(_frame_with_waterline(82))

    assert estimate.observed is True
    assert abs(estimate.y_at(160) - 82) <= 5
    assert estimate.confidence >= 0.55


def test_missing_frame_reuses_last_line_with_decaying_confidence():
    analyzer = WaterlineAnalyzer()
    first = analyzer.analyze(_frame_with_waterline(82))

    missing = analyzer.analyze(np.zeros((180, 320, 3), dtype=np.uint8))

    assert missing.observed is False
    assert 0.0 < missing.confidence < first.confidence
    assert missing.y_at(160) == first.y_at(160)


def test_rejects_single_frame_jump():
    analyzer = WaterlineAnalyzer()
    analyzer.analyze(_frame_with_waterline(80))

    jumped = analyzer.analyze(_frame_with_waterline(145))

    assert jumped.y_at(160) < 120
    assert jumped.confidence < 1.0


def test_waterline_serializes_normalized_coordinates():
    estimate = WaterlineAnalyzer().analyze(_frame_with_waterline(90, width=400, height=200))

    payload = estimate.to_dict(frame_width=400, frame_height=200)

    assert abs(payload["intercept_norm"] - 0.45) <= 0.03
    assert payload["confidence"] == estimate.confidence
    assert payload["observed"] is True
