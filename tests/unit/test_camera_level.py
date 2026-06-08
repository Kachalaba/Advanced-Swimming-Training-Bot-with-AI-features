"""Tests for relative optical camera-level estimation."""

import cv2
import numpy as np

from backend.app.services.camera_level import CameraLevelEstimator


def _feature_frame() -> np.ndarray:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for x in range(20, 320, 40):
        for y in range(20, 240, 40):
            cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
            cv2.line(frame, (x - 7, y), (x + 7, y), (120, 220, 255), 2)
            cv2.line(frame, (x, y - 7), (x, y + 7), (120, 220, 255), 2)
    cv2.putText(frame, "SPRINT", (72, 128), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 180, 80), 2)
    return frame


def _rotate(frame: np.ndarray, angle: float) -> np.ndarray:
    height, width = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(frame, matrix, (width, height))


def test_calibrated_frame_reports_zero_roll():
    estimator = CameraLevelEstimator()

    result = estimator.calibrate(_feature_frame())

    assert result["status"] == "level"
    assert result["angle_deg"] == 0.0
    assert result["confidence"] >= 0.5


def test_rotated_frame_reports_relative_roll():
    estimator = CameraLevelEstimator()
    estimator.calibrate(_feature_frame())

    result = estimator.measure(_rotate(_feature_frame(), 3.0))

    assert result["confidence"] >= 0.5
    assert result["angle_deg"] is not None
    assert 2.0 <= abs(result["angle_deg"]) <= 4.0
    assert result["status"] == "adjust"


def test_featureless_frame_requests_recalibration():
    estimator = CameraLevelEstimator()
    estimator.calibrate(_feature_frame())

    result = estimator.measure(np.zeros((240, 320, 3), dtype=np.uint8))

    assert result["status"] == "recalibrate"
    assert result["angle_deg"] is None
