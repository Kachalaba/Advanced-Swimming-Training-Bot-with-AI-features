"""API contract tests for web rehabilitation analysis."""

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api import rehabilitation


class _FakeSession:
    id = "session-123"

    def process_frame(self, frame, calibrate=False):
        return {
            "session_id": self.id,
            "sequence": 1,
            "pose_detected": True,
            "landmarks": {"left_shoulder": {"x": 0.3, "y": 0.3}},
            "posture": {"available": True},
            "camera_level": {"angle_deg": 0.0 if calibrate else 0.4, "status": "level"},
            "report": {"protocol": "shoulder_flexion"},
        }

    def snapshot(self):
        return self.process_frame(np.zeros((10, 10, 3), dtype=np.uint8))


class _FakeRegistry:
    def __init__(self):
        self.session = _FakeSession()

    def create(self, protocol, fps):
        return self.session

    def get(self, session_id):
        return self.session if session_id == self.session.id else None

    def delete(self, session_id):
        return session_id == self.session.id


def _client(monkeypatch):
    monkeypatch.setattr(rehabilitation, "live_registry", _FakeRegistry())
    app = FastAPI()
    app.include_router(rehabilitation.router, prefix="/api/analysis")
    return TestClient(app)


def _jpeg() -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.zeros((32, 48, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def test_create_live_session_validates_protocol(monkeypatch):
    client = _client(monkeypatch)

    response = client.post(
        "/api/analysis/rehabilitation/live",
        json={"protocol": "shoulder_flexion", "fps": 5},
    )
    invalid = client.post(
        "/api/analysis/rehabilitation/live",
        json={"protocol": "unknown", "fps": 5},
    )

    assert response.status_code == 200
    assert response.json()["session_id"] == "session-123"
    assert invalid.status_code == 400


def test_live_frame_returns_overlay_contract(monkeypatch):
    client = _client(monkeypatch)

    response = client.post(
        "/api/analysis/rehabilitation/live/session-123/frame",
        data={"calibrate": "true"},
        files={"image": ("frame.jpg", _jpeg(), "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["posture"]["available"] is True
    assert payload["camera_level"]["angle_deg"] == 0.0
    assert payload["report"]["protocol"] == "shoulder_flexion"


def test_live_frame_rejects_bad_image_and_unknown_session(monkeypatch):
    client = _client(monkeypatch)

    bad_image = client.post(
        "/api/analysis/rehabilitation/live/session-123/frame",
        files={"image": ("frame.jpg", b"not-an-image", "image/jpeg")},
    )
    missing = client.get("/api/analysis/rehabilitation/live/missing")

    assert bad_image.status_code == 400
    assert missing.status_code == 404


def test_live_report_can_be_saved_to_history(monkeypatch):
    client = _client(monkeypatch)
    saved = {}

    def fake_save(**kwargs):
        saved.update(kwargs)
        return 42

    monkeypatch.setattr(rehabilitation, "save_analysis_to_db", fake_save)

    response = client.post(
        "/api/analysis/rehabilitation/live/session-123/save",
        json={"athlete_name": "Nikita K."},
    )

    assert response.status_code == 200
    assert response.json() == {"session_id": 42}
    assert saved["session_type"] == "rehab"
    assert saved["analysis"]["rehab_analysis"]["protocol"] == "shoulder_flexion"
