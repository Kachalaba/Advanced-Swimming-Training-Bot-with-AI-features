"""Tests for athlete and persisted history API contracts."""

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.app.api import athletes  # noqa: E402


class FakeDatabase:
    athlete = SimpleNamespace(id=3, name="Nikita K.")
    session = SimpleNamespace(
        id=17,
        athlete_id=3,
        session_type="rehab",
        date="2026-06-06T15:49:18",
        duration_sec=0,
        distance_m=0,
        reps=2,
        exercise_type="shoulder_flexion",
        ai_score=0,
        stability_score=88.0,
        symmetry_score=92.0,
        ai_summary="",
        notes="",
        video_path="/data/session-videos/rehab.mp4",
    )

    def get_athlete(self, athlete_id=None, name=None):
        if athlete_id == 3 or name == "Nikita K.":
            return self.athlete
        return None

    def get_all_athletes(self):
        return [self.athlete]

    def get_sessions(self, athlete_id, session_type=None, limit=100):
        return [self.session] if athlete_id == 3 else []


def test_athlete_and_session_endpoints_use_database(monkeypatch):
    monkeypatch.setattr(athletes, "get_database", lambda: FakeDatabase())
    app = FastAPI()
    app.include_router(athletes.router, prefix="/api/athletes")
    client = TestClient(app)

    me = client.get("/api/athletes/me")
    roster = client.get("/api/athletes")
    sessions = client.get("/api/athletes/3/sessions")

    assert me.json()["name"] == "Nikita K."
    assert roster.json()[0]["initials"] == "NK"
    assert sessions.json()[0] == {
        "id": 17,
        "athlete_id": 3,
        "session_type": "rehab",
        "date": "2026-06-06T15:49:18",
        "duration_sec": 0.0,
        "distance_m": 0.0,
        "reps": 2,
        "exercise_type": "shoulder_flexion",
        "score": 88.0,
        "summary": "shoulder_flexion",
        "has_video": True,
        "artifact_download_url": None,
    }


def test_unknown_athlete_returns_404(monkeypatch):
    monkeypatch.setattr(athletes, "get_database", lambda: FakeDatabase())
    app = FastAPI()
    app.include_router(athletes.router, prefix="/api/athletes")
    response = TestClient(app).get("/api/athletes/999/sessions")
    assert response.status_code == 404


def test_tool_session_exposes_download_url(monkeypatch):
    database = FakeDatabase()
    database.session = SimpleNamespace(
        **{
            **vars(FakeDatabase.session),
            "session_type": "tool",
            "exercise_type": "frame_extractor",
            "ai_summary": "Extracted 5 frames",
            "video_path": "/data/session-artifacts/tools/job/frames.zip",
        }
    )
    monkeypatch.setattr(athletes, "get_database", lambda: database)
    app = FastAPI()
    app.include_router(athletes.router, prefix="/api/athletes")

    payload = TestClient(app).get("/api/athletes/3/sessions").json()[0]

    assert payload["summary"] == "Extracted 5 frames"
    assert payload["artifact_download_url"] == "/api/tools/history/17/download"
