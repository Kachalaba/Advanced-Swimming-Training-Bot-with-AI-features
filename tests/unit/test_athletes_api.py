"""Tests for athlete and persisted history API contracts."""

import json
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


def _rehab_session(
    session_id,
    date,
    protocol="shoulder_flexion",
    *,
    left_rom=120.0,
    right_rom=110.0,
    symmetry=91.0,
    repetitions=3,
    completion=80.0,
    valid_frames=24,
    full_analysis=None,
):
    report = {
        "protocol": protocol,
        "valid_frames": valid_frames,
        "target_metrics": {
            "left": {"rom": left_rom},
            "right": {"rom": right_rom},
        },
        "symmetry": {"score": symmetry},
        "total_correct_reps": repetitions,
        "completion_score": completion,
    }
    return SimpleNamespace(
        **{
            **vars(FakeDatabase.session),
            "id": session_id,
            "date": date,
            "exercise_type": protocol,
            "full_analysis": (json.dumps({"rehab_analysis": report}) if full_analysis is None else full_analysis),
        }
    )


def test_rehabilitation_progress_normalizes_and_orders_sessions(monkeypatch):
    class ProgressDatabase(FakeDatabase):
        def get_sessions(self, athlete_id, session_type=None, limit=100):
            assert session_type == "rehab"
            return [
                _rehab_session(
                    22,
                    "2026-06-12T10:00:00",
                    left_rom=146.0,
                    right_rom=137.0,
                    symmetry=94.0,
                    repetitions=5,
                    completion=92.0,
                    valid_frames=None,
                ),
                _rehab_session(
                    18,
                    "2026-06-02T10:00:00",
                    left_rom=121.0,
                    right_rom=108.0,
                    symmetry=84.0,
                    repetitions=2,
                    completion=72.0,
                ),
            ]

    monkeypatch.setattr(athletes, "get_database", lambda: ProgressDatabase())
    app = FastAPI()
    app.include_router(athletes.router, prefix="/api/athletes")

    response = TestClient(app).get("/api/athletes/3/rehabilitation/progress")

    assert response.status_code == 200
    assert response.json() == {
        "athlete": {
            "id": "3",
            "name": "Nikita K.",
            "initials": "NK",
            "handle": "@kachamba_swim",
        },
        "protocols": ["shoulder_flexion"],
        "sessions": [
            {
                "id": 18,
                "date": "2026-06-02T10:00:00",
                "protocol": "shoulder_flexion",
                "leftRom": 121.0,
                "rightRom": 108.0,
                "symmetry": 84.0,
                "repetitions": 2,
                "completionScore": 72.0,
                "validFrames": 24,
                "hasVideo": True,
            },
            {
                "id": 22,
                "date": "2026-06-12T10:00:00",
                "protocol": "shoulder_flexion",
                "leftRom": 146.0,
                "rightRom": 137.0,
                "symmetry": 94.0,
                "repetitions": 5,
                "completionScore": 92.0,
                "validFrames": None,
                "hasVideo": True,
            },
        ],
    }


def test_rehabilitation_progress_filters_invalid_stored_reports(monkeypatch):
    class ProgressDatabase(FakeDatabase):
        def get_sessions(self, athlete_id, session_type=None, limit=100):
            return [
                _rehab_session(20, "2026-06-03", full_analysis="{not-json"),
                _rehab_session(21, "2026-06-04", protocol="unknown_protocol"),
                _rehab_session(22, "2026-06-05", left_rom=float("nan")),
                _rehab_session(23, "2026-06-06", protocol="knee_extension"),
            ]

    monkeypatch.setattr(athletes, "get_database", lambda: ProgressDatabase())
    app = FastAPI()
    app.include_router(athletes.router, prefix="/api/athletes")

    payload = TestClient(app).get("/api/athletes/3/rehabilitation/progress").json()

    assert [session["id"] for session in payload["sessions"]] == [23]
    assert payload["protocols"] == ["knee_extension"]


def test_rehabilitation_progress_unknown_athlete_returns_404(monkeypatch):
    monkeypatch.setattr(athletes, "get_database", lambda: FakeDatabase())
    app = FastAPI()
    app.include_router(athletes.router, prefix="/api/athletes")

    response = TestClient(app).get("/api/athletes/999/rehabilitation/progress")

    assert response.status_code == 404
