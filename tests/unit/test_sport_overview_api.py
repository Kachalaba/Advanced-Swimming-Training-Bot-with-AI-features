from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api import athletes
from video_analysis.athlete_database import Athlete, AthleteDatabase, TrainingSession


def _client(monkeypatch, tmp_path):
    database = AthleteDatabase(str(tmp_path / "athletes.db"))
    monkeypatch.setattr(athletes, "get_database", lambda: database)
    app = FastAPI()
    app.include_router(athletes.router, prefix="/api/athletes")
    return TestClient(app), database


def test_empty_running_overview_returns_stable_empty_contract(monkeypatch, tmp_path):
    client, database = _client(monkeypatch, tmp_path)
    athlete_id = database.add_athlete(Athlete(name="Runner"))

    response = client.get(f"/api/athletes/{athlete_id}/sports/running/overview")

    assert response.status_code == 200
    assert response.json() == {
        "athlete": {
            "id": str(athlete_id),
            "name": "Runner",
            "initials": "R",
            "handle": None,
        },
        "sport": "running",
        "total_sessions": 0,
        "latest_session_date": None,
        "latest_score": None,
        "headline_metrics": {},
        "insights": [],
        "score_series": [],
        "sessions": [],
    }


def test_running_overview_reads_persisted_sessions(monkeypatch, tmp_path):
    client, database = _client(monkeypatch, tmp_path)
    athlete_id = database.add_athlete(Athlete(name="Runner"))
    database.add_session(
        TrainingSession(
            athlete_id=athlete_id,
            session_type="running",
            date="2026-06-14T10:00:00",
            full_analysis=json.dumps(
                {
                    "running_analysis": {
                        "analysis": {
                            "cadence": 178,
                            "foot_strike_type": "midfoot",
                            "efficiency_score": 88,
                        }
                    }
                }
            ),
        )
    )

    response = client.get(f"/api/athletes/{athlete_id}/sports/running/overview")

    assert response.status_code == 200
    body = response.json()
    assert body["total_sessions"] == 1
    assert body["latest_score"] == 88.0
    assert body["headline_metrics"]["cadence"]["value"] == 178.0
    assert body["sessions"][0]["summary"] == "178 spm · Midfoot strike"


def test_sport_overview_returns_404_for_unknown_athlete(monkeypatch, tmp_path):
    client, _ = _client(monkeypatch, tmp_path)

    response = client.get("/api/athletes/999/sports/running/overview")

    assert response.status_code == 404


def test_sport_overview_rejects_unsupported_sport(monkeypatch, tmp_path):
    client, database = _client(monkeypatch, tmp_path)
    athlete_id = database.add_athlete(Athlete(name="Runner"))

    response = client.get(f"/api/athletes/{athlete_id}/sports/rowing/overview")

    assert response.status_code == 422
