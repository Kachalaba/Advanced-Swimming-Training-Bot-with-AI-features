from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.app.api import athletes  # noqa: E402
from video_analysis.athlete_database import Athlete, AthleteDatabase, TrainingSession  # noqa: E402


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


def test_cycling_overview_reads_fit_metrics_and_evidence(monkeypatch, tmp_path):
    client, database = _client(monkeypatch, tmp_path)
    athlete_id = database.add_athlete(Athlete(name="Cyclist"))
    database.add_session(
        TrainingSession(
            athlete_id=athlete_id,
            session_type="cycling",
            date="2026-06-14T11:00:00",
            duration_sec=15.0,
            video_path="/tmp/cycling.mp4",
            full_analysis=json.dumps(
                {
                    "cycling_analysis": {
                        "analysis": {
                            "cadence": 92,
                            "avg_knee_angle_bottom": 147,
                            "upper_body_stability": 89,
                            "pedal_smoothness": 84,
                            "bike_fit_score": 87,
                            "rock_detected": False,
                        },
                        "frames_total": 120,
                        "frames_with_pose": 108,
                    }
                }
            ),
        )
    )

    response = client.get(f"/api/athletes/{athlete_id}/sports/cycling/overview")

    assert response.status_code == 200
    body = response.json()
    assert body["latest_score"] == 87.0
    assert body["headline_metrics"]["cadence"]["value"] == 92.0
    assert body["headline_metrics"]["knee_extension"]["value"] == 147.0
    assert body["sessions"][0]["quality"]["pose_coverage"] == 90.0
    assert body["sessions"][0]["has_video"] is True
    assert body["sessions"][0]["summary"] == "92 rpm · 147° knee extension"


def test_dryland_overview_reads_repetition_metrics_and_evidence(monkeypatch, tmp_path):
    client, database = _client(monkeypatch, tmp_path)
    athlete_id = database.add_athlete(Athlete(name="Dryland Athlete"))
    database.add_session(
        TrainingSession(
            athlete_id=athlete_id,
            session_type="dryland",
            date="2026-06-14T12:00:00",
            duration_sec=11.0,
            video_path="/tmp/dryland.mp4",
            full_analysis=json.dumps(
                {
                    "dryland_analysis": {
                        "exercise_type": "push_up",
                        "analysis": {
                            "total_reps": 6,
                            "avg_tempo": 1.8,
                            "avg_range_of_motion": 86,
                            "stability_score": 91,
                        },
                        "quality": {
                            "pose_coverage": 84,
                            "metric_ready_frames": 92,
                            "minimum_required_frames": 20,
                        },
                    }
                }
            ),
        )
    )

    response = client.get(f"/api/athletes/{athlete_id}/sports/dryland/overview")

    assert response.status_code == 200
    body = response.json()
    assert body["latest_score"] == 91.0
    assert body["headline_metrics"]["total_reps"]["value"] == 6.0
    assert body["headline_metrics"]["avg_tempo"]["unit"] == "sec"
    assert body["sessions"][0]["quality"]["pose_coverage"] == 84.0
    assert body["sessions"][0]["quality"]["metric_ready_frames"] == 92.0
    assert body["sessions"][0]["summary"] == "Push-Up · 6 reps · 1.8s tempo"


def test_sport_overview_returns_404_for_unknown_athlete(monkeypatch, tmp_path):
    client, _ = _client(monkeypatch, tmp_path)

    response = client.get("/api/athletes/999/sports/running/overview")

    assert response.status_code == 404


def test_sport_overview_rejects_unsupported_sport(monkeypatch, tmp_path):
    client, database = _client(monkeypatch, tmp_path)
    athlete_id = database.add_athlete(Athlete(name="Runner"))

    response = client.get(f"/api/athletes/{athlete_id}/sports/rowing/overview")

    assert response.status_code == 422
