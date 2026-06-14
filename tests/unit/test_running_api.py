from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.app.api import analysis  # noqa: E402
from backend.app.services.jobs import JobRegistry  # noqa: E402


def _client(monkeypatch, tmp_path):
    registry = JobRegistry(tmp_path / "jobs")
    monkeypatch.setattr(analysis, "registry", registry)
    app = FastAPI()
    app.include_router(analysis.router, prefix="/api/analysis")
    return TestClient(app), registry


def _completed_job(registry):
    job = registry.create("running")
    job.status = "done"
    job.result = {
        "type": "result",
        "analysis": {
            "cadence": 176.0,
            "foot_strike_type": "midfoot",
            "arm_symmetry": 92.0,
            "efficiency_score": 87.0,
        },
        "frames_total": 100,
        "frames_with_pose": 90,
        "video_path": "annotated.mp4",
    }
    (job.workspace / "annotated.mp4").write_bytes(b"running-video")
    return job


def test_running_save_uses_stable_athlete_id_and_is_idempotent(monkeypatch, tmp_path):
    client, registry = _client(monkeypatch, tmp_path)
    job = _completed_job(registry)
    saved = []
    monkeypatch.setenv("SESSION_VIDEO_DIR", str(tmp_path / "videos"))

    def fake_save(**kwargs):
        saved.append(kwargs)
        return 42

    monkeypatch.setattr(analysis, "save_analysis_to_athlete", fake_save)

    first = client.post(
        f"/api/analysis/running/{job.id}/save",
        json={"athlete_id": 7},
    )
    second = client.post(
        f"/api/analysis/running/{job.id}/save",
        json={"athlete_id": 7},
    )

    assert first.status_code == 200
    assert first.json() == {"session_id": 42}
    assert second.json() == {"session_id": 42}
    assert len(saved) == 1
    assert saved[0]["athlete_id"] == 7
    assert saved[0]["session_type"] == "running"
    assert saved[0]["analysis"]["running_analysis"] == job.result
    assert Path(saved[0]["video_path"]).read_bytes() == b"running-video"
    assert registry.get(job.id).saved_session_id == 42


def test_running_save_is_idempotent_under_concurrent_requests(monkeypatch, tmp_path):
    _, registry = _client(monkeypatch, tmp_path)
    job = _completed_job(registry)
    calls = 0

    def fake_save(**kwargs):
        nonlocal calls
        calls += 1
        time.sleep(0.05)
        return 44

    monkeypatch.setattr(analysis, "save_analysis_to_athlete", fake_save)

    async def save_twice():
        request = analysis.SaveAnalysisRequest(athlete_id=7)
        return await asyncio.gather(
            analysis.save_running_job(job.id, request),
            analysis.save_running_job(job.id, request),
        )

    results = asyncio.run(save_twice())

    assert results == [{"session_id": 44}, {"session_id": 44}]
    assert calls == 1


def test_running_save_keeps_name_fallback(monkeypatch, tmp_path):
    client, registry = _client(monkeypatch, tmp_path)
    job = _completed_job(registry)
    saved = {}

    def fake_save(**kwargs):
        saved.update(kwargs)
        return 43

    monkeypatch.setattr(analysis, "save_analysis_to_db", fake_save)

    response = client.post(
        f"/api/analysis/running/{job.id}/save",
        json={"athlete_name": "Legacy Runner"},
    )

    assert response.status_code == 200
    assert saved["athlete_name"] == "Legacy Runner"


def test_running_save_rejects_incomplete_and_wrong_kind_jobs(monkeypatch, tmp_path):
    client, registry = _client(monkeypatch, tmp_path)
    queued = registry.create("running")
    swimming = registry.create("swimming")

    incomplete = client.post(
        f"/api/analysis/running/{queued.id}/save",
        json={"athlete_id": 7},
    )
    wrong_kind = client.post(
        f"/api/analysis/running/{swimming.id}/save",
        json={"athlete_id": 7},
    )

    assert incomplete.status_code == 409
    assert wrong_kind.status_code == 404


def test_running_status_includes_saved_session_id(monkeypatch, tmp_path):
    client, registry = _client(monkeypatch, tmp_path)
    job = _completed_job(registry)
    job.saved_session_id = 55

    response = client.get(f"/api/analysis/running/{job.id}")

    assert response.status_code == 200
    assert response.json()["saved_session_id"] == 55
