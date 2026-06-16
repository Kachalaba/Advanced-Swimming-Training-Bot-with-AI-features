from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.app.api import analysis  # noqa: E402
from backend.app.services.jobs import JobRegistry  # noqa: E402


class ImmediateThread:
    def __init__(self, target, args=(), kwargs=None, daemon=None):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}

    def start(self):
        self.target(*self.args, **self.kwargs)


def _result_payload():
    return {
        "type": "result",
        "exercise_type": "squat",
        "analysis": {
            "exercise_type": "squat",
            "tracked_joint": "knee",
            "total_reps": 4,
            "avg_tempo": 2.1,
            "avg_range_of_motion": 72.0,
            "stability_score": 88.0,
            "min_angle": 90.0,
            "max_angle": 170.0,
            "reps": [],
            "angle_history": [],
        },
        "frames_total": 100,
        "frames_with_pose": 88,
        "quality": {
            "status": "pass",
            "pose_coverage": 88.0,
            "metric_ready_frames": 80,
            "minimum_required_frames": 20,
            "warnings": [],
        },
        "video_path": "annotated.mp4",
    }


def _client(monkeypatch, tmp_path):
    registry = JobRegistry(tmp_path / "jobs")
    monkeypatch.setattr(analysis, "registry", registry)
    monkeypatch.setattr(analysis, "Thread", ImmediateThread)

    def fake_runner(job, video_path, exercise_type, fps):
        job.status = "running"
        job.push_event({"type": "progress", "pct": 25, "label": "Measuring dryland movement"})
        (job.workspace / "annotated.mp4").write_bytes(b"dryland-video")
        payload = _result_payload() | {"exercise_type": exercise_type}
        job.result = payload
        job.push_event(payload)
        job.status = "done"

    monkeypatch.setattr(
        analysis,
        "_run_dryland_pipeline_in_thread",
        fake_runner,
        raising=False,
    )
    app = FastAPI()
    app.include_router(analysis.router, prefix="/api/analysis")
    return TestClient(app), registry


def _video_file():
    return {"video": ("squat.mp4", b"video-bytes", "video/mp4")}


def test_dryland_upload_rejects_unsupported_exercise(monkeypatch, tmp_path):
    client, _registry = _client(monkeypatch, tmp_path)

    response = client.post(
        "/api/analysis/dryland",
        files=_video_file(),
        data={"exercise_type": "plank", "fps": "15"},
    )

    assert response.status_code == 400
    assert "Unsupported dryland exercise" in response.json()["detail"]


def test_dryland_upload_runs_job_and_exposes_result(monkeypatch, tmp_path):
    client, registry = _client(monkeypatch, tmp_path)

    response = client.post(
        "/api/analysis/dryland",
        files=_video_file(),
        data={"exercise_type": "squat", "fps": "15"},
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    job = registry.get(job_id)
    assert job is not None
    assert job.kind == "dryland"
    assert job.status == "done"
    assert job.result["analysis"]["total_reps"] == 4

    status = client.get(f"/api/analysis/dryland/{job_id}")
    events = client.get(f"/api/analysis/dryland/{job_id}/events")
    video = client.get(f"/api/analysis/dryland/{job_id}/video")
    assert status.status_code == 200
    assert status.json()["result"]["quality"]["status"] == "pass"
    assert '"type": "result"' in events.text
    assert video.content == b"dryland-video"


def test_dryland_save_is_idempotent_and_uses_stable_athlete_id(
    monkeypatch,
    tmp_path,
):
    client, registry = _client(monkeypatch, tmp_path)
    saved = []
    monkeypatch.setenv("SESSION_VIDEO_DIR", str(tmp_path / "videos"))

    def fake_save(**kwargs):
        saved.append(kwargs)
        return 91

    monkeypatch.setattr(analysis, "save_analysis_to_athlete", fake_save)
    job_id = client.post(
        "/api/analysis/dryland",
        files=_video_file(),
        data={"exercise_type": "squat"},
    ).json()["job_id"]

    first = client.post(
        f"/api/analysis/dryland/{job_id}/save",
        json={"athlete_id": 7},
    )
    second = client.post(
        f"/api/analysis/dryland/{job_id}/save",
        json={"athlete_id": 7},
    )

    assert first.json() == {"session_id": 91}
    assert second.json() == {"session_id": 91}
    assert len(saved) == 1
    assert saved[0]["athlete_id"] == 7
    assert saved[0]["session_type"] == "dryland"
    assert saved[0]["analysis"]["dryland_analysis"]["analysis"]["total_reps"] == 4
    assert Path(saved[0]["video_path"]).read_bytes() == b"dryland-video"
    assert registry.get(job_id).saved_session_id == 91


def test_dryland_save_requires_done_job(monkeypatch, tmp_path):
    client, registry = _client(monkeypatch, tmp_path)
    job = registry.create("dryland")

    response = client.post(
        f"/api/analysis/dryland/{job.id}/save",
        json={"athlete_name": "Nikita"},
    )

    assert response.status_code == 409


def test_dryland_routes_reject_other_job_kinds(monkeypatch, tmp_path):
    client, registry = _client(monkeypatch, tmp_path)
    running = registry.create("running")

    assert client.get(f"/api/analysis/dryland/{running.id}").status_code == 404
    assert (
        client.post(
            f"/api/analysis/dryland/{running.id}/save",
            json={"athlete_id": 7},
        ).status_code
        == 404
    )
