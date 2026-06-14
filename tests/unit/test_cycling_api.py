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
        "analysis": {
            "cadence": 91.0,
            "avg_knee_angle_top": 74.0,
            "avg_knee_angle_bottom": 146.0,
            "pedal_smoothness": 84.0,
            "upper_body_stability": 88.0,
            "bike_fit_score": 86.0,
            "efficiency_score": 83.0,
            "duration_sec": 12.0,
        },
        "frames_total": 120,
        "frames_with_pose": 108,
        "quality": {"status": "pass", "pose_coverage": 90.0},
        "video_path": "annotated.mp4",
    }


def _client(monkeypatch, tmp_path):
    registry = JobRegistry(tmp_path / "jobs")
    monkeypatch.setattr(analysis, "registry", registry)
    monkeypatch.setattr(analysis, "Thread", ImmediateThread)

    def fake_runner(job, video_path, fps):
        job.status = "running"
        job.push_event({"type": "progress", "pct": 25, "label": "Tracking cyclist"})
        (job.workspace / "annotated.mp4").write_bytes(b"cycling-video")
        job.result = _result_payload()
        job.push_event(job.result)
        job.status = "done"

    monkeypatch.setattr(
        analysis,
        "_run_cycling_pipeline_in_thread",
        fake_runner,
        raising=False,
    )
    app = FastAPI()
    app.include_router(analysis.router, prefix="/api/analysis")
    return TestClient(app), registry


def _video_file():
    return {"video": ("trainer.mp4", b"video-bytes", "video/mp4")}


def test_cycling_upload_runs_job_and_exposes_result(monkeypatch, tmp_path):
    client, registry = _client(monkeypatch, tmp_path)

    response = client.post(
        "/api/analysis/cycling",
        files=_video_file(),
        data={"fps": "30"},
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    job = registry.get(job_id)
    assert job is not None
    assert job.kind == "cycling"
    assert job.status == "done"
    assert job.result["analysis"]["cadence"] == 91.0

    status = client.get(f"/api/analysis/cycling/{job_id}")
    events = client.get(f"/api/analysis/cycling/{job_id}/events")
    video = client.get(f"/api/analysis/cycling/{job_id}/video")
    assert status.status_code == 200
    assert status.json()["result"]["quality"]["status"] == "pass"
    assert '"type": "result"' in events.text
    assert video.content == b"cycling-video"


def test_cycling_save_is_idempotent_and_uses_stable_athlete_id(
    monkeypatch,
    tmp_path,
):
    client, registry = _client(monkeypatch, tmp_path)
    saved = []
    monkeypatch.setenv("SESSION_VIDEO_DIR", str(tmp_path / "videos"))

    def fake_save(**kwargs):
        saved.append(kwargs)
        return 73

    monkeypatch.setattr(analysis, "save_analysis_to_athlete", fake_save)
    job_id = client.post(
        "/api/analysis/cycling",
        files=_video_file(),
    ).json()["job_id"]

    first = client.post(
        f"/api/analysis/cycling/{job_id}/save",
        json={"athlete_id": 7},
    )
    second = client.post(
        f"/api/analysis/cycling/{job_id}/save",
        json={"athlete_id": 7},
    )

    assert first.json() == {"session_id": 73}
    assert second.json() == {"session_id": 73}
    assert len(saved) == 1
    assert saved[0]["athlete_id"] == 7
    assert saved[0]["session_type"] == "cycling"
    assert saved[0]["analysis"]["cycling_analysis"]["analysis"]["cadence"] == 91.0
    assert Path(saved[0]["video_path"]).read_bytes() == b"cycling-video"
    assert registry.get(job_id).saved_session_id == 73


def test_cycling_routes_reject_other_job_kinds(monkeypatch, tmp_path):
    client, registry = _client(monkeypatch, tmp_path)
    running = registry.create("running")

    assert client.get(f"/api/analysis/cycling/{running.id}").status_code == 404
    assert (
        client.post(
            f"/api/analysis/cycling/{running.id}/save",
            json={"athlete_id": 7},
        ).status_code
        == 404
    )
