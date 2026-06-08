"""API contract tests for side-view freestyle analysis."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.app.api import analysis  # noqa: E402
from backend.app.services.jobs import JobRegistry  # noqa: E402
from backend.app.services.swimming import ProgressEvent, ResultEvent  # noqa: E402


class ImmediateThread:
    def __init__(self, target, args=(), kwargs=None, daemon=None):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}

    def start(self):
        self.target(*self.args, **self.kwargs)


def _result_payload():
    return {
        "analysis_type": "swimming_freestyle_side",
        "contract_version": "1.0",
        "quality": {"status": "pass", "warnings": []},
        "coverage": {"available_zones": 5, "total_zones": 5},
        "overall_score": 82.0,
        "cycles": [],
        "zones": [],
        "primary_issue": None,
        "prescription": None,
        "frames_total": 10,
        "frames_with_pose": 9,
        "video_path": "annotated.mp4",
    }


def _client(monkeypatch, tmp_path):
    registry = JobRegistry(tmp_path / "jobs")
    monkeypatch.setattr(analysis, "registry", registry)
    monkeypatch.setattr(analysis, "Thread", ImmediateThread)

    def fake_analyze(video_path, output_dir, fps=None):
        (output_dir / "annotated.mp4").write_bytes(b"h264-video")
        yield ProgressEvent("quality_gate", 5, "Checking video quality")
        yield ProgressEvent("completed", 100, "Analysis complete")
        yield ResultEvent(_result_payload())

    monkeypatch.setattr(analysis.swimming_pipeline, "analyze_swimming_video", fake_analyze)
    app = FastAPI()
    app.include_router(analysis.router, prefix="/api/analysis")
    return TestClient(app), registry


def _video_file():
    return {"video": ("freestyle.mp4", b"video-bytes", "video/mp4")}


def test_swimming_upload_starts_and_completes_job(monkeypatch, tmp_path):
    client, registry = _client(monkeypatch, tmp_path)

    response = client.post("/api/analysis/swimming", files=_video_file())

    assert response.status_code == 200
    job = registry.get(response.json()["job_id"])
    assert job is not None
    assert job.kind == "swimming"
    assert job.status == "done"
    assert job.result["analysis_type"] == "swimming_freestyle_side"


def test_swimming_status_rejects_other_job_kinds(monkeypatch, tmp_path):
    client, registry = _client(monkeypatch, tmp_path)
    running = registry.create("running")

    response = client.get(f"/api/analysis/swimming/{running.id}")

    assert response.status_code == 404


def test_swimming_events_and_video_are_available_after_completion(monkeypatch, tmp_path):
    client, _ = _client(monkeypatch, tmp_path)
    job_id = client.post("/api/analysis/swimming", files=_video_file()).json()["job_id"]

    events = client.get(f"/api/analysis/swimming/{job_id}/events")
    video = client.get(f"/api/analysis/swimming/{job_id}/video")

    assert events.status_code == 200
    assert '"stage": "quality_gate"' in events.text
    assert '"type": "result"' in events.text
    assert video.status_code == 200
    assert video.content == b"h264-video"


def test_swimming_video_returns_404_before_render(monkeypatch, tmp_path):
    client, registry = _client(monkeypatch, tmp_path)
    queued = registry.create("swimming")

    response = client.get(f"/api/analysis/swimming/{queued.id}/video")

    assert response.status_code == 404


def test_swimming_save_is_idempotent(monkeypatch, tmp_path):
    client, registry = _client(monkeypatch, tmp_path)
    saved = {}
    persistent_dir = tmp_path / "persistent-videos"
    monkeypatch.setenv("SESSION_VIDEO_DIR", str(persistent_dir))

    def fake_save(**kwargs):
        saved.update(kwargs)
        return 91

    monkeypatch.setattr(analysis, "save_analysis_to_db", fake_save)
    job_id = client.post("/api/analysis/swimming", files=_video_file()).json()["job_id"]

    first = client.post(
        f"/api/analysis/swimming/{job_id}/save",
        json={"athlete_name": "Nikita K."},
    )
    second = client.post(
        f"/api/analysis/swimming/{job_id}/save",
        json={"athlete_name": "Nikita K."},
    )

    assert first.json() == {"session_id": 91}
    assert second.json() == {"session_id": 91}
    assert saved["session_type"] == "swimming"
    assert saved["analysis"]["swimming_analysis"]["analysis_type"] == "swimming_freestyle_side"
    assert Path(saved["video_path"]).read_bytes() == b"h264-video"
    assert registry.get(job_id).saved_session_id == 91
