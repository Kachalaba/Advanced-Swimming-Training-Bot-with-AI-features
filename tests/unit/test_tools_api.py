"""API contract tests for working video tools."""

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.app.api import tools  # noqa: E402
from backend.app.services.jobs import JobRegistry  # noqa: E402
from backend.app.services.tools import SourceVideoInfo  # noqa: E402


class ImmediateThread:
    def __init__(self, target, args=(), kwargs=None, daemon=None):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}

    def start(self):
        self.target(*self.args, **self.kwargs)


def _client(monkeypatch, tmp_path):
    registry = JobRegistry(tmp_path / "jobs")
    monkeypatch.setattr(tools, "registry", registry)
    monkeypatch.setattr(tools, "Thread", ImmediateThread)
    monkeypatch.setattr(
        tools,
        "probe_video",
        lambda path: SourceVideoInfo(
            duration_sec=5.0,
            width=720,
            height=1280,
            has_audio=False,
        ),
    )

    def fake_trim(source, output, **kwargs):
        output.write_bytes(b"h264-output")
        return {
            "start_sec": kwargs["start_sec"],
            "end_sec": kwargs["end_sec"],
            "duration_sec": kwargs["end_sec"] - kwargs["start_sec"],
        }

    def fake_frames(source, output, **kwargs):
        output.write_bytes(b"zip-output")
        return {
            "mode": kwargs["mode"],
            "requested_value": kwargs["value"],
            "frame_count": 3,
        }

    monkeypatch.setattr(tools, "trim_video", fake_trim)
    monkeypatch.setattr(tools, "extract_frame_archive", fake_frames)
    app = FastAPI()
    app.include_router(tools.router, prefix="/api/tools")
    return TestClient(app), registry


def _video_file():
    return {"video": ("session.mov", b"video-bytes", "video/quicktime")}


def test_trim_job_completes_and_downloads_result(monkeypatch, tmp_path):
    client, registry = _client(monkeypatch, tmp_path)

    response = client.post(
        "/api/tools/trim",
        data={"start_sec": "1", "end_sec": "4"},
        files=_video_file(),
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    job = registry.get(job_id)
    assert job is not None
    assert job.status == "done"
    assert job.result["operation"] == "trim"
    assert job.result["media_type"] == "video/mp4"

    download = client.get(f"/api/tools/{job_id}/download")
    assert download.status_code == 200
    assert download.content == b"h264-output"
    assert download.headers["content-type"] == "video/mp4"
    assert "attachment" in download.headers["content-disposition"]


def test_frame_job_validates_mode_specific_fields(monkeypatch, tmp_path):
    client, _ = _client(monkeypatch, tmp_path)

    missing_interval = client.post(
        "/api/tools/frames",
        data={"mode": "interval"},
        files=_video_file(),
    )
    invalid_mode = client.post(
        "/api/tools/frames",
        data={"mode": "random", "frame_count": "3"},
        files=_video_file(),
    )

    assert missing_interval.status_code == 400
    assert "interval_sec" in missing_interval.json()["detail"]
    assert invalid_mode.status_code == 400
    assert "mode" in invalid_mode.json()["detail"].lower()


def test_frame_job_returns_zip_result(monkeypatch, tmp_path):
    client, registry = _client(monkeypatch, tmp_path)

    response = client.post(
        "/api/tools/frames",
        data={"mode": "count", "frame_count": "3"},
        files=_video_file(),
    )

    job = registry.get(response.json()["job_id"])
    assert job is not None
    assert job.result["operation"] == "frame_extractor"
    assert job.result["media_type"] == "application/zip"
    assert job.result["metadata"]["frame_count"] == 3


def test_unknown_and_unfinished_downloads_are_rejected(monkeypatch, tmp_path):
    client, registry = _client(monkeypatch, tmp_path)
    unfinished = registry.create("tool")

    assert client.get("/api/tools/missing/download").status_code == 404
    assert client.get(f"/api/tools/{unfinished.id}/download").status_code == 409


def test_tool_job_save_is_idempotent_and_persists_artifact(monkeypatch, tmp_path):
    client, registry = _client(monkeypatch, tmp_path)
    saved = {}
    persistent_root = tmp_path / "persistent"
    monkeypatch.setenv("SESSION_ARTIFACT_DIR", str(persistent_root))

    def fake_save(**kwargs):
        saved.update(kwargs)
        return 42

    monkeypatch.setattr(tools, "save_analysis_to_db", fake_save)
    response = client.post(
        "/api/tools/trim",
        data={"start_sec": "1", "end_sec": "4"},
        files=_video_file(),
    )
    job_id = response.json()["job_id"]

    first = client.post(
        f"/api/tools/{job_id}/save",
        json={"athlete_name": "Nikita K."},
    )
    second = client.post(
        f"/api/tools/{job_id}/save",
        json={"athlete_name": "Nikita K."},
    )

    assert first.json() == {"session_id": 42}
    assert second.json() == {"session_id": 42}
    assert saved["session_type"] == "tool"
    assert saved["analysis"]["tool"]["operation"] == "trim"
    saved_path = Path(saved["video_path"])
    assert saved_path.exists()
    assert saved_path.read_bytes() == b"h264-output"
    assert registry.get(job_id).saved_session_id == 42


def test_saved_tool_artifact_download_uses_session_record(monkeypatch, tmp_path):
    client, _ = _client(monkeypatch, tmp_path)
    artifact = tmp_path / "saved.zip"
    artifact.write_bytes(b"saved-zip")
    database = SimpleNamespace(
        get_session=lambda session_id: SimpleNamespace(
            id=session_id,
            session_type="tool",
            video_path=str(artifact),
            exercise_type="frame_extractor",
        )
    )
    monkeypatch.setattr(tools, "get_database", lambda: database)

    response = client.get("/api/tools/history/77/download")

    assert response.status_code == 200
    assert response.content == b"saved-zip"
    assert response.headers["content-type"] == "application/zip"


def test_saved_download_rejects_non_tool_and_missing_artifact(monkeypatch, tmp_path):
    client, _ = _client(monkeypatch, tmp_path)

    monkeypatch.setattr(
        tools,
        "get_database",
        lambda: SimpleNamespace(
            get_session=lambda session_id: SimpleNamespace(
                session_type="running",
                video_path="/tmp/run.mp4",
                exercise_type="",
            )
        ),
    )
    assert client.get("/api/tools/history/5/download").status_code == 404

    monkeypatch.setattr(
        tools,
        "get_database",
        lambda: SimpleNamespace(
            get_session=lambda session_id: SimpleNamespace(
                session_type="tool",
                video_path=str(tmp_path / "missing.zip"),
                exercise_type="frame_extractor",
            )
        ),
    )
    assert client.get("/api/tools/history/6/download").status_code == 404
