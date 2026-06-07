"""Tests for browser-compatible video finalization."""

from types import SimpleNamespace

import pytest

from backend.app.services.video_encoding import finalize_browser_video


def test_finalize_browser_video_removes_intermediate(monkeypatch, tmp_path):
    source = tmp_path / "annotated.source.mp4"
    output = tmp_path / "annotated.mp4"
    source.write_bytes(b"source")

    def fake_run(command, **kwargs):
        assert "libx264" in command
        assert "yuv420p" in command
        output.write_bytes(b"h264")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("backend.app.services.video_encoding.subprocess.run", fake_run)

    finalize_browser_video(source, output)

    assert output.read_bytes() == b"h264"
    assert not source.exists()


def test_finalize_browser_video_raises_on_ffmpeg_failure(monkeypatch, tmp_path):
    source = tmp_path / "annotated.source.mp4"
    output = tmp_path / "annotated.mp4"
    source.write_bytes(b"source")

    monkeypatch.setattr(
        "backend.app.services.video_encoding.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="encoder failed"),
    )

    with pytest.raises(RuntimeError, match="encoder failed"):
        finalize_browser_video(source, output)

    assert not output.exists()
