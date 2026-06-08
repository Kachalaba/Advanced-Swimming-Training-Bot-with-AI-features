"""Unit tests for local video tool processing."""

import json
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.app.services.tools import (
    MAX_EXTRACTED_FRAMES,
    SourceVideoInfo,
    ToolProcessingError,
    count_timestamps,
    extract_frame_archive,
    interval_timestamps,
    probe_video,
    trim_video,
)


def test_interval_timestamps_stop_before_duration():
    assert interval_timestamps(5.0, 2.0) == [0.0, 2.0, 4.0]


def test_count_timestamps_are_evenly_spaced_before_end():
    assert count_timestamps(5.0, 1) == [0.0]
    assert count_timestamps(5.0, 5) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_timestamp_validation_rejects_invalid_and_excessive_requests():
    with pytest.raises(ToolProcessingError, match="positive"):
        interval_timestamps(5.0, 0.0)
    with pytest.raises(ToolProcessingError, match="positive"):
        count_timestamps(5.0, 0)
    with pytest.raises(ToolProcessingError, match=str(MAX_EXTRACTED_FRAMES)):
        interval_timestamps(1000.0, 1.0)


def test_probe_video_parses_ffprobe_json(monkeypatch, tmp_path):
    payload = {
        "format": {"duration": "5.25"},
        "streams": [
            {"codec_type": "video", "width": 720, "height": 1280},
            {"codec_type": "audio"},
        ],
    }
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr("backend.app.services.tools.subprocess.run", fake_run)

    info = probe_video(tmp_path / "input.mov")

    assert info == SourceVideoInfo(
        duration_sec=5.25,
        width=720,
        height=1280,
        has_audio=True,
    )
    assert calls[0][0][0] == "ffprobe"
    assert calls[0][1]["shell"] is False


def test_probe_video_rejects_unreadable_source(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "backend.app.services.tools.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="invalid data",
        ),
    )

    with pytest.raises(ToolProcessingError, match="Could not inspect video"):
        probe_video(tmp_path / "broken.mov")


def test_trim_video_builds_browser_compatible_ffmpeg_command(monkeypatch, tmp_path):
    source = tmp_path / "input.mov"
    output = tmp_path / "trim.mp4"
    source.write_bytes(b"source")
    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        output.write_bytes(b"h264")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("backend.app.services.tools.subprocess.run", fake_run)

    metadata = trim_video(
        source,
        output,
        start_sec=1.0,
        end_sec=4.0,
        info=SourceVideoInfo(5.0, 720, 1280, True),
    )

    command, kwargs = commands[0]
    assert command[0] == "ffmpeg"
    assert ["-c:v", "libx264"] == command[command.index("-c:v") : command.index("-c:v") + 2]
    assert "yuv420p" in command
    assert "+faststart" in command
    assert ["-c:a", "aac"] == command[command.index("-c:a") : command.index("-c:a") + 2]
    assert kwargs["shell"] is False
    assert metadata["duration_sec"] == 3.0
    assert output.exists()


def test_extract_frame_archive_contains_images_and_manifest(monkeypatch, tmp_path):
    source = tmp_path / "input.mov"
    output = tmp_path / "frames.zip"
    source.write_bytes(b"source")

    def fake_run(command, **kwargs):
        frame_path = Path(command[-1])
        frame_path.write_bytes(f"jpeg-{frame_path.stem}".encode())
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("backend.app.services.tools.subprocess.run", fake_run)

    metadata = extract_frame_archive(
        source,
        output,
        mode="count",
        value=3,
        info=SourceVideoInfo(6.0, 720, 1280, False),
        source_name="session.mov",
    )

    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == [
            "frame-0001.jpg",
            "frame-0002.jpg",
            "frame-0003.jpg",
            "manifest.json",
        ]
        manifest = json.loads(archive.read("manifest.json"))

    assert manifest["source_filename"] == "session.mov"
    assert manifest["timestamps_sec"] == [0.0, 2.0, 4.0]
    assert manifest["actual_frame_count"] == 3
    assert metadata["frame_count"] == 3


def test_processing_failure_raises_stable_error(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "backend.app.services.tools.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            1,
            "",
            "encoder failed",
        ),
    )

    with pytest.raises(ToolProcessingError, match="encoder failed"):
        trim_video(
            tmp_path / "input.mov",
            tmp_path / "trim.mp4",
            start_sec=0.0,
            end_sec=1.0,
            info=SourceVideoInfo(2.0, 320, 240, False),
        )
