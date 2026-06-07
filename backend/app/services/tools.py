"""Local FFmpeg-backed video utility processing."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_EXTRACTED_FRAMES = 200


class ToolProcessingError(RuntimeError):
    """Raised when tool input or media processing is invalid."""


@dataclass(frozen=True)
class SourceVideoInfo:
    duration_sec: float
    width: int
    height: int
    has_audio: bool


def _run_media_command(command: list[str], error_prefix: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown media processing error"
        raise ToolProcessingError(f"{error_prefix}: {detail}")
    return completed


def probe_video(path: Path) -> SourceVideoInfo:
    """Inspect source duration, dimensions, and audio presence with FFprobe."""
    completed = _run_media_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,width,height",
            "-of",
            "json",
            str(path),
        ],
        "Could not inspect video",
    )
    try:
        payload = json.loads(completed.stdout)
        duration = float(payload["format"]["duration"])
        streams = payload.get("streams", [])
        video = next(stream for stream in streams if stream.get("codec_type") == "video")
        width = int(video["width"])
        height = int(video["height"])
    except (KeyError, TypeError, ValueError, StopIteration, json.JSONDecodeError) as exc:
        raise ToolProcessingError("Could not inspect video: incomplete FFprobe metadata") from exc

    if duration <= 0 or width <= 0 or height <= 0:
        raise ToolProcessingError("Could not inspect video: source metadata is invalid")

    return SourceVideoInfo(
        duration_sec=duration,
        width=width,
        height=height,
        has_audio=any(stream.get("codec_type") == "audio" for stream in streams),
    )


def interval_timestamps(duration_sec: float, interval_sec: float) -> list[float]:
    """Return timestamps at a fixed interval, strictly before source end."""
    if duration_sec <= 0:
        raise ToolProcessingError("Video duration must be positive")
    if interval_sec <= 0:
        raise ToolProcessingError("Frame interval must be positive")
    count = int(math.ceil(duration_sec / interval_sec))
    if count > MAX_EXTRACTED_FRAMES:
        raise ToolProcessingError(f"Frame extraction is limited to {MAX_EXTRACTED_FRAMES} images")
    return [round(index * interval_sec, 6) for index in range(count)]


def count_timestamps(duration_sec: float, frame_count: int) -> list[float]:
    """Return exactly frame_count timestamps evenly distributed before source end."""
    if duration_sec <= 0:
        raise ToolProcessingError("Video duration must be positive")
    if frame_count <= 0:
        raise ToolProcessingError("Frame count must be positive")
    if frame_count > MAX_EXTRACTED_FRAMES:
        raise ToolProcessingError(f"Frame extraction is limited to {MAX_EXTRACTED_FRAMES} images")
    return [round(index * duration_sec / frame_count, 6) for index in range(frame_count)]


def trim_video(
    source: Path,
    output: Path,
    *,
    start_sec: float,
    end_sec: float,
    info: SourceVideoInfo,
) -> dict[str, Any]:
    """Create a browser-compatible H.264 clip from the requested range."""
    if start_sec < 0:
        raise ToolProcessingError("Start time cannot be negative")
    if end_sec <= start_sec:
        raise ToolProcessingError("End time must be greater than start time")
    if end_sec > info.duration_sec + 0.001:
        raise ToolProcessingError(f"End time exceeds source duration ({info.duration_sec:.2f}s)")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    duration = end_sec - start_sec
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.6f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.6f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if info.has_audio:
        command.extend(["-c:a", "aac", "-b:a", "128k"])
    else:
        command.append("-an")
    command.append(str(output))

    _run_media_command(command, "Could not trim video")
    if not output.exists():
        raise ToolProcessingError("Could not trim video: FFmpeg did not create an output file")

    return {
        "start_sec": round(start_sec, 3),
        "end_sec": round(end_sec, 3),
        "duration_sec": round(duration, 3),
        "source_duration_sec": round(info.duration_sec, 3),
        "width": info.width,
        "height": info.height,
        "has_audio": info.has_audio,
    }


def extract_frame_archive(
    source: Path,
    output: Path,
    *,
    mode: str,
    value: float,
    info: SourceVideoInfo,
    source_name: str,
) -> dict[str, Any]:
    """Extract JPEG frames and package them with a manifest."""
    if mode == "interval":
        timestamps = interval_timestamps(info.duration_sec, float(value))
        requested_value: float | int = float(value)
    elif mode == "count":
        if not float(value).is_integer():
            raise ToolProcessingError("Frame count must be a whole number")
        requested_value = int(value)
        timestamps = count_timestamps(info.duration_sec, requested_value)
    else:
        raise ToolProcessingError("Frame extraction mode must be 'interval' or 'count'")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    manifest_frames = []

    with tempfile.TemporaryDirectory(prefix="sprint-tool-frames-", dir=str(output.parent)) as temp_dir:
        frame_dir = Path(temp_dir)
        for index, timestamp in enumerate(timestamps, start=1):
            filename = f"frame-{index:04d}.jpg"
            frame_path = frame_dir / filename
            _run_media_command(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{timestamp:.6f}",
                    "-i",
                    str(source),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(frame_path),
                ],
                f"Could not extract frame at {timestamp:.3f}s",
            )
            if not frame_path.exists():
                raise ToolProcessingError(f"Could not extract frame at {timestamp:.3f}s")
            manifest_frames.append(
                {
                    "filename": filename,
                    "timestamp_sec": timestamp,
                }
            )

        manifest = {
            "source_filename": Path(source_name).name,
            "source_duration_sec": round(info.duration_sec, 3),
            "mode": mode,
            "requested_value": requested_value,
            "actual_frame_count": len(manifest_frames),
            "timestamps_sec": timestamps,
            "frames": manifest_frames,
        }
        manifest_path = frame_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for frame in manifest_frames:
                archive.write(frame_dir / frame["filename"], frame["filename"])
            archive.write(manifest_path, "manifest.json")

    return {
        "mode": mode,
        "requested_value": requested_value,
        "frame_count": len(timestamps),
        "source_duration_sec": round(info.duration_sec, 3),
        "timestamps_sec": timestamps,
    }
