"""Browser-compatible video encoding helpers."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def open_intermediate_writer(
    output_path: Path,
    *,
    fps: float,
    frame_size: tuple[int, int],
) -> tuple[cv2.VideoWriter, Path]:
    """Open an MPEG-4 writer that FFmpeg can reliably transcode afterward."""
    source_path = output_path.with_name(f"{output_path.stem}.source{output_path.suffix}")
    source_path.unlink(missing_ok=True)
    writer = cv2.VideoWriter(
        str(source_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("Could not initialize intermediate video writer")
    return writer, source_path


def finalize_browser_video(source_path: Path, output_path: Path) -> None:
    """Transcode an intermediate video to H.264/yuv420p for web playback."""
    output_path.unlink(missing_ok=True)
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-an",
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
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not output_path.exists():
        output_path.unlink(missing_ok=True)
        detail = completed.stderr.strip() or "FFmpeg did not create an output file"
        raise RuntimeError(f"Could not encode browser-compatible video: {detail}")

    source_path.unlink(missing_ok=True)
    logger.info("Encoded browser-compatible video %s", output_path)
