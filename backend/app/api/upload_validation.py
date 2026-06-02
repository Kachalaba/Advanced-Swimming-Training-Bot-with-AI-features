"""Validation helpers for video upload endpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO

MAX_VIDEO_UPLOAD_BYTES = int(os.environ.get("MAX_VIDEO_UPLOAD_MB", "512")) * 1024 * 1024
SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}
SUPPORTED_VIDEO_CONTENT_TYPES = {
    "application/octet-stream",
    "video/mp4",
    "video/quicktime",
    "video/x-matroska",
    "video/x-msvideo",
}
CHUNK_SIZE_BYTES = 1024 * 1024


class UploadValidationError(ValueError):
    """Raised when an uploaded video fails validation."""


def validate_video_upload(
    filename: str | None,
    content_type: str | None,
    declared_size: int | None,
) -> str:
    """Validate upload metadata and return a normalized file suffix."""
    suffix = Path(filename or "video.mp4").suffix.lower() or ".mp4"
    if suffix not in SUPPORTED_VIDEO_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_VIDEO_SUFFIXES))
        raise UploadValidationError(f"Unsupported video format '{suffix}'. Allowed: {allowed}")

    if content_type and content_type.lower() not in SUPPORTED_VIDEO_CONTENT_TYPES:
        raise UploadValidationError(f"Unsupported content type '{content_type}'")

    if declared_size is not None and declared_size > MAX_VIDEO_UPLOAD_BYTES:
        limit_mb = MAX_VIDEO_UPLOAD_BYTES // (1024 * 1024)
        raise UploadValidationError(f"Uploaded video is too large. Limit is {limit_mb} MB")

    return suffix


def copy_upload_with_limit(
    source: BinaryIO,
    target: BinaryIO,
    max_bytes: int = MAX_VIDEO_UPLOAD_BYTES,
) -> int:
    """Copy uploaded bytes while enforcing a hard size limit."""
    total = 0
    while True:
        chunk = source.read(CHUNK_SIZE_BYTES)
        if not chunk:
            return total
        total += len(chunk)
        if total > max_bytes:
            limit_mb = max_bytes // (1024 * 1024)
            raise UploadValidationError(f"Uploaded video is too large. Limit is {limit_mb} MB")
        target.write(chunk)
