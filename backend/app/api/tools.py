"""Working local video utility endpoints."""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
import shutil
from pathlib import Path
from threading import Thread
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from video_analysis.athlete_database import get_database, save_analysis_to_db

from ..services.jobs import Job, registry, stream_job_events
from ..services.tools import extract_frame_archive, probe_video, trim_video
from .upload_validation import UploadValidationError, copy_upload_with_limit, validate_video_upload

logger = logging.getLogger(__name__)
router = APIRouter(tags=["tools"])


class SaveToolRequest(BaseModel):
    athlete_name: str = Field(default="Nikita K.", min_length=1, max_length=120)


def _get_tool_job(job_id: str) -> Job:
    job = registry.get(job_id)
    if job is None or job.kind != "tool":
        raise HTTPException(status_code=404, detail="Tool job not found")
    return job


def _set_job_error(job: Job, exc: Exception) -> None:
    job.error = str(exc)
    job.push_event({"type": "error", "message": str(exc)})
    job.status = "error"


def _result_event(
    job: Job,
    *,
    operation: str,
    artifact_path: Path,
    media_type: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "result",
        "operation": operation,
        "artifact_name": artifact_path.name,
        "media_type": media_type,
        "size_bytes": artifact_path.stat().st_size,
        "metadata": metadata,
    }


def _run_trim_job(job: Job, source_path: Path, start_sec: float, end_sec: float) -> None:
    try:
        job.status = "running"
        job.push_event({"type": "progress", "pct": 10, "label": "Inspecting source video"})
        info = probe_video(source_path)
        job.push_event({"type": "progress", "pct": 30, "label": "Encoding H.264 clip"})
        artifact_path = job.workspace / f"trim-{job.id}.mp4"
        metadata = trim_video(
            source_path,
            artifact_path,
            start_sec=start_sec,
            end_sec=end_sec,
            info=info,
        )
        job.push_event({"type": "progress", "pct": 90, "label": "Finalizing clip"})
        event = _result_event(
            job,
            operation="trim",
            artifact_path=artifact_path,
            media_type="video/mp4",
            metadata=metadata,
        )
        job.artifact_path = artifact_path
        job.result = event
        job.push_event(event)
        job.status = "done"
    except Exception as exc:
        logger.exception("Trim tool job %s failed", job.id)
        _set_job_error(job, exc)


def _run_frame_job(job: Job, source_path: Path, mode: str, value: float) -> None:
    try:
        job.status = "running"
        job.push_event({"type": "progress", "pct": 10, "label": "Inspecting source video"})
        info = probe_video(source_path)
        job.push_event({"type": "progress", "pct": 30, "label": "Extracting JPEG frames"})
        artifact_path = job.workspace / f"frames-{job.id}.zip"
        metadata = extract_frame_archive(
            source_path,
            artifact_path,
            mode=mode,
            value=value,
            info=info,
            source_name=job.source_name or source_path.name,
        )
        job.push_event({"type": "progress", "pct": 90, "label": "Creating ZIP archive"})
        event = _result_event(
            job,
            operation="frame_extractor",
            artifact_path=artifact_path,
            media_type="application/zip",
            metadata=metadata,
        )
        job.artifact_path = artifact_path
        job.result = event
        job.push_event(event)
        job.status = "done"
    except Exception as exc:
        logger.exception("Frame tool job %s failed", job.id)
        _set_job_error(job, exc)


def _create_upload_job(video: UploadFile, operation: str) -> tuple[Job, Path]:
    try:
        suffix = validate_video_upload(
            filename=video.filename,
            content_type=video.content_type,
            declared_size=getattr(video, "size", None),
        )
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = registry.create(kind="tool")
    job.operation = operation
    job.source_name = Path(video.filename or f"source{suffix}").name
    source_path = job.workspace / f"input{suffix}"
    try:
        with source_path.open("wb") as target:
            copy_upload_with_limit(video.file, target)
    except UploadValidationError as exc:
        shutil.rmtree(job.workspace, ignore_errors=True)
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    return job, source_path


@router.post("/trim")
async def start_trim(
    video: UploadFile = File(...),
    start_sec: float = Form(...),
    end_sec: float = Form(...),
) -> dict[str, str]:
    if start_sec < 0:
        raise HTTPException(status_code=400, detail="start_sec cannot be negative")
    if end_sec <= start_sec:
        raise HTTPException(status_code=400, detail="end_sec must be greater than start_sec")
    job, source_path = _create_upload_job(video, "trim")
    Thread(
        target=_run_trim_job,
        args=(job, source_path, start_sec, end_sec),
        daemon=True,
    ).start()
    return {"job_id": job.id}


@router.post("/frames")
async def start_frames(
    video: UploadFile = File(...),
    mode: str = Form(...),
    interval_sec: Optional[float] = Form(None),
    frame_count: Optional[int] = Form(None),
) -> dict[str, str]:
    if mode == "interval":
        if interval_sec is None:
            raise HTTPException(status_code=400, detail="interval_sec is required for interval mode")
        value = float(interval_sec)
    elif mode == "count":
        if frame_count is None:
            raise HTTPException(status_code=400, detail="frame_count is required for count mode")
        value = float(frame_count)
    else:
        raise HTTPException(status_code=400, detail="mode must be 'interval' or 'count'")

    if value <= 0:
        raise HTTPException(status_code=400, detail="Extraction value must be positive")

    job, source_path = _create_upload_job(video, "frame_extractor")
    Thread(
        target=_run_frame_job,
        args=(job, source_path, mode, value),
        daemon=True,
    ).start()
    return {"job_id": job.id}


@router.get("/history/{session_id}/download")
async def download_saved_tool_artifact(session_id: int):
    session = get_database().get_session(session_id)
    if session is None or session.session_type != "tool" or not session.video_path:
        raise HTTPException(status_code=404, detail="Saved tool artifact not found")
    artifact_path = Path(session.video_path)
    if not artifact_path.is_file():
        raise HTTPException(status_code=404, detail="Saved tool artifact not found")
    media_type = mimetypes.guess_type(artifact_path.name)[0] or "application/octet-stream"
    return FileResponse(
        artifact_path,
        media_type=media_type,
        filename=artifact_path.name,
        content_disposition_type="attachment",
    )


@router.get("/{job_id}")
async def get_tool_job(job_id: str) -> dict[str, Any]:
    job = _get_tool_job(job_id)
    return {
        "id": job.id,
        "status": job.status,
        "result": job.result,
        "error": job.error,
        "event_count": len(job.events),
        "saved_session_id": job.saved_session_id,
    }


@router.get("/{job_id}/events")
async def stream_tool_events(job_id: str):
    job = _get_tool_job(job_id)
    return StreamingResponse(
        stream_job_events(job),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{job_id}/download")
async def download_tool_result(job_id: str):
    job = _get_tool_job(job_id)
    if job.status != "done" or job.artifact_path is None:
        raise HTTPException(status_code=409, detail="Tool artifact is not ready")
    if not job.artifact_path.is_file():
        raise HTTPException(status_code=404, detail="Tool artifact has expired")
    media_type = (job.result or {}).get("media_type", "application/octet-stream")
    return FileResponse(
        job.artifact_path,
        media_type=media_type,
        filename=(job.result or {}).get("artifact_name", job.artifact_path.name),
        content_disposition_type="attachment",
    )


def _persist_tool_artifact(job: Job) -> Path:
    if job.artifact_path is None or not job.artifact_path.is_file():
        raise HTTPException(status_code=409, detail="Tool artifact is not ready")
    artifact_root = Path(os.environ.get("SESSION_ARTIFACT_DIR", "/data/session-artifacts"))
    target_dir = artifact_root / "tools" / job.id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / job.artifact_path.name
    shutil.copy2(job.artifact_path, target)
    return target


@router.post("/{job_id}/save")
async def save_tool_result(job_id: str, request: SaveToolRequest) -> dict[str, int]:
    job = _get_tool_job(job_id)
    if job.status != "done" or job.result is None:
        raise HTTPException(status_code=409, detail="Tool result is not ready")
    if job.saved_session_id is None:
        persistent_path = _persist_tool_artifact(job)
        job.saved_session_id = await asyncio.to_thread(
            save_analysis_to_db,
            athlete_name=request.athlete_name,
            session_type="tool",
            analysis={
                "tool": {
                    "operation": job.result["operation"],
                    "artifact_name": job.result["artifact_name"],
                    "media_type": job.result["media_type"],
                    "size_bytes": job.result["size_bytes"],
                    "source_name": job.source_name,
                    "metadata": job.result["metadata"],
                }
            },
            video_path=str(persistent_path),
        )
    return {"session_id": job.saved_session_id}
