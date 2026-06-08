"""Web rehabilitation upload and live-camera endpoints."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from video_analysis.athlete_database import save_analysis_to_db
from video_analysis.constants import REHAB_PROTOCOLS

from ..services.jobs import Job
from ..services.jobs import registry as job_registry
from ..services.jobs import stream_job_events
from ..services.rehabilitation import analyze_rehabilitation_video
from ..services.rehabilitation import registry as live_registry
from .upload_validation import UploadValidationError, copy_upload_with_limit, validate_video_upload

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rehabilitation"])
MAX_LIVE_FRAME_BYTES = 2 * 1024 * 1024


class LiveSessionRequest(BaseModel):
    protocol: str = "shoulder_flexion"
    fps: float = Field(default=5.0, ge=1.0, le=15.0)


class SaveLiveSessionRequest(BaseModel):
    athlete_name: str = Field(default="Athlete", min_length=1, max_length=120)


def _validate_protocol(protocol: str) -> None:
    if protocol not in REHAB_PROTOCOLS:
        raise HTTPException(status_code=400, detail="Unknown rehabilitation protocol")


def _persist_job_video(job: Job) -> str:
    source = job.workspace / "annotated.mp4"
    if not source.exists():
        return ""
    default_dir = Path(os.environ.get("ATHLETE_DB_PATH", "data/athletes.db")).parent / "session-videos"
    video_dir = Path(os.environ.get("SESSION_VIDEO_DIR", str(default_dir)))
    video_dir.mkdir(parents=True, exist_ok=True)
    target = video_dir / f"rehab-{job.id}.mp4"
    shutil.copy2(source, target)
    return str(target)


@router.post("/rehabilitation/live")
async def create_live_session(request: LiveSessionRequest) -> dict:
    _validate_protocol(request.protocol)
    session = live_registry.create(request.protocol, request.fps)
    return {
        "session_id": session.id,
        "protocol": request.protocol,
        "analysis_fps": request.fps,
    }


@router.post("/rehabilitation/live/{session_id}/frame")
async def analyze_live_frame(
    session_id: str,
    image: UploadFile = File(...),
    calibrate: bool = Form(False),
) -> dict:
    session = live_registry.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Live session not found")
    if image.content_type not in {"image/jpeg", "image/webp", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Live frame must be JPEG or WebP")

    raw = await image.read(MAX_LIVE_FRAME_BYTES + 1)
    if len(raw) > MAX_LIVE_FRAME_BYTES:
        raise HTTPException(status_code=413, detail="Live frame is too large")
    encoded = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode live frame")
    try:
        return await asyncio.to_thread(
            session.process_frame,
            frame,
            calibrate,
        )
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/rehabilitation/live/{session_id}")
async def get_live_session(session_id: str) -> dict:
    session = live_registry.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Live session not found")
    return session.snapshot()


@router.post("/rehabilitation/live/{session_id}/save")
async def save_live_session(
    session_id: str,
    request: SaveLiveSessionRequest,
) -> dict[str, int]:
    session = live_registry.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Live session not found")
    report = session.snapshot().get("report")
    if not report:
        raise HTTPException(status_code=409, detail="Live report is not ready")
    saved_session_id = await asyncio.to_thread(
        save_analysis_to_db,
        athlete_name=request.athlete_name,
        session_type="rehab",
        analysis={"rehab_analysis": report},
        video_path="",
    )
    return {"session_id": saved_session_id}


@router.delete("/rehabilitation/live/{session_id}")
async def delete_live_session(session_id: str) -> dict[str, bool]:
    if not live_registry.delete(session_id):
        raise HTTPException(status_code=404, detail="Live session not found")
    return {"deleted": True}


def _run_upload_job(
    job: Job,
    video_path: Path,
    protocol: str,
    fps: float,
) -> None:
    try:
        job.status = "running"

        def on_progress(pct: int, label: str) -> None:
            job.push_event({"type": "progress", "pct": pct, "label": label})

        result = analyze_rehabilitation_video(
            video_path=video_path,
            output_dir=job.workspace,
            protocol=protocol,
            fps=fps,
            on_progress=on_progress,
        )
        event = {"type": "result", **result}
        job.result = event
        job.push_event(event)
        job.status = "done"
    except Exception as exc:
        logger.exception("Rehabilitation job %s failed", job.id)
        job.error = str(exc)
        job.push_event({"type": "error", "message": str(exc)})
        job.status = "error"


@router.post("/rehabilitation")
async def upload_rehabilitation(
    video: UploadFile = File(...),
    protocol: str = Form("shoulder_flexion"),
    fps: float = Form(15.0),
) -> dict[str, str]:
    _validate_protocol(protocol)
    try:
        suffix = validate_video_upload(
            filename=video.filename,
            content_type=video.content_type,
            declared_size=getattr(video, "size", None),
        )
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = job_registry.create(kind="rehabilitation")
    video_path = job.workspace / f"input{suffix}"
    try:
        with video_path.open("wb") as target:
            copy_upload_with_limit(video.file, target)
    except UploadValidationError as exc:
        shutil.rmtree(job.workspace, ignore_errors=True)
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    threading.Thread(
        target=_run_upload_job,
        args=(job, video_path, protocol, fps),
        daemon=True,
    ).start()
    return {"job_id": job.id}


@router.get("/rehabilitation/{job_id}")
async def get_rehabilitation_job(job_id: str) -> dict[str, Any]:
    job = job_registry.get(job_id)
    if job is None or job.kind != "rehabilitation":
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": job.id,
        "status": job.status,
        "result": job.result,
        "error": job.error,
        "event_count": len(job.events),
        "saved_session_id": job.saved_session_id,
    }


@router.post("/rehabilitation/{job_id}/save")
async def save_rehabilitation_job(
    job_id: str,
    request: SaveLiveSessionRequest,
) -> dict[str, int]:
    job = job_registry.get(job_id)
    if job is None or job.kind != "rehabilitation":
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done" or not job.result:
        raise HTTPException(status_code=409, detail="Rehabilitation report is not ready")
    if job.saved_session_id is None:
        report = job.result.get("report")
        if not report:
            raise HTTPException(status_code=409, detail="Rehabilitation report is not ready")
        job.saved_session_id = await asyncio.to_thread(
            save_analysis_to_db,
            athlete_name=request.athlete_name,
            session_type="rehab",
            analysis={"rehab_analysis": report},
            video_path=_persist_job_video(job),
        )
    return {"session_id": job.saved_session_id}


@router.get("/rehabilitation/{job_id}/events")
async def stream_rehabilitation_events(job_id: str):
    job = job_registry.get(job_id)
    if job is None or job.kind != "rehabilitation":
        raise HTTPException(status_code=404, detail="Job not found")

    return StreamingResponse(
        stream_job_events(job),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/rehabilitation/{job_id}/video")
async def stream_rehabilitation_video(job_id: str):
    job = job_registry.get(job_id)
    if job is None or job.kind != "rehabilitation":
        raise HTTPException(status_code=404, detail="Job not found")
    video_path = job.workspace / "annotated.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Annotated video not ready")
    return FileResponse(video_path, media_type="video/mp4")
