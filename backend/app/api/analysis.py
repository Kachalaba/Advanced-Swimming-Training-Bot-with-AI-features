"""Analysis endpoints (upload → progress stream → result + video)."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from threading import Thread
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from video_analysis.athlete_database import save_analysis_to_athlete, save_analysis_to_db
from video_analysis.constants import DRYLAND_SUPPORTED_EXERCISES

from ..services import cycling as cycling_pipeline
from ..services import dryland as dryland_pipeline
from ..services import running as running_pipeline
from ..services import swimming as swimming_pipeline
from ..services.jobs import Job, registry, stream_job_events
from .upload_validation import UploadValidationError, copy_upload_with_limit, validate_video_upload

logger = logging.getLogger(__name__)

router = APIRouter(tags=["analysis"])


class SaveAnalysisRequest(BaseModel):
    athlete_id: Optional[int] = Field(default=None, ge=1)
    athlete_name: str = Field(default="Athlete", min_length=1, max_length=120)


def _run_pipeline_in_thread(job: Job, video_path: Path, fps: float) -> None:
    """Worker function that drives the generator and pushes events to the job."""
    try:
        job.status = "running"
        for event in running_pipeline.analyze_running_video(
            video_path=video_path,
            output_dir=job.workspace,
            fps=fps,
        ):
            ev = event.to_dict()
            job.push_event(ev)
            if ev["type"] == "result":
                job.result = ev
                job.status = "done"
            elif ev["type"] == "error":
                job.error = ev["message"]
                job.status = "error"
        if job.status == "running":
            job.status = "done"
    except Exception as exc:
        logger.exception("Job %s failed", job.id)
        job.error = str(exc)
        job.push_event({"type": "error", "message": str(exc)})
        job.status = "error"


def _run_swimming_pipeline_in_thread(
    job: Job,
    video_path: Path,
    fps: Optional[float],
) -> None:
    try:
        job.status = "running"
        for event in swimming_pipeline.analyze_swimming_video(
            video_path=video_path,
            output_dir=job.workspace,
            fps=fps,
        ):
            payload = event.to_dict()
            job.push_event(payload)
            if payload["type"] == "result":
                job.result = payload
                job.status = "done"
            elif payload["type"] == "error":
                job.result = payload
                job.error = payload["message"]
                job.status = "error"
        if job.status == "running":
            job.status = "done"
    except Exception as exc:
        logger.exception("Swimming job %s failed", job.id)
        payload = {
            "type": "error",
            "code": "internal_error",
            "message": str(exc),
            "reshoot_guidance": "",
        }
        job.result = payload
        job.error = str(exc)
        job.push_event(payload)
        job.status = "error"


def _run_cycling_pipeline_in_thread(
    job: Job,
    video_path: Path,
    fps: float,
) -> None:
    try:
        job.status = "running"
        for event in cycling_pipeline.analyze_cycling_video(
            video_path=video_path,
            output_dir=job.workspace,
            fps=fps,
        ):
            payload = event.to_dict()
            job.push_event(payload)
            if payload["type"] == "result":
                job.result = payload
                job.status = "done"
            elif payload["type"] == "error":
                job.error = payload["message"]
                job.status = "error"
        if job.status == "running":
            job.status = "done"
    except Exception as exc:
        logger.exception("Cycling job %s failed", job.id)
        job.error = str(exc)
        job.push_event({"type": "error", "message": str(exc)})
        job.status = "error"


def _run_dryland_pipeline_in_thread(
    job: Job,
    video_path: Path,
    exercise_type: str,
    fps: float,
) -> None:
    try:
        job.status = "running"
        for event in dryland_pipeline.analyze_dryland_video(
            video_path=video_path,
            output_dir=job.workspace,
            exercise_type=exercise_type,
            fps=fps,
        ):
            payload = event.to_dict()
            job.push_event(payload)
            if payload["type"] == "result":
                job.result = payload
                job.status = "done"
            elif payload["type"] == "error":
                job.error = payload["message"]
                job.status = "error"
        if job.status == "running":
            job.status = "done"
    except Exception as exc:
        logger.exception("Dryland job %s failed", job.id)
        job.error = str(exc)
        job.push_event({"type": "error", "message": str(exc)})
        job.status = "error"


def _get_swimming_job(job_id: str) -> Job:
    job = registry.get(job_id)
    if job is None or job.kind != "swimming":
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _get_running_job(job_id: str) -> Job:
    job = registry.get(job_id)
    if job is None or job.kind != "running":
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _get_cycling_job(job_id: str) -> Job:
    job = registry.get(job_id)
    if job is None or job.kind != "cycling":
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _get_dryland_job(job_id: str) -> Job:
    job = registry.get(job_id)
    if job is None or job.kind != "dryland":
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _persist_sport_video(job: Job, sport: str) -> str:
    source = job.workspace / "annotated.mp4"
    if not source.exists():
        return ""
    default_dir = Path(os.environ.get("ATHLETE_DB_PATH", "data/athletes.db")).parent / "session-videos"
    video_dir = Path(os.environ.get("SESSION_VIDEO_DIR", str(default_dir)))
    video_dir.mkdir(parents=True, exist_ok=True)
    target = video_dir / f"{sport}-{job.id}.mp4"
    shutil.copy2(source, target)
    return str(target)


def _save_completed_sport_job(
    job: Job,
    request: SaveAnalysisRequest,
    sport: str,
) -> int:
    """Persist a completed job exactly once, including concurrent requests."""

    with job.save_lock:
        if job.saved_session_id is not None:
            return job.saved_session_id

        video_path = ""
        try:
            video_path = _persist_sport_video(job, sport)
            kwargs = {
                "session_type": sport,
                "analysis": {f"{sport}_analysis": job.result},
                "video_path": video_path,
            }
            if request.athlete_id is not None:
                job.saved_session_id = save_analysis_to_athlete(
                    athlete_id=request.athlete_id,
                    **kwargs,
                )
            else:
                job.saved_session_id = save_analysis_to_db(
                    athlete_name=request.athlete_name,
                    **kwargs,
                )
        except Exception:
            if video_path:
                Path(video_path).unlink(missing_ok=True)
            raise

        return job.saved_session_id


@router.post("/running")
async def upload_running(
    video: UploadFile = File(...),
    fps: float = Form(30.0),
) -> dict[str, str]:
    """Accept a video upload, kick off analysis, return a job id."""
    try:
        suffix = validate_video_upload(
            filename=video.filename,
            content_type=video.content_type,
            declared_size=getattr(video, "size", None),
        )
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = registry.create(kind="running")
    video_path = job.workspace / f"input{suffix}"
    try:
        with video_path.open("wb") as f:
            bytes_written = copy_upload_with_limit(video.file, f)
    except UploadValidationError as exc:
        shutil.rmtree(job.workspace, ignore_errors=True)
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    logger.info("Saved upload %s (%d bytes)", video_path, video_path.stat().st_size)
    logger.debug("Copied %d upload bytes for job %s", bytes_written, job.id)

    thread = Thread(
        target=_run_pipeline_in_thread,
        args=(job, video_path, fps),
        daemon=True,
    )
    thread.start()
    return {"job_id": job.id}


@router.get("/running/{job_id}")
async def get_job_status(job_id: str) -> dict:
    job = _get_running_job(job_id)
    return {
        "id": job.id,
        "status": job.status,
        "result": job.result,
        "error": job.error,
        "event_count": len(job.events),
        "saved_session_id": job.saved_session_id,
    }


@router.get("/running/{job_id}/events")
async def stream_events(job_id: str):
    """Server-Sent Events stream of progress updates and final result."""
    job = _get_running_job(job_id)

    return StreamingResponse(
        stream_job_events(job),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/running/{job_id}/video")
async def stream_video(job_id: str):
    job = _get_running_job(job_id)
    video_path = job.workspace / "annotated.mp4"
    if not video_path.exists():
        raise HTTPException(404, "Annotated video not ready")
    return FileResponse(video_path, media_type="video/mp4")


@router.post("/running/{job_id}/save")
async def save_running_job(
    job_id: str,
    request: SaveAnalysisRequest,
) -> dict[str, int]:
    job = _get_running_job(job_id)
    if job.status != "done" or not job.result:
        raise HTTPException(status_code=409, detail="Running analysis is not ready")
    try:
        session_id = await asyncio.to_thread(
            _save_completed_sport_job,
            job,
            request,
            "running",
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"session_id": session_id}


@router.post("/cycling")
async def upload_cycling(
    video: UploadFile = File(...),
    fps: float = Form(30.0),
) -> dict[str, str]:
    try:
        suffix = validate_video_upload(
            filename=video.filename,
            content_type=video.content_type,
            declared_size=getattr(video, "size", None),
        )
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = registry.create(kind="cycling")
    job.source_name = video.filename
    video_path = job.workspace / f"input{suffix}"
    try:
        with video_path.open("wb") as target:
            copy_upload_with_limit(video.file, target)
    except UploadValidationError as exc:
        shutil.rmtree(job.workspace, ignore_errors=True)
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    Thread(
        target=_run_cycling_pipeline_in_thread,
        args=(job, video_path, fps),
        daemon=True,
    ).start()
    return {"job_id": job.id}


@router.get("/cycling/{job_id}")
async def get_cycling_job(job_id: str) -> dict:
    job = _get_cycling_job(job_id)
    return {
        "id": job.id,
        "status": job.status,
        "result": job.result,
        "error": job.error,
        "event_count": len(job.events),
        "saved_session_id": job.saved_session_id,
    }


@router.get("/cycling/{job_id}/events")
async def stream_cycling_events(job_id: str):
    job = _get_cycling_job(job_id)
    return StreamingResponse(
        stream_job_events(job),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/cycling/{job_id}/video")
async def stream_cycling_video(job_id: str):
    job = _get_cycling_job(job_id)
    video_path = job.workspace / "annotated.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Annotated video not ready")
    return FileResponse(video_path, media_type="video/mp4")


@router.post("/cycling/{job_id}/save")
async def save_cycling_job(
    job_id: str,
    request: SaveAnalysisRequest,
) -> dict[str, int]:
    job = _get_cycling_job(job_id)
    if job.status != "done" or not job.result:
        raise HTTPException(status_code=409, detail="Cycling analysis is not ready")
    try:
        session_id = await asyncio.to_thread(
            _save_completed_sport_job,
            job,
            request,
            "cycling",
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"session_id": session_id}


@router.post("/dryland")
async def upload_dryland(
    video: UploadFile = File(...),
    exercise_type: str = Form(...),
    fps: float = Form(15.0),
) -> dict[str, str]:
    if exercise_type not in DRYLAND_SUPPORTED_EXERCISES:
        raise HTTPException(status_code=400, detail=f"Unsupported dryland exercise: {exercise_type}")
    try:
        suffix = validate_video_upload(
            filename=video.filename,
            content_type=video.content_type,
            declared_size=getattr(video, "size", None),
        )
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = registry.create(kind="dryland")
    job.source_name = video.filename
    video_path = job.workspace / f"input{suffix}"
    try:
        with video_path.open("wb") as target:
            copy_upload_with_limit(video.file, target)
    except UploadValidationError as exc:
        shutil.rmtree(job.workspace, ignore_errors=True)
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    Thread(
        target=_run_dryland_pipeline_in_thread,
        args=(job, video_path, exercise_type, fps),
        daemon=True,
    ).start()
    return {"job_id": job.id}


@router.get("/dryland/{job_id}")
async def get_dryland_job(job_id: str) -> dict:
    job = _get_dryland_job(job_id)
    return {
        "id": job.id,
        "status": job.status,
        "result": job.result,
        "error": job.error,
        "event_count": len(job.events),
        "saved_session_id": job.saved_session_id,
    }


@router.get("/dryland/{job_id}/events")
async def stream_dryland_events(job_id: str):
    job = _get_dryland_job(job_id)
    return StreamingResponse(
        stream_job_events(job),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/dryland/{job_id}/video")
async def stream_dryland_video(job_id: str):
    job = _get_dryland_job(job_id)
    video_path = job.workspace / "annotated.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Annotated video not ready")
    return FileResponse(video_path, media_type="video/mp4")


@router.post("/dryland/{job_id}/save")
async def save_dryland_job(
    job_id: str,
    request: SaveAnalysisRequest,
) -> dict[str, int]:
    job = _get_dryland_job(job_id)
    if job.status != "done" or not job.result:
        raise HTTPException(status_code=409, detail="Dryland analysis is not ready")
    try:
        session_id = await asyncio.to_thread(
            _save_completed_sport_job,
            job,
            request,
            "dryland",
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"session_id": session_id}


@router.post("/swimming")
async def upload_swimming(
    video: UploadFile = File(...),
    fps: Optional[float] = Form(None),
) -> dict[str, str]:
    try:
        suffix = validate_video_upload(
            filename=video.filename,
            content_type=video.content_type,
            declared_size=getattr(video, "size", None),
        )
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = registry.create(kind="swimming")
    job.source_name = video.filename
    video_path = job.workspace / f"input{suffix}"
    try:
        with video_path.open("wb") as target:
            copy_upload_with_limit(video.file, target)
    except UploadValidationError as exc:
        shutil.rmtree(job.workspace, ignore_errors=True)
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    Thread(
        target=_run_swimming_pipeline_in_thread,
        args=(job, video_path, fps),
        daemon=True,
    ).start()
    return {"job_id": job.id}


@router.get("/swimming/{job_id}")
async def get_swimming_job(job_id: str) -> dict:
    job = _get_swimming_job(job_id)
    return {
        "id": job.id,
        "status": job.status,
        "result": job.result,
        "error": job.error,
        "event_count": len(job.events),
        "saved_session_id": job.saved_session_id,
    }


@router.get("/swimming/{job_id}/events")
async def stream_swimming_events(job_id: str):
    job = _get_swimming_job(job_id)
    return StreamingResponse(
        stream_job_events(job),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/swimming/{job_id}/video")
async def stream_swimming_video(job_id: str):
    job = _get_swimming_job(job_id)
    video_path = job.workspace / "annotated.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Annotated video not ready")
    return FileResponse(video_path, media_type="video/mp4")


@router.post("/swimming/{job_id}/save")
async def save_swimming_job(
    job_id: str,
    request: SaveAnalysisRequest,
) -> dict[str, int]:
    job = _get_swimming_job(job_id)
    if job.status != "done" or not job.result:
        raise HTTPException(status_code=409, detail="Swimming analysis is not ready")
    try:
        session_id = await asyncio.to_thread(
            _save_completed_sport_job,
            job,
            request,
            "swimming",
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"session_id": session_id}
