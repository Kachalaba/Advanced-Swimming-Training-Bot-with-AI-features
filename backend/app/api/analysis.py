"""Analysis endpoints (upload → progress stream → result + video)."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import threading
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from app.api.upload_validation import UploadValidationError, copy_upload_with_limit, validate_video_upload
from app.services import running as running_pipeline
from app.services.jobs import Job, registry

logger = logging.getLogger(__name__)

router = APIRouter(tags=["analysis"])


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
        job.status = "error"
        job.error = str(exc)
        job.push_event({"type": "error", "message": str(exc)})


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

    thread = threading.Thread(
        target=_run_pipeline_in_thread,
        args=(job, video_path, fps),
        daemon=True,
    )
    thread.start()
    return {"job_id": job.id}


@router.get("/running/{job_id}")
async def get_job_status(job_id: str) -> dict:
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return {
        "id": job.id,
        "status": job.status,
        "result": job.result,
        "error": job.error,
        "event_count": len(job.events),
    }


@router.get("/running/{job_id}/events")
async def stream_events(job_id: str):
    """Server-Sent Events stream of progress updates and final result."""
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    async def event_generator():
        # Replay any events that arrived before this client connected
        for ev in list(job.events):
            yield f"data: {json.dumps(ev)}\n\n"
            if ev["type"] in ("result", "error"):
                return

        # Then stream new ones until the job is finished
        while True:
            try:
                ev = await asyncio.wait_for(job.queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # Heartbeat so proxies don't close the connection
                yield ": heartbeat\n\n"
                if job.status in ("done", "error"):
                    return
                continue
            yield f"data: {json.dumps(ev)}\n\n"
            if ev["type"] in ("result", "error"):
                return

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/running/{job_id}/video")
async def stream_video(job_id: str):
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    video_path = job.workspace / "annotated.mp4"
    if not video_path.exists():
        raise HTTPException(404, "Annotated video not ready")
    return FileResponse(video_path, media_type="video/mp4")
