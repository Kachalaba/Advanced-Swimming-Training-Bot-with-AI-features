"""In-memory job registry for analysis jobs.

Single-process / single-server only. When we move to a real worker
queue this whole module gets replaced; until then it keeps the API
surface tiny and lets the frontend poll or subscribe to events.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    kind: str  # running | swimming | rehabilitation | tool
    workspace: Path
    status: str = "queued"  # queued | running | done | error
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    saved_session_id: int | None = None
    artifact_path: Path | None = None
    source_name: str | None = None
    operation: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    save_lock: threading.Lock = field(default_factory=threading.Lock)

    def push_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            self.events.append(event)

    def events_since(self, index: int) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.events[index:])


async def stream_job_events(
    job: Job,
    *,
    heartbeat_seconds: float = 30.0,
    poll_seconds: float = 0.1,
):
    """Yield each job event exactly once for a single SSE subscriber."""
    cursor = 0
    last_emit = time.monotonic()
    while True:
        events = job.events_since(cursor)
        for event in events:
            cursor += 1
            last_emit = time.monotonic()
            yield f"data: {json.dumps(event)}\n\n"
            if event["type"] in ("result", "error"):
                return

        if job.status in ("done", "error"):
            return

        now = time.monotonic()
        if now - last_emit >= heartbeat_seconds:
            yield ": heartbeat\n\n"
            last_emit = now
        await asyncio.sleep(poll_seconds)


class JobRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, kind: str) -> Job:
        job_id = uuid.uuid4().hex[:12]
        workspace = self.root / job_id
        workspace.mkdir(parents=True, exist_ok=True)
        job = Job(id=job_id, kind=kind, workspace=workspace)
        with self._lock:
            self._jobs[job_id] = job
        logger.info("Created job %s (kind=%s)", job_id, kind)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)


# Single shared registry instance keyed off /tmp/sprint-ai-jobs.
registry = JobRegistry(Path("/tmp/sprint-ai-jobs"))
