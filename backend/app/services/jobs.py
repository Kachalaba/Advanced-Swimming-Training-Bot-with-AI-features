"""In-memory job registry for analysis jobs.

Single-process / single-server only. When we move to a real worker
queue this whole module gets replaced; until then it keeps the API
surface tiny and lets the frontend poll or subscribe to events.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    kind: str  # "running" for now
    workspace: Path
    status: str = "queued"  # queued | running | done | error
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    queue: asyncio.Queue[dict[str, Any]] = field(
        default_factory=lambda: asyncio.Queue()
    )
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def push_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            self.events.append(event)
        # asyncio.Queue.put is async, but we're called from a worker
        # thread. Use put_nowait — this queue is unbounded so it never
        # blocks. The async listener wraps await for next event.
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            pass


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
