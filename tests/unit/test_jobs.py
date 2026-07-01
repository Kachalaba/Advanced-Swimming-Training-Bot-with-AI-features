"""Tests for the in-memory analysis job event stream."""

import asyncio
import json
import os
import time

from backend.app.services.jobs import Job, stream_job_events


def test_stream_replays_each_event_exactly_once(tmp_path):
    job = Job(id="job-1", kind="running", workspace=tmp_path)
    job.push_event({"type": "progress", "pct": 10, "label": "Started"})
    job.push_event({"type": "progress", "pct": 50, "label": "Halfway"})
    job.push_event({"type": "result", "analysis": {}})
    job.status = "done"

    async def collect():
        return [
            chunk
            async for chunk in stream_job_events(
                job,
                heartbeat_seconds=1.0,
                poll_seconds=0.001,
            )
        ]

    chunks = asyncio.run(collect())
    events = [json.loads(chunk.removeprefix("data: ").strip()) for chunk in chunks]

    assert [event["type"] for event in events] == ["progress", "progress", "result"]
    assert len(events) == len(job.events)


def test_stream_includes_events_added_after_subscription(tmp_path):
    job = Job(id="job-2", kind="rehabilitation", workspace=tmp_path)

    async def collect():
        stream = stream_job_events(
            job,
            heartbeat_seconds=1.0,
            poll_seconds=0.001,
        )
        pending = asyncio.create_task(stream.__anext__())
        await asyncio.sleep(0.005)
        job.push_event({"type": "result", "report": {}})
        job.status = "done"
        return await pending

    chunk = asyncio.run(collect())
    event = json.loads(chunk.removeprefix("data: ").strip())
    assert event["type"] == "result"


def test_prune_stale_removes_expired_jobs_and_orphan_dirs(tmp_path):
    from backend.app.services.jobs import JobRegistry

    registry = JobRegistry(tmp_path / "jobs")

    # Finished job past the TTL: dropped from the registry, workspace removed.
    old_done = registry.create("running")
    old_done.status = "done"
    old_done.created_at -= 100_000
    stale_mtime = time.time() - 100_000
    os.utime(old_done.workspace, (stale_mtime, stale_mtime))

    # Running job past the TTL: must survive untouched.
    old_running = registry.create("swimming")
    old_running.status = "running"
    old_running.created_at -= 100_000
    os.utime(old_running.workspace, (stale_mtime, stale_mtime))

    # Orphaned directory from a previous process, older than the TTL.
    orphan = registry.root / "orphan123"
    orphan.mkdir()
    os.utime(orphan, (stale_mtime, stale_mtime))

    registry.prune_stale(ttl_seconds=3600)

    assert registry.get(old_done.id) is None
    assert not old_done.workspace.exists()
    assert registry.get(old_running.id) is old_running
    assert old_running.workspace.exists()
    assert not orphan.exists()


def test_prune_stale_keeps_recent_finished_jobs(tmp_path):
    from backend.app.services.jobs import JobRegistry

    registry = JobRegistry(tmp_path / "jobs")
    job = registry.create("cycling")
    job.status = "done"

    registry.prune_stale(ttl_seconds=3600)

    assert registry.get(job.id) is job
    assert job.workspace.exists()
