"""Tests for the in-memory analysis job event stream."""

import asyncio
import json

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
