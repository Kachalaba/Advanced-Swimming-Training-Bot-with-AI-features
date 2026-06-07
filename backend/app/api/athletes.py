"""Athlete roster and persisted training history endpoints."""

from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from video_analysis.athlete_database import Athlete as DatabaseAthlete
from video_analysis.athlete_database import TrainingSession, get_database

router = APIRouter(tags=["athletes"])


class Athlete(BaseModel):
    id: str
    name: str
    initials: str
    handle: Optional[str] = None


class SessionSummary(BaseModel):
    id: int
    athlete_id: int
    session_type: str
    date: str
    duration_sec: float
    distance_m: float
    reps: int
    exercise_type: str
    score: float
    summary: str
    has_video: bool
    artifact_download_url: Optional[str] = None


def _initials(name: str) -> str:
    parts = [part for part in name.split() if part]
    return "".join(part[0].upper() for part in parts[:2]) or "A"


def _athlete_payload(athlete: DatabaseAthlete) -> Athlete:
    handle = "@kachamba_swim" if athlete.name == "Nikita K." else None
    return Athlete(
        id=str(athlete.id),
        name=athlete.name,
        initials=_initials(athlete.name),
        handle=handle,
    )


def _session_payload(session: TrainingSession) -> SessionSummary:
    score = session.ai_score or session.stability_score or session.symmetry_score or 0
    summary = session.ai_summary or session.notes or session.exercise_type
    return SessionSummary(
        id=int(session.id or 0),
        athlete_id=session.athlete_id,
        session_type=session.session_type,
        date=session.date,
        duration_sec=float(session.duration_sec or 0),
        distance_m=float(session.distance_m or 0),
        reps=int(session.reps or 0),
        exercise_type=session.exercise_type or "",
        score=float(score),
        summary=summary or "",
        has_video=bool(session.video_path),
        artifact_download_url=(
            f"/api/tools/history/{session.id}/download"
            if session.session_type == "tool" and session.video_path and session.id
            else None
        ),
    )


@router.get("/me")
def me() -> Athlete:
    database = get_database()
    athlete_name = os.environ.get("CURRENT_ATHLETE_NAME", "Nikita K.")
    athlete = database.get_athlete(name=athlete_name)
    if athlete is None:
        athlete_id = database.add_athlete(DatabaseAthlete(name=athlete_name))
        athlete = database.get_athlete(athlete_id=athlete_id)
    if athlete is None:
        raise HTTPException(status_code=500, detail="Could not load current athlete")
    return _athlete_payload(athlete)


@router.get("")
def list_athletes() -> list[Athlete]:
    return [_athlete_payload(athlete) for athlete in get_database().get_all_athletes()]


@router.get("/{athlete_id}/sessions")
def list_sessions(
    athlete_id: int,
    session_type: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[SessionSummary]:
    database = get_database()
    if database.get_athlete(athlete_id=athlete_id) is None:
        raise HTTPException(status_code=404, detail="Athlete not found")
    sessions = database.get_sessions(
        athlete_id=athlete_id,
        session_type=session_type,
        limit=limit,
    )
    return [_session_payload(session) for session in sessions]
