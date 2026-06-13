"""Athlete roster and persisted training history endpoints."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

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


class RehabProgressSession(BaseModel):
    id: int
    date: str
    protocol: str
    left_rom: float = Field(serialization_alias="leftRom")
    right_rom: float = Field(serialization_alias="rightRom")
    symmetry: float
    repetitions: int
    completion_score: float = Field(serialization_alias="completionScore")
    valid_frames: Optional[int] = Field(serialization_alias="validFrames")
    has_video: bool = Field(serialization_alias="hasVideo")


class RehabProgressResponse(BaseModel):
    athlete: Athlete
    sessions: list[RehabProgressSession]
    protocols: list[str]


REHAB_PROTOCOLS = frozenset(
    {
        "shoulder_flexion",
        "shoulder_abduction",
        "elbow_flexion",
        "knee_extension",
        "hip_abduction",
    }
)


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


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rehab_progress_payload(
    session: TrainingSession,
) -> Optional[RehabProgressSession]:
    try:
        stored = json.loads(session.full_analysis or "{}")
    except (TypeError, json.JSONDecodeError):
        return None

    report = stored.get("rehab_analysis", stored)
    if not isinstance(report, dict):
        return None
    protocol = report.get("protocol")
    if protocol not in REHAB_PROTOCOLS:
        return None

    target_metrics = report.get("target_metrics")
    symmetry_data = report.get("symmetry")
    if not isinstance(target_metrics, dict) or not isinstance(symmetry_data, dict):
        return None
    left = target_metrics.get("left")
    right = target_metrics.get("right")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None

    left_rom = _finite_number(left.get("rom"))
    right_rom = _finite_number(right.get("rom"))
    symmetry = _finite_number(symmetry_data.get("score"))
    completion = _finite_number(report.get("completion_score"))
    if None in (left_rom, right_rom, symmetry, completion):
        return None

    valid_frames = report.get("valid_frames")
    return RehabProgressSession(
        id=int(session.id or 0),
        date=session.date,
        protocol=protocol,
        left_rom=left_rom,
        right_rom=right_rom,
        symmetry=symmetry,
        repetitions=int(report.get("total_correct_reps") or 0),
        completion_score=completion,
        valid_frames=int(valid_frames) if valid_frames is not None else None,
        has_video=bool(session.video_path),
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


@router.get(
    "/{athlete_id}/rehabilitation/progress",
    response_model=RehabProgressResponse,
)
def rehabilitation_progress(athlete_id: int) -> RehabProgressResponse:
    database = get_database()
    athlete = database.get_athlete(athlete_id=athlete_id)
    if athlete is None:
        raise HTTPException(status_code=404, detail="Athlete not found")

    observations = [
        observation
        for session in database.get_sessions(
            athlete_id=athlete_id,
            session_type="rehab",
            limit=500,
        )
        if (observation := _rehab_progress_payload(session)) is not None
    ]
    observations.sort(key=lambda item: item.date)
    protocols = list(dict.fromkeys(item.protocol for item in observations))
    return RehabProgressResponse(
        athlete=_athlete_payload(athlete),
        sessions=observations,
        protocols=protocols,
    )
