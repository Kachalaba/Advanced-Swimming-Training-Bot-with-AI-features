"""Clinical Pilot Mode patient, episode, and visit endpoints."""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Literal, Optional, TypeVar

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from video_analysis.athlete_database import get_database
from video_analysis.clinical_repository import (
    ClinicalConflictError,
    ClinicalNotFoundError,
    ClinicalRepository,
    ClinicalValidationError,
    get_clinical_repository,
)

router = APIRouter(tags=["clinical"])
T = TypeVar("T")


def _to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(part.capitalize() for part in rest)


class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        from_attributes=True,
    )


class AthleteSummary(CamelModel):
    id: int
    name: str


class EpisodePayload(CamelModel):
    id: int
    patient_profile_id: int
    title: str
    protocol: str
    functional_goal: str
    target_left_rom: Optional[float]
    target_right_rom: Optional[float]
    status: Literal["active", "completed", "archived"]
    started_at: str
    completed_at: Optional[str]
    created_at: str
    updated_at: str


class VisitPayload(CamelModel):
    id: int
    rehab_episode_id: int
    training_session_id: Optional[int]
    visited_at: str
    capture_source: Literal["live", "upload"]
    pre_session_note: str
    specialist_observation: str
    capture_quality: Optional[Literal["acceptable", "accepted_with_warning", "repeat_required"]]
    capture_quality_details: str
    warning_acknowledged: bool
    status: Literal["draft", "finalized"]
    created_at: str
    updated_at: str


class ProgressPayload(CamelModel):
    visit_id: int
    training_session_id: int
    date: str
    protocol: str
    left_rom: float
    right_rom: float
    symmetry: float
    repetitions: int
    completion_score: float
    valid_frames: Optional[int]
    has_video: bool
    capture_quality: Literal["acceptable", "accepted_with_warning"]
    capture_quality_details: str


class PatientPayload(CamelModel):
    id: int
    athlete_id: int
    display_name: str
    affected_side: Literal["left", "right", "bilateral", "unspecified"]
    clinical_context: str
    precautions: str
    status: Literal["active", "archived"]
    created_at: str
    updated_at: str
    athlete: AthleteSummary
    active_episode: Optional[EpisodePayload]
    latest_visit: Optional[VisitPayload]


class PatientDetailPayload(PatientPayload):
    episodes: list[EpisodePayload]


class EpisodeDetailPayload(EpisodePayload):
    patient: PatientPayload
    athlete: AthleteSummary
    visits: list[VisitPayload]
    progress: list[ProgressPayload]


class CreatePatientRequest(CamelModel):
    athlete_id: int = Field(ge=1)
    display_name: str = Field(min_length=1, max_length=120)
    affected_side: Literal["left", "right", "bilateral", "unspecified"]
    clinical_context: str = Field(default="", max_length=2000)
    precautions: str = Field(default="", max_length=2000)


class UpdatePatientRequest(CamelModel):
    display_name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    affected_side: Optional[Literal["left", "right", "bilateral", "unspecified"]] = None
    clinical_context: Optional[str] = Field(default=None, max_length=2000)
    precautions: Optional[str] = Field(default=None, max_length=2000)


class CreateEpisodeRequest(CamelModel):
    title: str = Field(min_length=1, max_length=160)
    protocol: str
    functional_goal: str = Field(min_length=1, max_length=2000)
    target_left_rom: Optional[float] = Field(default=None, ge=0, le=360)
    target_right_rom: Optional[float] = Field(default=None, ge=0, le=360)


class UpdateEpisodeRequest(CamelModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=160)
    functional_goal: Optional[str] = Field(default=None, min_length=1, max_length=2000)
    target_left_rom: Optional[float] = Field(default=None, ge=0, le=360)
    target_right_rom: Optional[float] = Field(default=None, ge=0, le=360)
    status: Optional[Literal["active", "completed", "archived"]] = None


class CreateVisitRequest(CamelModel):
    capture_source: Literal["live", "upload"]
    pre_session_note: str = Field(default="", max_length=2000)


class UpdateVisitRequest(CamelModel):
    training_session_id: Optional[int] = Field(default=None, ge=1)
    specialist_observation: Optional[str] = Field(default=None, max_length=4000)
    capture_quality: Optional[Literal["acceptable", "accepted_with_warning", "repeat_required"]] = None
    capture_quality_details: Optional[str] = Field(default=None, max_length=2000)
    warning_acknowledged: Optional[bool] = None


def _call(operation: Callable[[], T]) -> T:
    try:
        return operation()
    except ClinicalNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ClinicalConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ClinicalValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def _repository() -> ClinicalRepository:
    return get_clinical_repository()


def _athlete_payload(athlete_id: int) -> dict:
    athlete = get_database().get_athlete(athlete_id=athlete_id)
    if athlete is None:
        raise ClinicalNotFoundError("Athlete not found")
    return {"id": int(athlete.id), "name": athlete.name}


def _episode_payload(episode) -> dict:
    return asdict(episode)


def _visit_payload(visit) -> dict:
    return asdict(visit)


def _progress_payload(observation) -> dict:
    return dict(observation) if isinstance(observation, dict) else asdict(observation)


def _patient_payload(patient, *, include_episodes: bool) -> dict:
    repository = _repository()
    episodes = repository.list_patient_episodes(int(patient.id))
    active_episode = next(
        (episode for episode in episodes if episode.status == "active"),
        None,
    )
    visits = [visit for episode in episodes for visit in repository.list_episode_visits(int(episode.id))]
    latest_visit = max(
        visits,
        key=lambda visit: (visit.visited_at, int(visit.id or 0)),
        default=None,
    )
    payload = {
        **asdict(patient),
        "athlete": _athlete_payload(patient.athlete_id),
        "active_episode": (_episode_payload(active_episode) if active_episode else None),
        "latest_visit": _visit_payload(latest_visit) if latest_visit else None,
    }
    if include_episodes:
        payload["episodes"] = [_episode_payload(episode) for episode in episodes]
    return payload


@router.get("/patients", response_model=list[PatientPayload])
def list_patients(
    include_archived: bool = Query(default=False, alias="includeArchived"),
):
    return _call(
        lambda: [
            _patient_payload(patient, include_episodes=False)
            for patient in _repository().list_patients(include_archived)
        ]
    )


@router.post(
    "/patients",
    response_model=PatientPayload,
    status_code=status.HTTP_201_CREATED,
)
def create_patient(request: CreatePatientRequest):
    return _call(
        lambda: _patient_payload(
            _repository().create_patient(**request.model_dump()),
            include_episodes=False,
        )
    )


@router.get("/patients/{patient_id}", response_model=PatientDetailPayload)
def get_patient(patient_id: int):
    return _call(
        lambda: _patient_payload(
            _repository().get_patient(patient_id),
            include_episodes=True,
        )
    )


@router.patch("/patients/{patient_id}", response_model=PatientPayload)
def update_patient(patient_id: int, request: UpdatePatientRequest):
    def operation():
        current = _repository().get_patient(patient_id)
        values = request.model_dump(exclude_none=True)
        return _patient_payload(
            _repository().update_patient(
                patient_id,
                display_name=values.get("display_name", current.display_name),
                affected_side=values.get("affected_side", current.affected_side),
                clinical_context=values.get("clinical_context", current.clinical_context),
                precautions=values.get("precautions", current.precautions),
            ),
            include_episodes=False,
        )

    return _call(operation)


@router.post("/patients/{patient_id}/archive", response_model=PatientPayload)
def archive_patient(patient_id: int):
    return _call(
        lambda: _patient_payload(
            _repository().archive_patient(patient_id),
            include_episodes=False,
        )
    )


@router.post(
    "/patients/{patient_id}/episodes",
    response_model=EpisodePayload,
    status_code=status.HTTP_201_CREATED,
)
def create_episode(patient_id: int, request: CreateEpisodeRequest):
    return _call(
        lambda: _episode_payload(
            _repository().create_episode(
                patient_profile_id=patient_id,
                **request.model_dump(),
            )
        )
    )


@router.get("/episodes/{episode_id}", response_model=EpisodeDetailPayload)
def get_episode(episode_id: int):
    def operation():
        repository = _repository()
        episode = repository.get_episode(episode_id)
        patient = repository.get_patient(episode.patient_profile_id)
        visits = repository.list_episode_visits(episode_id)
        return {
            **_episode_payload(episode),
            "patient": _patient_payload(patient, include_episodes=False),
            "athlete": _athlete_payload(patient.athlete_id),
            "visits": [_visit_payload(visit) for visit in visits],
            "progress": [
                _progress_payload(observation) for observation in repository.list_episode_progress(episode_id)
            ],
        }

    return _call(operation)


@router.patch("/episodes/{episode_id}", response_model=EpisodePayload)
def update_episode(episode_id: int, request: UpdateEpisodeRequest):
    def operation():
        repository = _repository()
        current = repository.get_episode(episode_id)
        values = request.model_dump(exclude_none=True)
        return _episode_payload(
            repository.update_episode(
                episode_id,
                title=values.get("title", current.title),
                functional_goal=values.get("functional_goal", current.functional_goal),
                target_left_rom=values.get("target_left_rom", current.target_left_rom),
                target_right_rom=values.get("target_right_rom", current.target_right_rom),
                status=values.get("status", current.status),
            )
        )

    return _call(operation)


@router.post("/episodes/{episode_id}/archive", response_model=EpisodePayload)
def archive_episode(episode_id: int):
    return _call(lambda: _episode_payload(_repository().archive_episode(episode_id)))


@router.post(
    "/episodes/{episode_id}/visits",
    response_model=VisitPayload,
    status_code=status.HTTP_201_CREATED,
)
def create_visit(episode_id: int, request: CreateVisitRequest):
    return _call(
        lambda: _visit_payload(
            _repository().create_visit(
                rehab_episode_id=episode_id,
                **request.model_dump(),
            )
        )
    )


@router.get("/visits/{visit_id}", response_model=VisitPayload)
def get_visit(visit_id: int):
    return _call(lambda: _visit_payload(_repository().get_visit(visit_id)))


@router.patch("/visits/{visit_id}", response_model=VisitPayload)
def update_visit(visit_id: int, request: UpdateVisitRequest):
    return _call(
        lambda: _visit_payload(
            _repository().update_visit(
                visit_id,
                **request.model_dump(exclude_none=True),
            )
        )
    )


@router.post("/visits/{visit_id}/finalize", response_model=VisitPayload)
def finalize_visit(visit_id: int):
    return _call(lambda: _visit_payload(_repository().finalize_visit(visit_id)))
