"""API contract tests for Clinical Pilot Mode."""

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.app.api import clinical  # noqa: E402
from video_analysis.clinical_repository import (  # noqa: E402
    ClinicalConflictError,
    ClinicalNotFoundError,
    ClinicalValidationError,
    ClinicalVisit,
    PatientProfile,
    RehabEpisode,
)

NOW = "2026-06-13T12:00:00+00:00"


def _patient(status="active"):
    return PatientProfile(
        id=11,
        athlete_id=3,
        display_name="Patient A",
        affected_side="right",
        clinical_context="Shoulder mobility",
        precautions="Stop on sharp pain",
        status=status,
        created_at=NOW,
        updated_at=NOW,
    )


def _episode(status="active"):
    return RehabEpisode(
        id=21,
        patient_profile_id=11,
        title="Shoulder recovery",
        protocol="shoulder_flexion",
        functional_goal="Reach overhead",
        target_left_rom=150.0,
        target_right_rom=140.0,
        status=status,
        started_at=NOW,
        completed_at=None,
        created_at=NOW,
        updated_at=NOW,
    )


def _visit(
    visit_id=31,
    *,
    status="draft",
    quality=None,
    session_id=None,
):
    return ClinicalVisit(
        id=visit_id,
        rehab_episode_id=21,
        training_session_id=session_id,
        visited_at=NOW,
        capture_source="live",
        pre_session_note="Stable",
        specialist_observation=("ROM remains asymmetric." if status == "finalized" else ""),
        capture_quality=quality,
        capture_quality_details="Camera tilt" if quality else "",
        warning_acknowledged=quality == "accepted_with_warning",
        status=status,
        created_at=NOW,
        updated_at=NOW,
    )


class FakeAthleteDatabase:
    def get_athlete(self, athlete_id=None, name=None):
        if athlete_id == 3:
            return SimpleNamespace(id=3, name="Patient A")
        return None


class FakeClinicalRepository:
    def __init__(self):
        self.patient = _patient()
        self.episode = _episode()
        self.visit = _visit()

    def list_patients(self, include_archived=False):
        return [self.patient]

    def create_patient(self, **kwargs):
        return self.patient

    def get_patient(self, patient_id):
        if patient_id != 11:
            raise ClinicalNotFoundError("Patient not found")
        return self.patient

    def update_patient(self, patient_id, **kwargs):
        return self.get_patient(patient_id)

    def archive_patient(self, patient_id):
        self.patient = _patient(status="archived")
        return self.patient

    def list_patient_episodes(self, patient_id):
        return [self.episode] if patient_id == 11 else []

    def create_episode(self, patient_profile_id, **kwargs):
        if patient_profile_id != 11:
            raise ClinicalNotFoundError("Patient not found")
        return self.episode

    def get_episode(self, episode_id):
        if episode_id != 21:
            raise ClinicalNotFoundError("Episode not found")
        return self.episode

    def update_episode(self, episode_id, **kwargs):
        return self.get_episode(episode_id)

    def archive_episode(self, episode_id):
        self.episode = _episode(status="archived")
        return self.episode

    def create_visit(self, rehab_episode_id, **kwargs):
        if rehab_episode_id != 21:
            raise ClinicalNotFoundError("Episode not found")
        return self.visit

    def get_visit(self, visit_id):
        if visit_id != 31:
            raise ClinicalNotFoundError("Visit not found")
        return self.visit

    def update_visit(self, visit_id, **kwargs):
        return self.get_visit(visit_id)

    def finalize_visit(self, visit_id):
        self.visit = _visit(
            visit_id=visit_id,
            status="finalized",
            quality="acceptable",
            session_id=91,
        )
        return self.visit

    def list_episode_visits(self, episode_id):
        return [
            _visit(31),
            _visit(
                32,
                status="finalized",
                quality="accepted_with_warning",
                session_id=91,
            ),
            _visit(33, quality="repeat_required", session_id=92),
        ]

    def list_episode_progress(self, episode_id):
        return [
            {
                "visit_id": 32,
                "training_session_id": 91,
                "date": NOW,
                "protocol": "shoulder_flexion",
                "left_rom": 142.0,
                "right_rom": 130.0,
                "symmetry": 91.0,
                "repetitions": 4,
                "completion_score": 88.0,
                "valid_frames": 30,
                "has_video": True,
                "capture_quality": "accepted_with_warning",
                "capture_quality_details": "Camera tilt",
            }
        ]


def _client(monkeypatch, repository=None):
    repository = repository or FakeClinicalRepository()
    monkeypatch.setattr(clinical, "get_clinical_repository", lambda: repository)
    monkeypatch.setattr(clinical, "get_database", lambda: FakeAthleteDatabase())
    app = FastAPI()
    app.include_router(clinical.router, prefix="/api/clinical")
    return TestClient(app)


def test_patient_roster_and_detail_contract(monkeypatch):
    client = _client(monkeypatch)

    created = client.post(
        "/api/clinical/patients",
        json={
            "athleteId": 3,
            "displayName": "Patient A",
            "affectedSide": "right",
            "clinicalContext": "Shoulder mobility",
            "precautions": "Stop on sharp pain",
        },
    )
    roster = client.get("/api/clinical/patients")
    detail = client.get("/api/clinical/patients/11")

    assert created.status_code == 201
    assert created.json()["affectedSide"] == "right"
    assert roster.json()[0]["activeEpisode"]["id"] == 21
    assert detail.json()["athlete"] == {"id": 3, "name": "Patient A"}
    assert detail.json()["episodes"][0]["functionalGoal"] == "Reach overhead"


def test_episode_and_visit_finalization_contract(monkeypatch):
    client = _client(monkeypatch)

    episode = client.post(
        "/api/clinical/patients/11/episodes",
        json={
            "title": "Shoulder recovery",
            "protocol": "shoulder_flexion",
            "functionalGoal": "Reach overhead",
            "targetLeftRom": 150,
            "targetRightRom": 140,
        },
    )
    visit = client.post(
        "/api/clinical/episodes/21/visits",
        json={"captureSource": "live", "preSessionNote": "Stable"},
    )
    finalized = client.post("/api/clinical/visits/31/finalize")

    assert episode.status_code == 201
    assert visit.status_code == 201
    assert finalized.json()["status"] == "finalized"
    assert finalized.json()["trainingSessionId"] == 91


def test_episode_detail_contains_only_finalized_usable_progress(monkeypatch):
    payload = _client(monkeypatch).get("/api/clinical/episodes/21").json()

    assert [visit["id"] for visit in payload["visits"]] == [31, 32, 33]
    assert [observation["visitId"] for observation in payload["progress"]] == [32]
    assert payload["progress"][0]["trainingSessionId"] == 91
    assert payload["progress"][0]["leftRom"] == 142.0


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (ClinicalNotFoundError("missing"), 404),
        (ClinicalConflictError("conflict"), 409),
        (ClinicalValidationError("invalid"), 422),
    ],
)
def test_domain_errors_map_to_http_status(monkeypatch, error, expected_status):
    class FailingRepository(FakeClinicalRepository):
        def create_patient(self, **kwargs):
            raise error

    response = _client(monkeypatch, FailingRepository()).post(
        "/api/clinical/patients",
        json={
            "athleteId": 3,
            "displayName": "Patient A",
            "affectedSide": "right",
            "clinicalContext": "",
            "precautions": "",
        },
    )

    assert response.status_code == expected_status
