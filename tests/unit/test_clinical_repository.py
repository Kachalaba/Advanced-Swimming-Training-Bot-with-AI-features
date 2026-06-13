"""Tests for local Clinical Pilot persistence and domain rules."""

import json
import sqlite3
from pathlib import Path

import pytest

from video_analysis.clinical_repository import ClinicalConflictError, ClinicalRepository, ClinicalValidationError


def _table_names(path: Path) -> set[str]:
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    return {row[0] for row in rows}


def _existing_athlete_database(tmp_path: Path) -> Path:
    path = tmp_path / "athletes.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE athletes (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        connection.execute("INSERT INTO athletes (id, name) VALUES (1, 'Patient A')")
    return path


def _repository_with_athlete(tmp_path: Path) -> ClinicalRepository:
    path = tmp_path / "clinical.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE athletes (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        connection.execute(
            """
            CREATE TABLE sessions (
                id INTEGER PRIMARY KEY,
                athlete_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                full_analysis TEXT,
                video_path TEXT
            )
            """
        )
        connection.execute("INSERT INTO athletes (id, name) VALUES (7, 'Patient A')")
        connection.execute(
            """
            INSERT INTO sessions (
                id, athlete_id, date, full_analysis, video_path
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                31,
                7,
                "2026-06-13T12:00:00+00:00",
                json.dumps(
                    {
                        "rehab_analysis": {
                            "protocol": "shoulder_flexion",
                            "valid_frames": 30,
                            "target_metrics": {
                                "left": {"rom": 142.0},
                                "right": {"rom": 130.0},
                            },
                            "symmetry": {"score": 91.0},
                            "total_correct_reps": 4,
                            "completion_score": 88.0,
                        }
                    }
                ),
                "/data/session-videos/rehab.mp4",
            ),
        )
    return ClinicalRepository(path)


def _patient_and_episode(
    repository: ClinicalRepository,
):
    patient = repository.create_patient(
        athlete_id=7,
        display_name="Patient A",
        affected_side="right",
        clinical_context="Post-operative shoulder mobility",
        precautions="Stop on sharp pain",
    )
    episode = repository.create_episode(
        patient_profile_id=int(patient.id or 0),
        title="Right shoulder recovery",
        protocol="shoulder_flexion",
        functional_goal="Reach an overhead shelf",
        target_left_rom=150,
        target_right_rom=140,
    )
    return patient, episode


def test_initialization_backs_up_existing_database_before_adding_tables(
    tmp_path: Path,
):
    path = _existing_athlete_database(tmp_path)

    ClinicalRepository(path)

    assert {
        "patient_profiles",
        "rehab_episodes",
        "clinical_visits",
    } <= _table_names(path)
    backups = list(tmp_path.glob("athletes.pre-clinical-*.db"))
    assert len(backups) == 1
    assert _table_names(backups[0]) == {"athletes"}


def test_initialization_is_idempotent_and_does_not_create_second_backup(
    tmp_path: Path,
):
    path = _existing_athlete_database(tmp_path)

    ClinicalRepository(path)
    ClinicalRepository(path)

    assert len(list(tmp_path.glob("athletes.pre-clinical-*.db"))) == 1


def test_patient_episode_and_draft_visit_round_trip(tmp_path: Path):
    repository = _repository_with_athlete(tmp_path)

    patient, episode = _patient_and_episode(repository)
    visit = repository.create_visit(
        rehab_episode_id=int(episode.id or 0),
        capture_source="live",
        pre_session_note="No pain at rest",
    )

    assert repository.get_patient(int(patient.id or 0)) == patient
    assert repository.get_episode(int(episode.id or 0)) == episode
    assert repository.list_patient_episodes(int(patient.id or 0)) == [episode]
    assert repository.get_visit(int(visit.id or 0)) == visit
    assert repository.list_episode_visits(int(episode.id or 0)) == [visit]
    assert visit.status == "draft"


def test_rejects_second_active_episode_for_same_patient_and_protocol(
    tmp_path: Path,
):
    repository = _repository_with_athlete(tmp_path)
    patient, _ = _patient_and_episode(repository)

    with pytest.raises(ClinicalConflictError):
        repository.create_episode(
            patient_profile_id=int(patient.id or 0),
            title="Duplicate",
            protocol="shoulder_flexion",
            functional_goal="Duplicate goal",
        )


def test_finalize_requires_session_quality_and_observation(tmp_path: Path):
    repository = _repository_with_athlete(tmp_path)
    _, episode = _patient_and_episode(repository)
    visit = repository.create_visit(
        rehab_episode_id=int(episode.id or 0),
        capture_source="live",
        pre_session_note="No pain at rest",
    )

    with pytest.raises(ClinicalValidationError):
        repository.finalize_visit(int(visit.id or 0))

    updated = repository.update_visit(
        int(visit.id or 0),
        training_session_id=31,
        capture_quality="accepted_with_warning",
        capture_quality_details="Camera tilt acknowledged",
        warning_acknowledged=True,
        specialist_observation="ROM remains asymmetric.",
    )
    finalized = repository.finalize_visit(int(updated.id or 0))

    assert finalized.status == "finalized"
    assert finalized.training_session_id == 31
    assert finalized.warning_acknowledged is True


def test_warning_requires_acknowledgement_and_repeat_required_cannot_finalize(
    tmp_path: Path,
):
    repository = _repository_with_athlete(tmp_path)
    _, episode = _patient_and_episode(repository)
    visit = repository.create_visit(
        rehab_episode_id=int(episode.id or 0),
        capture_source="upload",
        pre_session_note="Stable",
    )

    repository.update_visit(
        int(visit.id or 0),
        training_session_id=31,
        capture_quality="accepted_with_warning",
        capture_quality_details="Low confidence",
        specialist_observation="Review completed.",
    )
    with pytest.raises(ClinicalValidationError):
        repository.finalize_visit(int(visit.id or 0))

    repository.update_visit(
        int(visit.id or 0),
        capture_quality="repeat_required",
        warning_acknowledged=True,
    )
    with pytest.raises(ClinicalValidationError):
        repository.finalize_visit(int(visit.id or 0))


def test_archiving_patient_archives_active_episodes(tmp_path: Path):
    repository = _repository_with_athlete(tmp_path)
    patient, episode = _patient_and_episode(repository)

    archived = repository.archive_patient(int(patient.id or 0))

    assert archived.status == "archived"
    assert repository.get_episode(int(episode.id or 0)).status == "archived"
    assert repository.list_patients() == []
    assert repository.list_patients(include_archived=True) == [archived]


def test_episode_progress_uses_finalized_compatible_session_report(tmp_path: Path):
    repository = _repository_with_athlete(tmp_path)
    _, episode = _patient_and_episode(repository)
    visit = repository.create_visit(
        rehab_episode_id=int(episode.id or 0),
        capture_source="live",
        pre_session_note="Stable",
    )
    repository.update_visit(
        int(visit.id or 0),
        training_session_id=31,
        capture_quality="acceptable",
        specialist_observation="Review completed.",
    )
    repository.finalize_visit(int(visit.id or 0))

    observations = repository.list_episode_progress(int(episode.id or 0))

    assert len(observations) == 1
    assert observations[0].visit_id == visit.id
    assert observations[0].left_rom == 142.0
    assert observations[0].right_rom == 130.0
    assert observations[0].has_video is True
