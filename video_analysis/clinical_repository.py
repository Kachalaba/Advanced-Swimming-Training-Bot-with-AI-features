"""Local persistence for the Clinical Pilot rehabilitation workflow."""

from __future__ import annotations

import os
import shutil
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal, Optional

AffectedSide = Literal["left", "right", "bilateral", "unspecified"]
PatientStatus = Literal["active", "archived"]
EpisodeStatus = Literal["active", "completed", "archived"]
CaptureSource = Literal["live", "upload"]
CaptureQuality = Literal[
    "acceptable",
    "accepted_with_warning",
    "repeat_required",
]
VisitStatus = Literal["draft", "finalized"]

CLINICAL_TABLES = frozenset({"patient_profiles", "rehab_episodes", "clinical_visits"})
REHAB_PROTOCOLS = frozenset(
    {
        "shoulder_flexion",
        "shoulder_abduction",
        "elbow_flexion",
        "knee_extension",
        "hip_abduction",
    }
)


class ClinicalRepositoryError(Exception):
    """Base error for clinical persistence."""


class ClinicalNotFoundError(ClinicalRepositoryError):
    """Requested clinical record does not exist."""


class ClinicalConflictError(ClinicalRepositoryError):
    """Requested change conflicts with an existing clinical record."""


class ClinicalValidationError(ClinicalRepositoryError):
    """Requested clinical state is incomplete or invalid."""


@dataclass(frozen=True)
class PatientProfile:
    id: Optional[int]
    athlete_id: int
    display_name: str
    affected_side: AffectedSide
    clinical_context: str
    precautions: str
    status: PatientStatus
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class RehabEpisode:
    id: Optional[int]
    patient_profile_id: int
    title: str
    protocol: str
    functional_goal: str
    target_left_rom: Optional[float]
    target_right_rom: Optional[float]
    status: EpisodeStatus
    started_at: str
    completed_at: Optional[str]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ClinicalVisit:
    id: Optional[int]
    rehab_episode_id: int
    training_session_id: Optional[int]
    visited_at: str
    capture_source: CaptureSource
    pre_session_note: str
    specialist_observation: str
    capture_quality: Optional[CaptureQuality]
    capture_quality_details: str
    warning_acknowledged: bool
    status: VisitStatus
    created_at: str
    updated_at: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ClinicalRepository:
    """Additive clinical records stored beside the existing athlete database."""

    def __init__(self, db_path: Optional[Path] = None):
        configured_path = os.environ.get("ATHLETE_DB_PATH", "data/athletes.db")
        self.db_path = Path(db_path) if db_path is not None else Path(configured_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._backup_before_first_migration()
        self._initialize_schema()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.db_path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _table_names(self) -> set[str]:
        if not self.db_path.exists():
            return set()
        with sqlite3.connect(str(self.db_path)) as connection:
            rows = connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                """
            ).fetchall()
        return {str(row[0]) for row in rows}

    def _backup_before_first_migration(self) -> None:
        if not self.db_path.exists() or self.db_path.stat().st_size == 0:
            return
        if CLINICAL_TABLES.issubset(self._table_names()):
            return
        pattern = f"{self.db_path.stem}.pre-clinical-*.db"
        if any(self.db_path.parent.glob(pattern)):
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = self.db_path.with_name(f"{self.db_path.stem}.pre-clinical-{stamp}.db")
        shutil.copy2(self.db_path, backup)

    def _initialize_schema(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS patient_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    athlete_id INTEGER NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    affected_side TEXT NOT NULL
                        CHECK (affected_side IN (
                            'left', 'right', 'bilateral', 'unspecified'
                        )),
                    clinical_context TEXT NOT NULL DEFAULT '',
                    precautions TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'archived')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (athlete_id) REFERENCES athletes(id)
                );

                CREATE TABLE IF NOT EXISTS rehab_episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_profile_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    protocol TEXT NOT NULL,
                    functional_goal TEXT NOT NULL,
                    target_left_rom REAL,
                    target_right_rom REAL,
                    status TEXT NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'completed', 'archived')),
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (patient_profile_id)
                        REFERENCES patient_profiles(id)
                );

                CREATE TABLE IF NOT EXISTS clinical_visits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rehab_episode_id INTEGER NOT NULL,
                    training_session_id INTEGER,
                    visited_at TEXT NOT NULL,
                    capture_source TEXT NOT NULL
                        CHECK (capture_source IN ('live', 'upload')),
                    pre_session_note TEXT NOT NULL DEFAULT '',
                    specialist_observation TEXT NOT NULL DEFAULT '',
                    capture_quality TEXT
                        CHECK (capture_quality IS NULL OR capture_quality IN (
                            'acceptable',
                            'accepted_with_warning',
                            'repeat_required'
                        )),
                    capture_quality_details TEXT NOT NULL DEFAULT '',
                    warning_acknowledged INTEGER NOT NULL DEFAULT 0
                        CHECK (warning_acknowledged IN (0, 1)),
                    status TEXT NOT NULL DEFAULT 'draft'
                        CHECK (status IN ('draft', 'finalized')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (rehab_episode_id)
                        REFERENCES rehab_episodes(id),
                    FOREIGN KEY (training_session_id) REFERENCES sessions(id)
                );

                CREATE UNIQUE INDEX IF NOT EXISTS
                    ux_rehab_episodes_active_protocol
                ON rehab_episodes(patient_profile_id, protocol)
                WHERE status = 'active';

                CREATE INDEX IF NOT EXISTS ix_clinical_visits_episode_date
                ON clinical_visits(rehab_episode_id, visited_at);
                """
            )

    @staticmethod
    def _patient_from_row(row: sqlite3.Row) -> PatientProfile:
        return PatientProfile(
            id=int(row["id"]),
            athlete_id=int(row["athlete_id"]),
            display_name=str(row["display_name"]),
            affected_side=row["affected_side"],
            clinical_context=str(row["clinical_context"]),
            precautions=str(row["precautions"]),
            status=row["status"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _episode_from_row(row: sqlite3.Row) -> RehabEpisode:
        return RehabEpisode(
            id=int(row["id"]),
            patient_profile_id=int(row["patient_profile_id"]),
            title=str(row["title"]),
            protocol=str(row["protocol"]),
            functional_goal=str(row["functional_goal"]),
            target_left_rom=(float(row["target_left_rom"]) if row["target_left_rom"] is not None else None),
            target_right_rom=(float(row["target_right_rom"]) if row["target_right_rom"] is not None else None),
            status=row["status"],
            started_at=str(row["started_at"]),
            completed_at=(str(row["completed_at"]) if row["completed_at"] is not None else None),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _visit_from_row(row: sqlite3.Row) -> ClinicalVisit:
        return ClinicalVisit(
            id=int(row["id"]),
            rehab_episode_id=int(row["rehab_episode_id"]),
            training_session_id=(int(row["training_session_id"]) if row["training_session_id"] is not None else None),
            visited_at=str(row["visited_at"]),
            capture_source=row["capture_source"],
            pre_session_note=str(row["pre_session_note"]),
            specialist_observation=str(row["specialist_observation"]),
            capture_quality=row["capture_quality"],
            capture_quality_details=str(row["capture_quality_details"]),
            warning_acknowledged=bool(row["warning_acknowledged"]),
            status=row["status"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _required_text(value: str, field: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ClinicalValidationError(f"{field} is required")
        return normalized

    def create_patient(
        self,
        athlete_id: int,
        display_name: str,
        affected_side: AffectedSide,
        clinical_context: str,
        precautions: str,
    ) -> PatientProfile:
        if affected_side not in {"left", "right", "bilateral", "unspecified"}:
            raise ClinicalValidationError("Unknown affected side")
        now = _utc_now()
        with self._connection() as connection:
            athlete = connection.execute("SELECT id FROM athletes WHERE id = ?", (athlete_id,)).fetchone()
            if athlete is None:
                raise ClinicalNotFoundError("Athlete not found")
            try:
                cursor = connection.execute(
                    """
                    INSERT INTO patient_profiles (
                        athlete_id, display_name, affected_side,
                        clinical_context, precautions, status,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'active', ?, ?)
                    """,
                    (
                        athlete_id,
                        self._required_text(display_name, "display_name"),
                        affected_side,
                        clinical_context.strip(),
                        precautions.strip(),
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise ClinicalConflictError("Athlete already has a patient profile") from error
            patient_id = int(cursor.lastrowid)
        return self.get_patient(patient_id)

    def get_patient(self, patient_id: int) -> PatientProfile:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM patient_profiles WHERE id = ?", (patient_id,)).fetchone()
        if row is None:
            raise ClinicalNotFoundError("Patient not found")
        return self._patient_from_row(row)

    def list_patients(self, include_archived: bool = False) -> list[PatientProfile]:
        query = "SELECT * FROM patient_profiles"
        parameters: tuple[object, ...] = ()
        if not include_archived:
            query += " WHERE status = ?"
            parameters = ("active",)
        query += " ORDER BY display_name COLLATE NOCASE, id"
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._patient_from_row(row) for row in rows]

    def update_patient(
        self,
        patient_id: int,
        *,
        display_name: str,
        affected_side: AffectedSide,
        clinical_context: str,
        precautions: str,
    ) -> PatientProfile:
        if affected_side not in {"left", "right", "bilateral", "unspecified"}:
            raise ClinicalValidationError("Unknown affected side")
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE patient_profiles
                SET display_name = ?, affected_side = ?,
                    clinical_context = ?, precautions = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    self._required_text(display_name, "display_name"),
                    affected_side,
                    clinical_context.strip(),
                    precautions.strip(),
                    _utc_now(),
                    patient_id,
                ),
            )
            if cursor.rowcount == 0:
                raise ClinicalNotFoundError("Patient not found")
        return self.get_patient(patient_id)

    def archive_patient(self, patient_id: int) -> PatientProfile:
        now = _utc_now()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE patient_profiles
                SET status = 'archived', updated_at = ?
                WHERE id = ?
                """,
                (now, patient_id),
            )
            if cursor.rowcount == 0:
                raise ClinicalNotFoundError("Patient not found")
            connection.execute(
                """
                UPDATE rehab_episodes
                SET status = 'archived', updated_at = ?
                WHERE patient_profile_id = ? AND status = 'active'
                """,
                (now, patient_id),
            )
        return self.get_patient(patient_id)

    def create_episode(
        self,
        patient_profile_id: int,
        title: str,
        protocol: str,
        functional_goal: str,
        target_left_rom: Optional[float] = None,
        target_right_rom: Optional[float] = None,
    ) -> RehabEpisode:
        if protocol not in REHAB_PROTOCOLS:
            raise ClinicalValidationError("Unknown rehabilitation protocol")
        now = _utc_now()
        with self._connection() as connection:
            patient = connection.execute(
                """
                SELECT id FROM patient_profiles
                WHERE id = ? AND status = 'active'
                """,
                (patient_profile_id,),
            ).fetchone()
            if patient is None:
                raise ClinicalNotFoundError("Active patient not found")
            try:
                cursor = connection.execute(
                    """
                    INSERT INTO rehab_episodes (
                        patient_profile_id, title, protocol, functional_goal,
                        target_left_rom, target_right_rom, status, started_at,
                        completed_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'active', ?, NULL, ?, ?)
                    """,
                    (
                        patient_profile_id,
                        self._required_text(title, "title"),
                        protocol,
                        self._required_text(functional_goal, "functional_goal"),
                        target_left_rom,
                        target_right_rom,
                        now,
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise ClinicalConflictError("An active episode already exists for this protocol") from error
            episode_id = int(cursor.lastrowid)
        return self.get_episode(episode_id)

    def get_episode(self, episode_id: int) -> RehabEpisode:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM rehab_episodes WHERE id = ?", (episode_id,)).fetchone()
        if row is None:
            raise ClinicalNotFoundError("Episode not found")
        return self._episode_from_row(row)

    def list_patient_episodes(self, patient_id: int) -> list[RehabEpisode]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM rehab_episodes
                WHERE patient_profile_id = ?
                ORDER BY started_at DESC, id DESC
                """,
                (patient_id,),
            ).fetchall()
        return [self._episode_from_row(row) for row in rows]

    def update_episode(
        self,
        episode_id: int,
        *,
        title: str,
        functional_goal: str,
        target_left_rom: Optional[float],
        target_right_rom: Optional[float],
        status: EpisodeStatus,
    ) -> RehabEpisode:
        if status not in {"active", "completed", "archived"}:
            raise ClinicalValidationError("Unknown episode status")
        completed_at = _utc_now() if status == "completed" else None
        try:
            with self._connection() as connection:
                cursor = connection.execute(
                    """
                    UPDATE rehab_episodes
                    SET title = ?, functional_goal = ?,
                        target_left_rom = ?, target_right_rom = ?,
                        status = ?, completed_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        self._required_text(title, "title"),
                        self._required_text(functional_goal, "functional_goal"),
                        target_left_rom,
                        target_right_rom,
                        status,
                        completed_at,
                        _utc_now(),
                        episode_id,
                    ),
                )
                if cursor.rowcount == 0:
                    raise ClinicalNotFoundError("Episode not found")
        except sqlite3.IntegrityError as error:
            raise ClinicalConflictError("An active episode already exists for this protocol") from error
        return self.get_episode(episode_id)

    def archive_episode(self, episode_id: int) -> RehabEpisode:
        episode = self.get_episode(episode_id)
        return self.update_episode(
            episode_id,
            title=episode.title,
            functional_goal=episode.functional_goal,
            target_left_rom=episode.target_left_rom,
            target_right_rom=episode.target_right_rom,
            status="archived",
        )

    def create_visit(
        self,
        rehab_episode_id: int,
        capture_source: CaptureSource,
        pre_session_note: str,
    ) -> ClinicalVisit:
        if capture_source not in {"live", "upload"}:
            raise ClinicalValidationError("Unknown capture source")
        now = _utc_now()
        with self._connection() as connection:
            episode = connection.execute(
                """
                SELECT id FROM rehab_episodes
                WHERE id = ? AND status = 'active'
                """,
                (rehab_episode_id,),
            ).fetchone()
            if episode is None:
                raise ClinicalNotFoundError("Active episode not found")
            cursor = connection.execute(
                """
                INSERT INTO clinical_visits (
                    rehab_episode_id, training_session_id, visited_at,
                    capture_source, pre_session_note,
                    specialist_observation, capture_quality,
                    capture_quality_details, warning_acknowledged,
                    status, created_at, updated_at
                ) VALUES (?, NULL, ?, ?, ?, '', NULL, '', 0, 'draft', ?, ?)
                """,
                (
                    rehab_episode_id,
                    now,
                    capture_source,
                    pre_session_note.strip(),
                    now,
                    now,
                ),
            )
            visit_id = int(cursor.lastrowid)
        return self.get_visit(visit_id)

    def get_visit(self, visit_id: int) -> ClinicalVisit:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM clinical_visits WHERE id = ?", (visit_id,)).fetchone()
        if row is None:
            raise ClinicalNotFoundError("Visit not found")
        return self._visit_from_row(row)

    def list_episode_visits(self, episode_id: int) -> list[ClinicalVisit]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM clinical_visits
                WHERE rehab_episode_id = ?
                ORDER BY visited_at, id
                """,
                (episode_id,),
            ).fetchall()
        return [self._visit_from_row(row) for row in rows]

    def update_visit(
        self,
        visit_id: int,
        *,
        training_session_id: Optional[int] = None,
        specialist_observation: Optional[str] = None,
        capture_quality: Optional[CaptureQuality] = None,
        capture_quality_details: Optional[str] = None,
        warning_acknowledged: Optional[bool] = None,
    ) -> ClinicalVisit:
        visit = self.get_visit(visit_id)
        if visit.status == "finalized":
            raise ClinicalConflictError("Finalized visits cannot be edited")
        if capture_quality is not None and capture_quality not in {
            "acceptable",
            "accepted_with_warning",
            "repeat_required",
        }:
            raise ClinicalValidationError("Unknown capture quality")
        updates: list[str] = []
        values: list[object] = []
        fields = {
            "training_session_id": training_session_id,
            "specialist_observation": (specialist_observation.strip() if specialist_observation is not None else None),
            "capture_quality": capture_quality,
            "capture_quality_details": (
                capture_quality_details.strip() if capture_quality_details is not None else None
            ),
            "warning_acknowledged": (int(warning_acknowledged) if warning_acknowledged is not None else None),
        }
        for column, value in fields.items():
            if value is not None:
                updates.append(f"{column} = ?")
                values.append(value)
        if not updates:
            return visit
        updates.append("updated_at = ?")
        values.extend((_utc_now(), visit_id))
        with self._connection() as connection:
            connection.execute(
                f"UPDATE clinical_visits SET {', '.join(updates)} WHERE id = ?",
                tuple(values),
            )
        return self.get_visit(visit_id)

    def finalize_visit(self, visit_id: int) -> ClinicalVisit:
        visit = self.get_visit(visit_id)
        if visit.status == "finalized":
            return visit
        if visit.training_session_id is None:
            raise ClinicalValidationError("A saved rehabilitation session is required")
        if visit.capture_quality not in {
            "acceptable",
            "accepted_with_warning",
        }:
            raise ClinicalValidationError("Capture quality does not allow finalization")
        if visit.capture_quality == "accepted_with_warning" and not visit.warning_acknowledged:
            raise ClinicalValidationError("Capture warning must be acknowledged")
        if not visit.specialist_observation.strip():
            raise ClinicalValidationError("Specialist observation is required")
        with self._connection() as connection:
            matching_session = connection.execute(
                """
                SELECT sessions.id
                FROM sessions
                JOIN patient_profiles
                    ON patient_profiles.athlete_id = sessions.athlete_id
                JOIN rehab_episodes
                    ON rehab_episodes.patient_profile_id = patient_profiles.id
                WHERE sessions.id = ?
                    AND rehab_episodes.id = ?
                """,
                (visit.training_session_id, visit.rehab_episode_id),
            ).fetchone()
            if matching_session is None:
                raise ClinicalValidationError("Session does not belong to the episode patient")
            connection.execute(
                """
                UPDATE clinical_visits
                SET status = 'finalized', updated_at = ?
                WHERE id = ?
                """,
                (_utc_now(), visit_id),
            )
        return self.get_visit(visit_id)
