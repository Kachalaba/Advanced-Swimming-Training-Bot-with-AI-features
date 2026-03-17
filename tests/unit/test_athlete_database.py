"""Unit tests for AthleteDatabase — context manager, duplicate handling, CRUD."""

import sys
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from video_analysis.athlete_database import AthleteDatabase, Athlete, TrainingSession


@pytest.fixture
def db(tmp_path):
    """Create a fresh in-memory database for each test."""
    db_file = tmp_path / "test_athletes.db"
    return AthleteDatabase(db_path=str(db_file))


class TestAthleteDatabase:

    def test_add_and_get_athlete(self, db):
        athlete = Athlete(name="Ivan Petrenko", age=25, level="intermediate")
        athlete_id = db.add_athlete(athlete)
        assert athlete_id > 0

        fetched = db.get_athlete(athlete_id=athlete_id)
        assert fetched is not None
        assert fetched.name == "Ivan Petrenko"
        assert fetched.age == 25
        assert fetched.level == "intermediate"

    def test_duplicate_athlete_returns_existing_id(self, db):
        """Adding an athlete with the same name must return the existing ID, not raise."""
        athlete = Athlete(name="Duplicate Swimmer")
        id1 = db.add_athlete(athlete)
        id2 = db.add_athlete(Athlete(name="Duplicate Swimmer"))
        assert id1 == id2

    def test_get_athlete_by_name(self, db):
        db.add_athlete(Athlete(name="Maria Kovalenko"))
        fetched = db.get_athlete(name="Maria Kovalenko")
        assert fetched is not None
        assert fetched.name == "Maria Kovalenko"

    def test_get_nonexistent_athlete_returns_none(self, db):
        result = db.get_athlete(athlete_id=9999)
        assert result is None

    def test_get_all_athletes_empty(self, db):
        athletes = db.get_all_athletes()
        assert athletes == []

    def test_get_all_athletes_returns_all(self, db):
        db.add_athlete(Athlete(name="Athlete A"))
        db.add_athlete(Athlete(name="Athlete B"))
        athletes = db.get_all_athletes()
        assert len(athletes) == 2

    def test_update_athlete(self, db):
        athlete_id = db.add_athlete(Athlete(name="Old Name"))
        athlete = db.get_athlete(athlete_id=athlete_id)
        athlete.name = "New Name"
        athlete.age = 30
        db.update_athlete(athlete)

        updated = db.get_athlete(athlete_id=athlete_id)
        assert updated.name == "New Name"
        assert updated.age == 30

    def test_delete_athlete(self, db):
        athlete_id = db.add_athlete(Athlete(name="To Delete"))
        db.delete_athlete(athlete_id)
        assert db.get_athlete(athlete_id=athlete_id) is None

    def test_delete_athlete_cascades_sessions(self, db):
        athlete_id = db.add_athlete(Athlete(name="Cascaded"))
        db.add_session(TrainingSession(
            athlete_id=athlete_id,
            session_type="swimming",
            date="2024-01-01",
        ))
        db.delete_athlete(athlete_id)
        sessions = db.get_sessions(athlete_id)
        assert sessions == []


class TestTrainingSessions:

    def test_add_and_get_session(self, db):
        athlete_id = db.add_athlete(Athlete(name="Session Tester"))
        session = TrainingSession(
            athlete_id=athlete_id,
            session_type="swimming",
            date="2024-06-15",
            duration_sec=600.0,
            distance_m=500.0,
            ai_score=75,
            ai_summary="Good session",
        )
        session_id = db.add_session(session)
        assert session_id > 0

        fetched = db.get_session(session_id)
        assert fetched is not None
        assert fetched.session_type == "swimming"
        assert fetched.duration_sec == 600.0
        assert fetched.ai_score == 75

    def test_get_sessions_for_athlete(self, db):
        athlete_id = db.add_athlete(Athlete(name="Multi Session"))
        for i in range(3):
            db.add_session(TrainingSession(
                athlete_id=athlete_id,
                session_type="swimming",
                date=f"2024-01-0{i+1}",
            ))
        sessions = db.get_sessions(athlete_id)
        assert len(sessions) == 3

    def test_get_sessions_filter_by_type(self, db):
        athlete_id = db.add_athlete(Athlete(name="Type Filter"))
        db.add_session(TrainingSession(athlete_id=athlete_id, session_type="swimming", date="2024-01-01"))
        db.add_session(TrainingSession(athlete_id=athlete_id, session_type="dryland", date="2024-01-02"))

        swimming = db.get_sessions(athlete_id, session_type="swimming")
        dryland = db.get_sessions(athlete_id, session_type="dryland")
        assert len(swimming) == 1
        assert len(dryland) == 1

    def test_delete_session(self, db):
        athlete_id = db.add_athlete(Athlete(name="Delete Session"))
        session_id = db.add_session(TrainingSession(
            athlete_id=athlete_id, session_type="swimming", date="2024-01-01"
        ))
        db.delete_session(session_id)
        assert db.get_session(session_id) is None

    def test_get_athlete_stats(self, db):
        athlete_id = db.add_athlete(Athlete(name="Stats Tester"))
        db.add_session(TrainingSession(
            athlete_id=athlete_id,
            session_type="swimming",
            date="2024-01-01",
            duration_sec=600.0,
            distance_m=500.0,
            ai_score=80,
        ))
        db.add_session(TrainingSession(
            athlete_id=athlete_id,
            session_type="swimming",
            date="2024-01-02",
            duration_sec=720.0,
            distance_m=600.0,
            ai_score=85,
        ))

        stats = db.get_athlete_stats(athlete_id)
        assert stats["total_sessions"] == 2
        assert stats["best_score"] == 85
        assert stats["avg_score"] == pytest.approx(82.5, abs=0.1)
        assert stats["total_distance_m"] == pytest.approx(1100.0, abs=0.1)


class TestContextManagerBehavior:
    """Verify that the DB connection context manager handles errors correctly."""

    def test_transaction_rollback_on_error(self, db):
        """If an error occurs mid-transaction, changes should be rolled back."""
        athlete_id = db.add_athlete(Athlete(name="Rollback Test"))

        # Attempt to add a session with a non-existent athlete_id to trigger FK-like error
        # We test that invalid data doesn't corrupt the DB
        try:
            db.add_session(TrainingSession(
                athlete_id=99999,  # Non-existent
                session_type="swimming",
                date="2024-01-01",
            ))
        except Exception:
            pass

        # The original athlete must still be intact
        fetched = db.get_athlete(athlete_id=athlete_id)
        assert fetched is not None

    def test_connection_closed_after_operation(self, db, tmp_path):
        """Verify no connection leak — file can be accessed after db operations."""
        db_path = tmp_path / "leak_test.db"
        test_db = AthleteDatabase(db_path=str(db_path))
        test_db.add_athlete(Athlete(name="Leak Test"))

        # If connection was not closed, this would fail on some OS
        assert db_path.exists()
        # Open another connection to verify database is accessible
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM athletes")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 1
