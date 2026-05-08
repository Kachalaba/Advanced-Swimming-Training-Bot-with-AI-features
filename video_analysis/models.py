"""
SQLAlchemy ORM models for athletes and training sessions.

These models mirror the schema defined in ``athlete_database.py`` (raw SQLite)
but expose a proper ORM layer with relationships, type annotations, and indexes.

Usage::

    from video_analysis.models import Base, AthleteModel, SessionModel, get_engine
    engine = get_engine()          # creates tables if absent
    with Session(engine) as s:
        athlete = s.get(AthleteModel, 1)

The existing ``AthleteDatabase`` class (raw SQLite) remains the active backend
for all pages. These models are the migration target and can be used alongside.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.pool import StaticPool


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class AthleteModel(Base):
    """Athlete profile."""

    __tablename__ = "athletes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    level: Mapped[str] = mapped_column(String(50), default="amateur")
    specialization: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    sessions: Mapped[List["SessionModel"]] = relationship(
        "SessionModel", back_populates="athlete", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_athletes_name", "name"),
    )

    def __repr__(self) -> str:
        return f"<AthleteModel id={self.id} name={self.name!r}>"


class SessionModel(Base):
    """Training session linked to an athlete."""

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    athlete_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("athletes.id", ondelete="CASCADE"), nullable=False
    )
    session_type: Mapped[str] = mapped_column(String(50), nullable=False)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    duration_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    distance_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Swimming metrics
    avg_speed: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stroke_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stroke_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    symmetry_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    body_roll: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    streamline_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Dryland / exercise metrics
    reps: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    exercise_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    avg_tempo: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stability_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # AI coaching
    ai_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ai_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Full JSON payload and media
    full_analysis: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    video_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    athlete: Mapped["AthleteModel"] = relationship("AthleteModel", back_populates="sessions")

    __table_args__ = (
        Index("ix_sessions_athlete_id", "athlete_id"),
        Index("ix_sessions_date", "date"),
        Index("ix_sessions_type_athlete", "session_type", "athlete_id"),
    )

    def __repr__(self) -> str:
        return f"<SessionModel id={self.id} type={self.session_type!r} athlete_id={self.athlete_id}>"

    def get_full_analysis(self) -> dict:
        """Deserialize full_analysis JSON, returning empty dict on failure."""
        if not self.full_analysis:
            return {}
        try:
            return json.loads(self.full_analysis)
        except (json.JSONDecodeError, TypeError):
            return {}


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

_DB_PATH = Path(os.getenv("ATHLETE_DB_PATH", "data/athletes_orm.db"))


def get_engine(db_url: str | None = None, echo: bool = False):
    """Return a SQLAlchemy engine, creating tables on first call.

    Args:
        db_url: SQLAlchemy database URL. Defaults to SQLite at ``data/athletes_orm.db``
                or the ``ATHLETE_DB_PATH`` env variable.
        echo:   Log all SQL statements (useful for debugging).
    """
    if db_url is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        db_url = f"sqlite:///{_DB_PATH}"

    connect_args = {}
    kwargs: dict = {}
    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        kwargs["poolclass"] = StaticPool

    engine = create_engine(db_url, echo=echo, connect_args=connect_args, **kwargs)

    # Enable WAL mode for SQLite (better concurrent read performance)
    if db_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _set_wal(dbapi_conn, _record):
            dbapi_conn.execute("PRAGMA journal_mode=WAL")
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    return engine
