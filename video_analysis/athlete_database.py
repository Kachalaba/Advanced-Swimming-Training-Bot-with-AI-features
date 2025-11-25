"""
📊 Athlete Database for Training History

Features:
- SQLite database for athletes and sessions
- Store analysis results
- Progress tracking over time
- Session comparison
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "athletes.db"


@dataclass
class Athlete:
    """Athlete profile."""
    id: Optional[int] = None
    name: str = ""
    age: Optional[int] = None
    level: str = "amateur"  # amateur, intermediate, professional
    specialization: str = ""  # freestyle, backstroke, breaststroke, butterfly
    created_at: str = ""
    notes: str = ""


@dataclass
class TrainingSession:
    """Training session record."""
    id: Optional[int] = None
    athlete_id: int = 0
    session_type: str = "swimming"  # swimming, dryland
    date: str = ""
    duration_sec: float = 0
    distance_m: float = 0
    
    # Swimming metrics
    avg_speed: float = 0
    stroke_rate: float = 0
    stroke_count: int = 0
    symmetry_score: float = 0
    body_roll: float = 0
    streamline_score: float = 0
    
    # Dryland metrics
    reps: int = 0
    exercise_type: str = ""
    avg_tempo: float = 0
    stability_score: float = 0
    
    # AI coaching
    ai_score: int = 0
    ai_summary: str = ""
    
    # Raw data (JSON)
    full_analysis: str = ""
    video_path: str = ""
    
    notes: str = ""


class AthleteDatabase:
    """SQLite database for athletes and training sessions."""
    
    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Athletes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS athletes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                age INTEGER,
                level TEXT DEFAULT 'amateur',
                specialization TEXT,
                created_at TEXT,
                notes TEXT
            )
        """)
        
        # Training sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                athlete_id INTEGER NOT NULL,
                session_type TEXT NOT NULL,
                date TEXT NOT NULL,
                duration_sec REAL,
                distance_m REAL,
                
                avg_speed REAL,
                stroke_rate REAL,
                stroke_count INTEGER,
                symmetry_score REAL,
                body_roll REAL,
                streamline_score REAL,
                
                reps INTEGER,
                exercise_type TEXT,
                avg_tempo REAL,
                stability_score REAL,
                
                ai_score INTEGER,
                ai_summary TEXT,
                
                full_analysis TEXT,
                video_path TEXT,
                notes TEXT,
                
                FOREIGN KEY (athlete_id) REFERENCES athletes(id)
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    # ========================================================================
    # ATHLETES
    # ========================================================================
    
    def add_athlete(self, athlete: Athlete) -> int:
        """Add new athlete. Returns athlete ID."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO athletes (name, age, level, specialization, created_at, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                athlete.name,
                athlete.age,
                athlete.level,
                athlete.specialization,
                datetime.now().isoformat(),
                athlete.notes,
            ))
            conn.commit()
            athlete_id = cursor.lastrowid
            logger.info(f"Added athlete: {athlete.name} (ID: {athlete_id})")
            return athlete_id
        except sqlite3.IntegrityError:
            # Athlete already exists, get their ID
            cursor.execute("SELECT id FROM athletes WHERE name = ?", (athlete.name,))
            result = cursor.fetchone()
            return result[0] if result else -1
        finally:
            conn.close()
    
    def get_athlete(self, athlete_id: int = None, name: str = None) -> Optional[Athlete]:
        """Get athlete by ID or name."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        if athlete_id:
            cursor.execute("SELECT * FROM athletes WHERE id = ?", (athlete_id,))
        elif name:
            cursor.execute("SELECT * FROM athletes WHERE name = ?", (name,))
        else:
            conn.close()
            return None
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return Athlete(
                id=row[0], name=row[1], age=row[2], level=row[3],
                specialization=row[4], created_at=row[5], notes=row[6]
            )
        return None
    
    def get_all_athletes(self) -> List[Athlete]:
        """Get all athletes."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM athletes ORDER BY name")
        rows = cursor.fetchall()
        conn.close()
        
        return [
            Athlete(id=r[0], name=r[1], age=r[2], level=r[3],
                   specialization=r[4], created_at=r[5], notes=r[6])
            for r in rows
        ]
    
    def update_athlete(self, athlete: Athlete) -> bool:
        """Update athlete."""
        if not athlete.id:
            return False
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE athletes SET name=?, age=?, level=?, specialization=?, notes=?
            WHERE id=?
        """, (athlete.name, athlete.age, athlete.level, 
              athlete.specialization, athlete.notes, athlete.id))
        conn.commit()
        conn.close()
        return True
    
    def delete_athlete(self, athlete_id: int) -> bool:
        """Delete athlete and their sessions."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sessions WHERE athlete_id = ?", (athlete_id,))
        cursor.execute("DELETE FROM athletes WHERE id = ?", (athlete_id,))
        conn.commit()
        conn.close()
        return True
    
    # ========================================================================
    # SESSIONS
    # ========================================================================
    
    def add_session(self, session: TrainingSession) -> int:
        """Add training session. Returns session ID."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO sessions (
                athlete_id, session_type, date, duration_sec, distance_m,
                avg_speed, stroke_rate, stroke_count, symmetry_score, body_roll, streamline_score,
                reps, exercise_type, avg_tempo, stability_score,
                ai_score, ai_summary, full_analysis, video_path, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session.athlete_id, session.session_type, session.date or datetime.now().isoformat(),
            session.duration_sec, session.distance_m,
            session.avg_speed, session.stroke_rate, session.stroke_count,
            session.symmetry_score, session.body_roll, session.streamline_score,
            session.reps, session.exercise_type, session.avg_tempo, session.stability_score,
            session.ai_score, session.ai_summary, session.full_analysis, session.video_path, session.notes
        ))
        
        conn.commit()
        session_id = cursor.lastrowid
        conn.close()
        
        logger.info(f"Added session {session_id} for athlete {session.athlete_id}")
        return session_id
    
    def get_sessions(self, athlete_id: int, session_type: str = None, limit: int = 50) -> List[TrainingSession]:
        """Get sessions for athlete."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        if session_type:
            cursor.execute("""
                SELECT * FROM sessions WHERE athlete_id = ? AND session_type = ?
                ORDER BY date DESC LIMIT ?
            """, (athlete_id, session_type, limit))
        else:
            cursor.execute("""
                SELECT * FROM sessions WHERE athlete_id = ?
                ORDER BY date DESC LIMIT ?
            """, (athlete_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_session(r) for r in rows]
    
    def get_session(self, session_id: int) -> Optional[TrainingSession]:
        """Get single session by ID."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        conn.close()
        return self._row_to_session(row) if row else None
    
    def _row_to_session(self, row) -> TrainingSession:
        """Convert database row to TrainingSession."""
        return TrainingSession(
            id=row[0], athlete_id=row[1], session_type=row[2], date=row[3],
            duration_sec=row[4], distance_m=row[5],
            avg_speed=row[6], stroke_rate=row[7], stroke_count=row[8],
            symmetry_score=row[9], body_roll=row[10], streamline_score=row[11],
            reps=row[12], exercise_type=row[13], avg_tempo=row[14], stability_score=row[15],
            ai_score=row[16], ai_summary=row[17], full_analysis=row[18],
            video_path=row[19], notes=row[20]
        )
    
    def delete_session(self, session_id: int) -> bool:
        """Delete session."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        conn.close()
        return True
    
    # ========================================================================
    # PROGRESS & ANALYTICS
    # ========================================================================
    
    def get_progress(self, athlete_id: int, metric: str = "ai_score", 
                    session_type: str = None, limit: int = 20) -> List[Dict]:
        """Get progress data for a metric over time."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        if session_type:
            cursor.execute(f"""
                SELECT date, {metric} FROM sessions 
                WHERE athlete_id = ? AND session_type = ?
                ORDER BY date ASC LIMIT ?
            """, (athlete_id, session_type, limit))
        else:
            cursor.execute(f"""
                SELECT date, {metric} FROM sessions 
                WHERE athlete_id = ?
                ORDER BY date ASC LIMIT ?
            """, (athlete_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [{"date": r[0], "value": r[1]} for r in rows]
    
    def get_athlete_stats(self, athlete_id: int) -> Dict:
        """Get overall statistics for athlete."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Count sessions
        cursor.execute("""
            SELECT session_type, COUNT(*) as count,
                   AVG(ai_score) as avg_score,
                   MAX(ai_score) as best_score
            FROM sessions WHERE athlete_id = ?
            GROUP BY session_type
        """, (athlete_id,))
        
        type_stats = {}
        for row in cursor.fetchall():
            type_stats[row[0]] = {
                "count": row[1],
                "avg_score": round(row[2] or 0, 1),
                "best_score": row[3] or 0
            }
        
        # Total stats
        cursor.execute("""
            SELECT COUNT(*), AVG(ai_score), MAX(ai_score), 
                   SUM(duration_sec), SUM(distance_m)
            FROM sessions WHERE athlete_id = ?
        """, (athlete_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        return {
            "total_sessions": row[0] or 0,
            "avg_score": round(row[1] or 0, 1),
            "best_score": row[2] or 0,
            "total_time_min": round((row[3] or 0) / 60, 1),
            "total_distance_m": round(row[4] or 0, 1),
            "by_type": type_stats
        }
    
    def compare_sessions(self, session_id_1: int, session_id_2: int) -> Dict:
        """Compare two sessions."""
        s1 = self.get_session(session_id_1)
        s2 = self.get_session(session_id_2)
        
        if not s1 or not s2:
            return {}
        
        comparison = {
            "session_1": asdict(s1),
            "session_2": asdict(s2),
            "improvements": [],
            "regressions": [],
        }
        
        metrics = [
            ("ai_score", "AI Score", True),  # higher is better
            ("avg_speed", "Швидкість", True),
            ("symmetry_score", "Симетрія", True),
            ("streamline_score", "Streamline", True),
            ("stability_score", "Стабільність", True),
        ]
        
        for metric, name, higher_better in metrics:
            v1 = getattr(s1, metric, 0) or 0
            v2 = getattr(s2, metric, 0) or 0
            diff = v2 - v1
            
            if abs(diff) > 0.5:
                if (diff > 0) == higher_better:
                    comparison["improvements"].append(f"{name}: +{diff:.1f}")
                else:
                    comparison["regressions"].append(f"{name}: {diff:.1f}")
        
        return comparison


# Singleton instance
_db_instance: Optional[AthleteDatabase] = None

def get_database() -> AthleteDatabase:
    """Get database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = AthleteDatabase()
    return _db_instance


def save_analysis_to_db(
    athlete_name: str,
    session_type: str,
    analysis: Dict,
    ai_advice: Any = None,
    video_path: str = ""
) -> int:
    """
    Convenience function to save analysis results to database.
    
    Returns session ID.
    """
    db = get_database()
    
    # Get or create athlete
    athlete = db.get_athlete(name=athlete_name)
    if not athlete:
        athlete_id = db.add_athlete(Athlete(name=athlete_name))
    else:
        athlete_id = athlete.id
    
    # Extract metrics
    session = TrainingSession(
        athlete_id=athlete_id,
        session_type=session_type,
        date=datetime.now().isoformat(),
        video_path=video_path,
        full_analysis=json.dumps(analysis, default=str),
    )
    
    if session_type == "swimming":
        summary = analysis.get("summary", {})
        session.duration_sec = summary.get("total_time_s", 0)
        session.distance_m = summary.get("total_distance_m", 0)
        session.avg_speed = summary.get("avg_speed_ms", 0)
        
        biomech = analysis.get("biomechanics", {})
        stroke = biomech.get("stroke_analysis")
        if stroke:
            session.stroke_rate = getattr(stroke, 'stroke_rate', 0) if hasattr(stroke, 'stroke_rate') else stroke.get('stroke_rate', 0)
            session.stroke_count = getattr(stroke, 'total_strokes', 0) if hasattr(stroke, 'total_strokes') else stroke.get('total_strokes', 0)
            session.symmetry_score = getattr(stroke, 'symmetry_score', 0) if hasattr(stroke, 'symmetry_score') else stroke.get('symmetry_score', 0)
            session.body_roll = getattr(stroke, 'avg_body_roll', 0) if hasattr(stroke, 'avg_body_roll') else stroke.get('avg_body_roll', 0)
        
        swimming_pose = biomech.get("swimming_pose", {})
        session.streamline_score = swimming_pose.get("avg_streamline", 0)
    
    elif session_type == "dryland":
        exercise_stats = analysis.get("exercise_stats")
        if exercise_stats:
            session.reps = getattr(exercise_stats, 'total_reps', 0)
            session.avg_tempo = getattr(exercise_stats, 'avg_tempo', 0)
            session.stability_score = getattr(exercise_stats, 'stability_score', 0)
        session.exercise_type = analysis.get("main_movement", "")
    
    if ai_advice:
        session.ai_score = getattr(ai_advice, 'score', 0)
        session.ai_summary = getattr(ai_advice, 'summary', '')
    
    return db.add_session(session)
