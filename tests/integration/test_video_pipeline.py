"""Integration tests for the video analysis pipeline using synthetic frames."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import pytest

# ---------------------------------------------------------------------------
# Helpers to generate synthetic test frames
# ---------------------------------------------------------------------------


def _create_synthetic_pool_frame(
    output_path: str,
    width: int = 640,
    height: int = 480,
    person_x: int = 320,
    person_y: int = 240,
) -> str:
    """Create a synthetic swimming pool frame with a rectangular 'person'."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Blue pool background
    frame[:, :] = [139, 90, 40]  # BGR: brownish-blue
    # Lane lines (white horizontal lines)
    for y in range(0, height, 60):
        cv2.line(frame, (0, y), (width, y), (200, 200, 200), 1)
    # Draw a person-shaped rectangle
    x1, y1 = person_x - 20, person_y - 50
    x2, y2 = person_x + 20, person_y + 50
    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 180, 160), -1)
    cv2.imwrite(output_path, frame)
    return output_path


def _create_frame_sequence(frames_dir: Path, n_frames: int = 10) -> list:
    """Create n_frames synthetic frames and return their info dicts."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_infos = []
    for i in range(n_frames):
        frame_path = str(frames_dir / f"frame_{i:04d}.jpg")
        person_x = 100 + i * 40  # Swimmer moving right
        _create_synthetic_pool_frame(frame_path, person_x=person_x)
        frame_infos.append(
            {
                "path": frame_path,
                "timestamp": i / 10.0,
                "video_frame": i,
            }
        )
    return frame_infos


# ---------------------------------------------------------------------------
# SwimmerDetector tests (requires YOLO — skipped if not available)
# ---------------------------------------------------------------------------


class TestSwimmerDetector:

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("ultralytics"),
        reason="ultralytics not installed",
    )
    def test_detect_on_synthetic_frame(self, tmp_path):
        """SwimmerDetector must not raise on a valid image."""
        from video_analysis.swimmer_detector import SwimmerDetector

        frame_path = str(tmp_path / "test_frame.jpg")
        _create_synthetic_pool_frame(frame_path)

        detector = SwimmerDetector()
        result = detector.detect_swimmer(frame_path)

        assert "bbox" in result
        assert "confidence" in result
        assert "center" in result
        assert "lane" in result

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("ultralytics"),
        reason="ultralytics not installed",
    )
    def test_detect_batch_tracking(self, tmp_path):
        """detect_batch with tracking must return one result per frame."""
        from video_analysis.swimmer_detector import SwimmerDetector

        frame_infos = _create_frame_sequence(tmp_path / "frames", n_frames=5)
        frame_paths = [f["path"] for f in frame_infos]

        detector = SwimmerDetector()
        results = detector.detect_batch(frame_paths, enable_tracking=True)

        assert len(results) == 5
        for r in results:
            assert "bbox" in r
            assert "frame_index" in r

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("ultralytics"),
        reason="ultralytics not installed",
    )
    def test_yolo_model_cached(self, tmp_path):
        """YOLO model cache should reuse the same instance."""
        from video_analysis.swimmer_detector import _get_yolo_model

        model1 = _get_yolo_model("yolov8n.pt")
        model2 = _get_yolo_model("yolov8n.pt")
        assert model1 is model2


# ---------------------------------------------------------------------------
# StrokeAnalyzer pipeline integration
# ---------------------------------------------------------------------------


class TestStrokeAnalyzerPipeline:

    def test_analyze_returns_valid_structure(self):
        """Full analyze() call with realistic-shaped keypoints must succeed."""
        from video_analysis.stroke_analyzer import StrokeAnalyzer

        # Build 60 frames of basic keypoints
        frames = []
        for i in range(60):
            import math

            t = i / 10.0
            left_y = 0.5 + 0.3 * math.sin(2 * math.pi * 0.5 * t)
            right_y = 0.5 + 0.3 * math.sin(2 * math.pi * 0.5 * t + math.pi)
            frames.append(
                {
                    11: {"x": 0.3, "y": 0.5, "visibility": 0.9},
                    12: {"x": 0.7, "y": 0.5, "visibility": 0.9},
                    13: {"x": 0.25, "y": 0.5, "visibility": 0.9},
                    14: {"x": 0.75, "y": 0.5, "visibility": 0.9},
                    15: {"x": 0.2, "y": left_y, "visibility": 0.9},
                    16: {"x": 0.8, "y": right_y, "visibility": 0.9},
                    23: {"x": 0.35, "y": 0.7, "visibility": 0.9},
                    24: {"x": 0.65, "y": 0.7, "visibility": 0.9},
                    27: {"x": 0.35, "y": 0.9, "visibility": 0.9},
                    28: {"x": 0.65, "y": 0.9, "visibility": 0.9},
                    25: {"x": 0.35, "y": 0.8, "visibility": 0.9},
                    26: {"x": 0.65, "y": 0.8, "visibility": 0.9},
                    0: {"x": 0.5, "y": 0.1, "visibility": 0.9},
                    7: {"x": 0.45, "y": 0.1, "visibility": 0.9},
                    8: {"x": 0.55, "y": 0.1, "visibility": 0.9},
                }
            )

        analyzer = StrokeAnalyzer(fps=10.0)
        result = analyzer.analyze(frames, fps=10.0)

        assert result.total_strokes >= 0
        assert result.symmetry_score >= 0
        assert result.stroke_rate >= 0
        assert result.left_strokes + result.right_strokes == result.total_strokes


# ---------------------------------------------------------------------------
# AthleteDatabase + save_analysis_to_db pipeline
# ---------------------------------------------------------------------------


class TestDatabasePipeline:

    def test_save_analysis_creates_session(self, tmp_path):
        """save_analysis_to_db must create athlete and session records."""
        from video_analysis.athlete_database import AthleteDatabase, save_analysis_to_db

        db_path = str(tmp_path / "pipeline_test.db")
        # Patch the DB path by using the module directly
        import video_analysis.athlete_database as adb

        original_instance = adb._db_instance
        adb._db_instance = AthleteDatabase(db_path=db_path)

        try:
            analysis = {
                "summary": {
                    "total_time_s": 120.0,
                    "total_distance_m": 100.0,
                    "avg_speed_ms": 0.83,
                },
                "biomechanics": {},
            }

            from video_analysis.ai_coach import CoachingAdvice

            ai_advice = CoachingAdvice(
                summary="Good session",
                strengths=["Fast"],
                improvements=["Work on turns"],
                drills=[],
                priority="technique",
                score=75,
            )

            session_id = save_analysis_to_db(
                athlete_name="Pipeline Test Athlete",
                session_type="swimming",
                analysis=analysis,
                ai_advice=ai_advice,
                video_path="",
            )

            assert session_id > 0

            # Verify athlete and session exist
            db = adb._db_instance
            athlete = db.get_athlete(name="Pipeline Test Athlete")
            assert athlete is not None

            session = db.get_session(session_id)
            assert session is not None
            assert session.session_type == "swimming"
            assert session.ai_score == 75

        finally:
            adb._db_instance = original_instance

    def test_duplicate_athlete_pipeline_idempotent(self, tmp_path):
        """Calling save_analysis_to_db twice for same athlete must not duplicate athlete."""
        import video_analysis.athlete_database as adb
        from video_analysis.athlete_database import AthleteDatabase, save_analysis_to_db

        db_path = str(tmp_path / "dedup_test.db")
        original_instance = adb._db_instance
        adb._db_instance = AthleteDatabase(db_path=db_path)

        try:
            analysis = {"summary": {}, "biomechanics": {}}
            save_analysis_to_db("Same Athlete", "swimming", analysis)
            save_analysis_to_db("Same Athlete", "swimming", analysis)

            db = adb._db_instance
            athletes = db.get_all_athletes()
            same_name = [a for a in athletes if a.name == "Same Athlete"]
            assert len(same_name) == 1  # Only one athlete record

            sessions = db.get_sessions(same_name[0].id)
            assert len(sessions) == 2  # Two separate sessions

        finally:
            adb._db_instance = original_instance
