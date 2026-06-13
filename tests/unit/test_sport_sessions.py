from __future__ import annotations

import json

from video_analysis.athlete_database import TrainingSession
from video_analysis.sport_sessions import build_sport_overview, normalize_sport_session


def _session(
    *,
    session_id: int = 1,
    session_type: str,
    payload: dict | str,
    date: str = "2026-06-14T08:00:00",
    video_path: str = "",
) -> TrainingSession:
    return TrainingSession(
        id=session_id,
        athlete_id=7,
        session_type=session_type,
        date=date,
        full_analysis=payload if isinstance(payload, str) else json.dumps(payload),
        video_path=video_path,
    )


def test_normalizes_running_analysis_into_headline_metrics_and_insights():
    session = _session(
        session_type="running",
        video_path="/tmp/running.mp4",
        payload={
            "running_analysis": {
                "type": "result",
                "analysis": {
                    "cadence": 176.4,
                    "foot_strike_type": "midfoot",
                    "arm_symmetry": 91.2,
                    "avg_knee_lift": 47.8,
                    "forward_lean": 10.5,
                    "avg_vertical_osc": 7.3,
                    "efficiency_score": 86.7,
                    "injury_risk_score": 15.0,
                    "overstriding_detected": False,
                    "arm_crossover_detected": True,
                },
                "frames_total": 120,
                "frames_with_pose": 108,
            }
        },
    )

    normalized = normalize_sport_session(session, "running")

    assert normalized is not None
    assert normalized["score"] == 86.7
    assert normalized["summary"] == "176 spm · Midfoot strike"
    assert normalized["has_video"] is True
    assert normalized["metrics"]["cadence"]["value"] == 176.4
    assert normalized["metrics"]["arm_symmetry"]["unit"] == "%"
    assert normalized["quality"]["pose_coverage"] == 90.0
    assert normalized["insights"][0]["code"] == "arm_crossover"


def test_normalizes_swimming_analysis_and_zone_scores():
    session = _session(
        session_type="swimming",
        payload={
            "swimming_analysis": {
                "analysis_type": "swimming_freestyle_side",
                "overall_score": 82.0,
                "coverage": {"available_zones": 4, "total_zones": 5},
                "primary_issue": {
                    "code": "rotation_asymmetry",
                    "title": "Torso rotation is asymmetric",
                },
                "zones": [
                    {"id": "body_position", "score": 88.0, "status": "good"},
                    {"id": "rotation", "score": 72.0, "status": "needs_attention"},
                ],
                "frames_total": 100,
                "frames_with_pose": 84,
            }
        },
    )

    normalized = normalize_sport_session(session, "swimming")

    assert normalized is not None
    assert normalized["score"] == 82.0
    assert normalized["summary"] == "Torso rotation is asymmetric"
    assert normalized["metrics"]["zone_coverage"]["value"] == 80.0
    assert normalized["metrics"]["body_position"]["value"] == 88.0
    assert normalized["quality"]["pose_coverage"] == 84.0


def test_malformed_json_is_kept_as_basic_session_without_invented_metrics():
    session = _session(
        session_type="running",
        payload="{broken",
        video_path="/tmp/running.mp4",
    )

    normalized = normalize_sport_session(session, "running")

    assert normalized == {
        "id": 1,
        "date": "2026-06-14T08:00:00",
        "duration_sec": 0.0,
        "score": None,
        "summary": "",
        "has_video": True,
        "metrics": {},
        "quality": {},
        "insights": [],
    }


def test_build_overview_uses_latest_valid_metrics_and_chronological_score_series():
    older = _session(
        session_id=1,
        session_type="running",
        date="2026-06-10T08:00:00",
        payload={
            "running_analysis": {
                "analysis": {
                    "cadence": 170,
                    "foot_strike_type": "heel",
                    "efficiency_score": 72,
                }
            }
        },
    )
    latest = _session(
        session_id=2,
        session_type="running",
        date="2026-06-14T08:00:00",
        payload={
            "running_analysis": {
                "analysis": {
                    "cadence": 176,
                    "foot_strike_type": "midfoot",
                    "efficiency_score": 86,
                }
            }
        },
    )

    overview = build_sport_overview("running", [latest, older])

    assert overview["total_sessions"] == 2
    assert overview["latest_session_date"] == latest.date
    assert overview["latest_score"] == 86.0
    assert overview["headline_metrics"]["cadence"]["value"] == 176.0
    assert overview["score_series"] == [
        {"date": older.date, "value": 72.0},
        {"date": latest.date, "value": 86.0},
    ]
    assert [item["id"] for item in overview["sessions"]] == [2, 1]
