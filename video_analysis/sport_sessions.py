"""Normalize persisted sport analyses for API and frontend consumers."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, Optional

from .athlete_database import TrainingSession

SUPPORTED_SPORTS = frozenset({"swimming", "running", "cycling", "dryland"})


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 1) if math.isfinite(number) else None


def _metric(
    key: str,
    label: str,
    value: Any,
    unit: str = "",
    *,
    higher_is_better: Optional[bool] = None,
) -> Optional[tuple[str, Dict[str, Any]]]:
    number = _number(value)
    if number is None:
        return None
    return (
        key,
        {
            "key": key,
            "label": label,
            "value": number,
            "unit": unit,
            "higher_is_better": higher_is_better,
        },
    )


def _pose_quality(payload: Dict[str, Any]) -> Dict[str, float]:
    total = _number(payload.get("frames_total"))
    pose = _number(payload.get("frames_with_pose"))
    if total is None or pose is None or total <= 0:
        return {}
    return {"pose_coverage": round(min(100.0, max(0.0, pose / total * 100)), 1)}


def _base_session(session: TrainingSession) -> Dict[str, Any]:
    return {
        "id": int(session.id or 0),
        "date": session.date,
        "duration_sec": float(session.duration_sec or 0),
        "score": None,
        "summary": session.ai_summary or session.notes or "",
        "has_video": bool(session.video_path),
        "metrics": {},
        "quality": {},
        "insights": [],
    }


def _stored_payload(session: TrainingSession) -> Dict[str, Any]:
    try:
        stored = json.loads(session.full_analysis or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return stored if isinstance(stored, dict) else {}


def _running_session(session: TrainingSession, stored: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _base_session(session)
    result = stored.get("running_analysis", stored)
    if not isinstance(result, dict):
        return normalized
    analysis = result.get("analysis", result)
    if not isinstance(analysis, dict):
        return normalized

    pairs = [
        _metric("cadence", "Cadence", analysis.get("cadence"), "spm", higher_is_better=None),
        _metric(
            "arm_symmetry",
            "Arm symmetry",
            analysis.get("arm_symmetry"),
            "%",
            higher_is_better=True,
        ),
        _metric(
            "knee_lift",
            "Knee lift",
            analysis.get("avg_knee_lift"),
            "deg",
            higher_is_better=None,
        ),
        _metric(
            "forward_lean",
            "Forward lean",
            analysis.get("forward_lean"),
            "deg",
            higher_is_better=None,
        ),
        _metric(
            "vertical_oscillation",
            "Vertical movement",
            analysis.get("avg_vertical_osc", analysis.get("vertical_oscillation_px")),
            "px",
            higher_is_better=False,
        ),
        _metric(
            "efficiency_score",
            "Efficiency",
            analysis.get("efficiency_score"),
            "/100",
            higher_is_better=True,
        ),
        _metric(
            "injury_risk_score",
            "Movement risk flags",
            analysis.get("injury_risk_score"),
            "/100",
            higher_is_better=False,
        ),
    ]
    normalized["metrics"] = {key: value for pair in pairs if pair for key, value in [pair]}
    normalized["score"] = _number(analysis.get("efficiency_score"))
    normalized["quality"] = _pose_quality(result)

    cadence = _number(analysis.get("cadence"))
    strike = str(analysis.get("foot_strike_type") or "").strip()
    summary_parts = []
    if cadence is not None:
        summary_parts.append(f"{cadence:.0f} spm")
    if strike:
        summary_parts.append(f"{strike.title()} strike")
    if summary_parts:
        normalized["summary"] = " · ".join(summary_parts)

    insights = []
    if analysis.get("arm_crossover_detected"):
        insights.append(
            {
                "code": "arm_crossover",
                "level": "warning",
                "title": "Arm crosses the body midline",
                "detail": "Review the annotated video and cue the hands to travel forward and back.",
            }
        )
    if analysis.get("overstriding_detected"):
        insights.append(
            {
                "code": "overstriding",
                "level": "warning",
                "title": "Foot lands ahead of the hip",
                "detail": "Use cadence and contact position together before changing technique.",
            }
        )
    if cadence is not None and cadence < 165:
        insights.append(
            {
                "code": "low_cadence",
                "level": "info",
                "title": "Cadence is below the common training range",
                "detail": "Treat cadence as context, not a universal target; compare it at the same pace.",
            }
        )
    if not insights and normalized["score"] is not None:
        insights.append(
            {
                "code": "stable_form",
                "level": "success",
                "title": "No major repeated flag in this clip",
                "detail": "Use the saved session as a baseline for the next comparable recording.",
            }
        )
    normalized["insights"] = insights
    return normalized


def _swimming_session(session: TrainingSession, stored: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _base_session(session)
    analysis = stored.get("swimming_analysis", stored)
    if not isinstance(analysis, dict):
        return normalized

    coverage = analysis.get("coverage")
    metrics: Dict[str, Any] = {}
    if isinstance(coverage, dict):
        available = _number(coverage.get("available_zones"))
        total = _number(coverage.get("total_zones"))
        if available is not None and total and total > 0:
            pair = _metric(
                "zone_coverage",
                "Technique coverage",
                available / total * 100,
                "%",
                higher_is_better=True,
            )
            if pair:
                metrics[pair[0]] = pair[1]

    zones = analysis.get("zones")
    if isinstance(zones, list):
        for zone in zones:
            if not isinstance(zone, dict) or not zone.get("id"):
                continue
            pair = _metric(
                str(zone["id"]),
                str(zone["id"]).replace("_", " ").title(),
                zone.get("score"),
                "/100",
                higher_is_better=True,
            )
            if pair:
                metrics[pair[0]] = pair[1]

    normalized["metrics"] = metrics
    normalized["score"] = _number(analysis.get("overall_score"))
    normalized["quality"] = _pose_quality(analysis)
    issue = analysis.get("primary_issue")
    if isinstance(issue, dict):
        title = str(issue.get("title") or "").strip()
        if title:
            normalized["summary"] = title
            normalized["insights"] = [
                {
                    "code": str(issue.get("code") or issue.get("issue_code") or "primary_issue"),
                    "level": "warning",
                    "title": title,
                    "detail": "Open the saved analysis to review the supporting stroke-cycle evidence.",
                }
            ]
    return normalized


def normalize_sport_session(
    session: TrainingSession,
    sport: str,
) -> Optional[Dict[str, Any]]:
    """Return one normalized session when it belongs to the requested sport."""

    if sport not in SUPPORTED_SPORTS or session.session_type != sport:
        return None
    stored = _stored_payload(session)
    if sport == "running":
        return _running_session(session, stored)
    if sport == "swimming":
        return _swimming_session(session, stored)
    return _base_session(session)


def build_sport_overview(
    sport: str,
    sessions: Iterable[TrainingSession],
) -> Dict[str, Any]:
    """Build a stable newest-first sport overview from persisted sessions."""

    if sport not in SUPPORTED_SPORTS:
        raise ValueError(f"Unsupported sport: {sport}")
    normalized = [item for session in sessions if (item := normalize_sport_session(session, sport)) is not None]
    normalized.sort(key=lambda item: item["date"], reverse=True)
    latest = next((item for item in normalized if item["metrics"]), None)
    score_series = [
        {"date": item["date"], "value": item["score"]} for item in reversed(normalized) if item["score"] is not None
    ]
    return {
        "sport": sport,
        "total_sessions": len(normalized),
        "latest_session_date": normalized[0]["date"] if normalized else None,
        "latest_score": latest["score"] if latest else None,
        "headline_metrics": latest["metrics"] if latest else {},
        "insights": latest["insights"] if latest else [],
        "score_series": score_series,
        "sessions": normalized,
    }
