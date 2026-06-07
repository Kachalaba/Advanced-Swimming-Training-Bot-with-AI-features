"""Tests for the confidence-aware freestyle technique contract."""

from __future__ import annotations

from typing import Dict, Optional

from video_analysis.swimming_technique_analyzer import ConfidenceInputs, SwimmingTechniqueAnalyzer

ZONE_IDS = ("body_position", "rotation", "catch", "breathing", "kick")


def _cycle(
    cycle_id: str,
    *,
    body_issue: bool,
    catch_available: bool = True,
    peak_sec: float = 2.4,
) -> Dict:
    zones = {}
    for zone_id in ZONE_IDS:
        available = catch_available or zone_id != "catch"
        issue_code: Optional[str] = "hips_drop" if zone_id == "body_position" and body_issue else None
        zones[zone_id] = {
            "available": available,
            "prerequisites_met": available,
            "confidence_inputs": {
                "landmark_visibility": 0.9,
                "temporal_continuity": 0.9,
                "waterline_clarity": 0.8,
                "identity_stability": 0.9,
                "cycle_coverage": 1.0,
            },
            "score": 62.0 if issue_code else 88.0,
            "issue_code": issue_code,
            "impact": 0.9 if issue_code else 0.0,
            "metrics": {"hip_drop_deg": 11.2} if zone_id == "body_position" else {},
        }
    return {
        "cycle_id": cycle_id,
        "start_sec": peak_sec - 0.5,
        "peak_sec": peak_sec,
        "end_sec": peak_sec + 0.5,
        "zones": zones,
    }


def test_metric_confidence_uses_agreed_weights():
    analyzer = SwimmingTechniqueAnalyzer()

    confidence = analyzer.metric_confidence(
        ConfidenceInputs(
            landmark_visibility=1.0,
            temporal_continuity=0.8,
            waterline_clarity=0.6,
            identity_stability=1.0,
            cycle_coverage=0.5,
        )
    )

    assert confidence == 0.84
    assert analyzer.confidence_level(confidence) == "high"


def test_confidence_below_hard_prerequisite_is_insufficient():
    analyzer = SwimmingTechniqueAnalyzer()

    confidence = analyzer.metric_confidence(
        ConfidenceInputs(
            landmark_visibility=1.0,
            temporal_continuity=1.0,
            waterline_clarity=1.0,
            identity_stability=1.0,
            cycle_coverage=1.0,
        ),
        prerequisites_met=False,
    )

    assert confidence == 0.0
    assert analyzer.confidence_level(confidence) == "insufficient"


def test_missing_zone_is_excluded_from_coverage_and_score():
    result = SwimmingTechniqueAnalyzer().build_result(
        cycle_metrics=[
            _cycle("cycle-1", body_issue=True, catch_available=False),
            _cycle("cycle-2", body_issue=True, catch_available=False),
        ]
    )

    assert result["coverage"] == {"available_zones": 4, "total_zones": 5}
    catch = next(zone for zone in result["zones"] if zone["id"] == "catch")
    assert catch["status"] == "insufficient_data"
    assert catch["score"] is None
    assert result["overall_score"] == 81.5


def test_primary_issue_requires_two_confirming_cycles():
    result = SwimmingTechniqueAnalyzer().build_result(
        cycle_metrics=[
            _cycle("cycle-1", body_issue=True),
            _cycle("cycle-2", body_issue=False),
        ]
    )

    assert result["primary_issue"] is None
    assert result["prescription"] is None


def test_primary_issue_includes_evidence_and_prescription():
    result = SwimmingTechniqueAnalyzer().build_result(
        cycle_metrics=[
            _cycle("cycle-1", body_issue=True, peak_sec=2.4),
            _cycle("cycle-2", body_issue=True, peak_sec=4.8),
        ]
    )

    assert result["primary_issue"]["zone_id"] == "body_position"
    assert result["primary_issue"]["issue_code"] == "hips_drop"
    assert result["primary_issue"]["evidence"][0]["peak_sec"] == 2.4
    assert result["primary_issue"]["confirming_cycles"] == 2
    assert result["prescription"]["drill"]["name"]
    assert result["prescription"]["mini_set"]["repetitions"] > 0
