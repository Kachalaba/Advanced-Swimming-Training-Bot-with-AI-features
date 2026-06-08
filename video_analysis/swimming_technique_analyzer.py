"""Confidence-aware aggregation for side-view freestyle technique analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from video_analysis.base_analyzer import BaseAnalyzer
from video_analysis.constants import SWIM_CONFIDENCE_HIGH, SWIM_CONFIDENCE_MEDIUM, SWIM_MIN_DIAGNOSIS_CYCLES


@dataclass(frozen=True)
class ConfidenceInputs:
    """Normalized evidence used to decide whether a metric is trustworthy."""

    landmark_visibility: float
    temporal_continuity: float
    waterline_clarity: float
    identity_stability: float
    cycle_coverage: float


_ISSUE_LIBRARY: Dict[str, Dict[str, Any]] = {
    "hips_drop": {
        "title": "Hips drop below the body line",
        "why_it_matters": "A lower hip position increases frontal drag and costs speed at the same effort.",
        "drill": {
            "name": "Side kick with rotation",
            "purpose": "Build a stable horizontal line led by the hips and trunk.",
            "execution": "Kick six beats on one side, rotate through the hips, then repeat on the other side.",
            "common_mistake": "Lifting the head to correct balance.",
            "success_cue": "Keep the crown long and feel the hips close to the surface.",
        },
        "mini_set": {
            "title": "Body-line control",
            "repetitions": 6,
            "distance_m": 50,
            "rest_sec": 20,
            "intensity": "easy aerobic",
            "focus": "Hold the hips near the surface through each breath.",
        },
    },
    "rotation_asymmetry": {
        "title": "Torso rotation is asymmetric",
        "why_it_matters": "Uneven rotation shortens one catch and creates avoidable lateral resistance.",
        "drill": {
            "name": "6-3-6 rotation",
            "purpose": "Match left and right rotation timing.",
            "execution": "Six kicks on the side, three strokes, then six kicks on the opposite side.",
            "common_mistake": "Rotating the shoulders without the hips.",
            "success_cue": "Shoulders and hips turn as one unit.",
        },
        "mini_set": {
            "title": "Symmetric rotation",
            "repetitions": 8,
            "distance_m": 50,
            "rest_sec": 15,
            "intensity": "easy to moderate",
            "focus": "Match the time spent on each side.",
        },
    },
    "crossover_entry": {
        "title": "Hand entry crosses the shoulder line",
        "why_it_matters": "A crossover entry disrupts alignment and reduces the space available for an effective catch.",
        "drill": {
            "name": "Railroad-track freestyle",
            "purpose": "Place each hand in line with its shoulder.",
            "execution": "Swim slowly while entering each hand on an imaginary rail extending from the shoulder.",
            "common_mistake": "Reaching toward the centre line.",
            "success_cue": "Fingertips enter forward of the same-side shoulder.",
        },
        "mini_set": {
            "title": "Entry alignment",
            "repetitions": 6,
            "distance_m": 50,
            "rest_sec": 20,
            "intensity": "easy aerobic",
            "focus": "Keep two separate hand-entry rails.",
        },
    },
    "late_head_return": {
        "title": "Head returns late after breathing",
        "why_it_matters": "A late head return delays body alignment and can push the hips downward.",
        "drill": {
            "name": "One-goggle breathing",
            "purpose": "Return the head with the torso instead of after it.",
            "execution": "Breathe with one goggle remaining in the water and start the return before hand entry.",
            "common_mistake": "Holding the face out until the recovering arm lands.",
            "success_cue": "The face is back in the water before the hand completes entry.",
        },
        "mini_set": {
            "title": "Breath timing",
            "repetitions": 8,
            "distance_m": 25,
            "rest_sec": 15,
            "intensity": "easy",
            "focus": "Return the head before hand entry.",
        },
    },
    "knee_driven_kick": {
        "title": "Kick is driven mainly from the knees",
        "why_it_matters": "Excess knee bend increases drag and spends energy without producing proportional propulsion.",
        "drill": {
            "name": "Streamline back kick",
            "purpose": "Initiate the kick from the hips with relaxed ankles.",
            "execution": "Kick on the back in streamline with small, continuous movements.",
            "common_mistake": "Cycling the lower legs.",
            "success_cue": "The knees stay close to the surface while the toes make small splashes.",
        },
        "mini_set": {
            "title": "Hip-led kick",
            "repetitions": 8,
            "distance_m": 25,
            "rest_sec": 20,
            "intensity": "easy to moderate",
            "focus": "Small kick from the hips with loose ankles.",
        },
    },
}


class SwimmingTechniqueAnalyzer(BaseAnalyzer):
    """Aggregate per-cycle observations into an evidence-backed result."""

    ZONE_IDS: Tuple[str, ...] = (
        "body_position",
        "rotation",
        "catch",
        "breathing",
        "kick",
    )

    def metric_confidence(
        self,
        inputs: ConfidenceInputs,
        *,
        prerequisites_met: bool = True,
    ) -> float:
        if not prerequisites_met:
            return 0.0
        values = asdict(inputs)
        normalized = {name: max(0.0, min(1.0, float(value))) for name, value in values.items()}
        confidence = (
            0.35 * normalized["landmark_visibility"]
            + 0.25 * normalized["temporal_continuity"]
            + 0.15 * normalized["waterline_clarity"]
            + 0.15 * normalized["identity_stability"]
            + 0.10 * normalized["cycle_coverage"]
        )
        return round(confidence, 2)

    @staticmethod
    def confidence_level(value: float) -> str:
        if value >= SWIM_CONFIDENCE_HIGH:
            return "high"
        if value >= SWIM_CONFIDENCE_MEDIUM:
            return "medium"
        return "insufficient"

    def build_result(self, cycle_metrics: List[Mapping[str, Any]]) -> Dict[str, Any]:
        zones = [self._aggregate_zone(zone_id, cycle_metrics) for zone_id in self.ZONE_IDS]
        available = [zone for zone in zones if zone["status"] != "insufficient_data"]
        primary_issue = self._select_primary_issue(zones)
        return {
            "coverage": {
                "available_zones": len(available),
                "total_zones": len(self.ZONE_IDS),
            },
            "overall_score": round(mean(zone["score"] for zone in available), 1) if available else None,
            "zones": zones,
            "primary_issue": primary_issue,
            "prescription": self._prescription(primary_issue),
        }

    def _aggregate_zone(self, zone_id: str, cycle_metrics: List[Mapping[str, Any]]) -> Dict[str, Any]:
        observations = []
        for cycle in cycle_metrics:
            zone = cycle.get("zones", {}).get(zone_id, {})
            if not zone.get("available", False):
                continue
            raw_inputs = zone.get("confidence_inputs", {})
            try:
                inputs = ConfidenceInputs(**raw_inputs)
            except TypeError:
                continue
            confidence = self.metric_confidence(
                inputs,
                prerequisites_met=bool(zone.get("prerequisites_met", True)),
            )
            if confidence < SWIM_CONFIDENCE_MEDIUM:
                continue
            observations.append(
                {
                    "cycle_id": cycle["cycle_id"],
                    "start_sec": float(cycle["start_sec"]),
                    "peak_sec": float(cycle["peak_sec"]),
                    "end_sec": float(cycle["end_sec"]),
                    "confidence": confidence,
                    "score": float(zone["score"]),
                    "issue_code": zone.get("issue_code"),
                    "impact": float(zone.get("impact", 0.0)),
                    "metrics": dict(zone.get("metrics", {})),
                }
            )

        if len(observations) < SWIM_MIN_DIAGNOSIS_CYCLES:
            return {
                "id": zone_id,
                "status": "insufficient_data",
                "score": None,
                "confidence": 0.0,
                "confidence_level": "insufficient",
                "metrics": {},
                "issues": [],
                "evidence": [],
            }

        confidence = round(mean(item["confidence"] for item in observations), 2)
        issue_groups: Dict[str, List[Dict[str, Any]]] = {}
        for item in observations:
            if item["issue_code"]:
                issue_groups.setdefault(item["issue_code"], []).append(item)

        issues = []
        for issue_code, matching in issue_groups.items():
            if len(matching) < SWIM_MIN_DIAGNOSIS_CYCLES:
                continue
            issues.append(
                {
                    "issue_code": issue_code,
                    "confirming_cycles": len(matching),
                    "confidence": round(mean(item["confidence"] for item in matching), 2),
                    "impact": round(mean(item["impact"] for item in matching), 2),
                    "evidence": [self._evidence(item) for item in matching],
                }
            )

        metrics = self._average_metrics([item["metrics"] for item in observations])
        return {
            "id": zone_id,
            "status": "needs_attention" if issues else "good",
            "score": round(mean(item["score"] for item in observations), 1),
            "confidence": confidence,
            "confidence_level": self.confidence_level(confidence),
            "metrics": metrics,
            "issues": issues,
            "evidence": [self._evidence(item) for item in observations],
        }

    @staticmethod
    def _average_metrics(metrics_list: List[Mapping[str, Any]]) -> Dict[str, Any]:
        keys = {key for metrics in metrics_list for key in metrics}
        result: Dict[str, Any] = {}
        for key in keys:
            values = [metrics[key] for metrics in metrics_list if isinstance(metrics.get(key), (int, float))]
            if values:
                result[key] = round(mean(float(value) for value in values), 2)
        return result

    @staticmethod
    def _evidence(observation: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "cycle_id": observation["cycle_id"],
            "start_sec": observation["start_sec"],
            "peak_sec": observation["peak_sec"],
            "end_sec": observation["end_sec"],
        }

    @staticmethod
    def _select_primary_issue(zones: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        candidates = []
        for zone in zones:
            for issue in zone.get("issues", []):
                candidates.append((issue["impact"] * issue["confidence"], zone["id"], issue))
        if not candidates:
            return None
        _, zone_id, issue = max(candidates, key=lambda candidate: candidate[0])
        library_entry = _ISSUE_LIBRARY.get(issue["issue_code"], {})
        return {
            "zone_id": zone_id,
            "issue_code": issue["issue_code"],
            "title": library_entry.get("title", issue["issue_code"].replace("_", " ").title()),
            "why_it_matters": library_entry.get("why_it_matters", ""),
            "confidence": issue["confidence"],
            "confidence_level": SwimmingTechniqueAnalyzer.confidence_level(issue["confidence"]),
            "confirming_cycles": issue["confirming_cycles"],
            "evidence": issue["evidence"],
        }

    @staticmethod
    def _prescription(primary_issue: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        if primary_issue is None:
            return None
        library_entry = _ISSUE_LIBRARY.get(str(primary_issue["issue_code"]))
        if library_entry is None:
            return None
        return {
            "drill": dict(library_entry["drill"]),
            "mini_set": dict(library_entry["mini_set"]),
        }
