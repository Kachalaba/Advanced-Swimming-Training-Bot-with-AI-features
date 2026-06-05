"""Bilateral range-of-motion analysis for rehabilitation exercises."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union, cast

from video_analysis.base_analyzer import BaseAnalyzer, Point2D
from video_analysis.constants import (
    MEDIAPIPE_LEFT_ANKLE,
    MEDIAPIPE_LEFT_ELBOW,
    MEDIAPIPE_LEFT_HIP,
    MEDIAPIPE_LEFT_KNEE,
    MEDIAPIPE_LEFT_SHOULDER,
    MEDIAPIPE_LEFT_WRIST,
    MEDIAPIPE_RIGHT_ANKLE,
    MEDIAPIPE_RIGHT_ELBOW,
    MEDIAPIPE_RIGHT_HIP,
    MEDIAPIPE_RIGHT_KNEE,
    MEDIAPIPE_RIGHT_SHOULDER,
    MEDIAPIPE_RIGHT_WRIST,
    REHAB_ASYMMETRY_GOOD_PCT,
    REHAB_ASYMMETRY_WARNING_PCT,
    REHAB_CORRECT_REP_COMPLETION_RATIO,
    REHAB_JOINTS,
    REHAB_MAX_REP_DURATION_SEC,
    REHAB_MIN_REP_DURATION_SEC,
    REHAB_MIN_VALID_FRAMES,
    REHAB_PROTOCOLS,
    REHAB_SMOOTHING_WINDOW,
)

_LANDMARK_ALIASES: Dict[str, List[Union[str, int]]] = {
    "left_shoulder": [MEDIAPIPE_LEFT_SHOULDER],
    "right_shoulder": [MEDIAPIPE_RIGHT_SHOULDER],
    "left_elbow": [MEDIAPIPE_LEFT_ELBOW],
    "right_elbow": [MEDIAPIPE_RIGHT_ELBOW],
    "left_wrist": [MEDIAPIPE_LEFT_WRIST],
    "right_wrist": [MEDIAPIPE_RIGHT_WRIST],
    "left_hip": [MEDIAPIPE_LEFT_HIP],
    "right_hip": [MEDIAPIPE_RIGHT_HIP],
    "left_knee": [MEDIAPIPE_LEFT_KNEE],
    "right_knee": [MEDIAPIPE_RIGHT_KNEE],
    "left_ankle": [MEDIAPIPE_LEFT_ANKLE],
    "right_ankle": [MEDIAPIPE_RIGHT_ANKLE],
}


class RehabAnalyzer(BaseAnalyzer):
    """Calculate bilateral ROM, repetitions, deficits, and asymmetry."""

    def __init__(self, fps: float = 30.0, ema_alpha: Optional[float] = None) -> None:
        super().__init__() if ema_alpha is None else super().__init__(ema_alpha=ema_alpha)
        self.fps = fps if fps > 0 else 30.0

    def analyze(self, keypoints_list: List[Dict], protocol: str = "shoulder_flexion") -> Dict[str, Any]:
        """Analyze a rehabilitation protocol from per-frame pose keypoints."""
        if protocol not in REHAB_PROTOCOLS:
            raise ValueError(f"Unknown rehabilitation protocol: {protocol}")

        self._ema_state.clear()
        angle_histories = self._collect_joint_angles(keypoints_list)
        joint_metrics: Dict[str, Dict[str, Any]] = {}
        for joint_name, side_histories in angle_histories.items():
            joint_metrics[joint_name] = self._summarize_joint(joint_name, side_histories)

        protocol_config = cast(Dict[str, Any], REHAB_PROTOCOLS[protocol])
        target_joint = str(protocol_config["joint_metric"])
        target_rom = cast(float, protocol_config["target_rom"])
        target_metrics = {
            side: self._with_protocol_target(joint_metrics[target_joint][side], target_rom)
            for side in ("left", "right")
        }
        symmetry = self._symmetry(target_metrics["left"]["rom"], target_metrics["right"]["rom"])
        completion_score = round(
            (target_metrics["left"]["completion_pct"] + target_metrics["right"]["completion_pct"]) / 2.0,
            1,
        )

        # Angles and their real video-frame indices for the target joint. Left and
        # right are sampled independently (a side is recorded only when all three of
        # its landmarks are visible), so the frame indices let consumers align the
        # two series on a shared time axis instead of by list position.
        target_angle_history = {side: list(angle_histories[target_joint][side]["angles"]) for side in ("left", "right")}
        target_angle_frames = {
            side: [int(f) for f in angle_histories[target_joint][side]["frames"]] for side in ("left", "right")
        }

        return {
            "protocol": protocol,
            "target_joint": target_joint,
            "target_rom": target_rom,
            "valid_frames": max(
                len(target_angle_history["left"]),
                len(target_angle_history["right"]),
            ),
            "joint_metrics": joint_metrics,
            "target_metrics": target_metrics,
            "symmetry": symmetry,
            "completion_score": completion_score,
            "total_correct_reps": max(
                int(target_metrics["left"]["correct_reps"]),
                int(target_metrics["right"]["correct_reps"]),
            ),
            "angle_history": target_angle_history,
            "angle_frames": target_angle_frames,
            "rep_rom_history": {
                side: [rep["rom"] for rep in target_metrics[side]["rep_details"]] for side in ("left", "right")
            },
            "feedback": self._build_feedback(
                target_angle_history,
                target_metrics,
                symmetry,
            ),
        }

    def _collect_joint_angles(self, keypoints_list: List[Dict]) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
        # Per joint/side we keep the smoothed angle series plus the real frame index
        # each sample came from. Frames without a full landmark triple are dropped
        # rather than carried as gaps, so the parallel "frames" list is what keeps
        # rep durations tied to wall-clock time instead of sample position.
        histories: Dict[str, Dict[str, Dict[str, List[float]]]] = {
            joint_name: {
                "left": {"angles": [], "frames": []},
                "right": {"angles": [], "frames": []},
            }
            for joint_name in REHAB_JOINTS
        }
        for frame_idx, keypoints in enumerate(keypoints_list):
            if not keypoints:
                continue
            for joint_name, raw_config in REHAB_JOINTS.items():
                config = cast(Dict[str, Any], raw_config)
                point_names = cast(Tuple[str, str, str], config["points"])
                for side in ("left", "right"):
                    points: List[Optional[Point2D]] = [
                        self._get_point(
                            keypoints,
                            f"{side}_{point_name}",
                            _LANDMARK_ALIASES,
                        )
                        for point_name in point_names
                    ]
                    p1, p2, p3 = points
                    if p1 is None or p2 is None or p3 is None:
                        continue
                    angle = self._calculate_angle(p1, p2, p3)
                    if angle > 0:
                        side_history = histories[joint_name][side]
                        side_history["angles"].append(self._ema(f"{joint_name}:{side}", angle))
                        side_history["frames"].append(frame_idx)
        return histories

    def _summarize_joint(
        self,
        joint_name: str,
        side_histories: Dict[str, Dict[str, List[float]]],
    ) -> Dict[str, Any]:
        config = cast(Dict[str, Any], REHAB_JOINTS[joint_name])
        target_rom = cast(float, config["target_rom"])
        left = self._summarize_side(
            joint_name, side_histories["left"]["angles"], side_histories["left"]["frames"], target_rom
        )
        right = self._summarize_side(
            joint_name, side_histories["right"]["angles"], side_histories["right"]["frames"], target_rom
        )
        symmetry = self._symmetry(left["rom"], right["rom"])
        return {
            "left": left,
            "right": right,
            "asymmetry_index": symmetry["asymmetry_index"],
            "symmetry_score": symmetry["score"],
        }

    def _summarize_side(
        self,
        joint_name: str,
        angles: List[float],
        frames: List[float],
        target_rom: float,
    ) -> Dict[str, Any]:
        if not angles:
            return self._empty_side(target_rom)

        smoothed = self._smooth(angles, window=REHAB_SMOOTHING_WINDOW)
        edge = REHAB_SMOOTHING_WINDOW // 2
        if edge:
            smoothed[:edge] = angles[:edge]
            smoothed[-edge:] = angles[-edge:]
        reps = self._detect_reps(smoothed, frames, joint_name, target_rom)
        minimum = min(smoothed)
        maximum = max(smoothed)
        achieved_rom = maximum - minimum
        return {
            "min_angle": round(minimum, 1),
            "max_angle": round(maximum, 1),
            "average_angle": round(sum(smoothed) / len(smoothed), 1),
            "rom": round(achieved_rom, 1),
            "target_rom": target_rom,
            "deficit_deg": round(max(0.0, target_rom - achieved_rom), 1),
            "completion_pct": round(min(100.0, achieved_rom / target_rom * 100.0), 1),
            "meets_target": achieved_rom >= target_rom,
            "reps": len(reps),
            "correct_reps": sum(1 for rep in reps if rep["correct"]),
            "rep_details": reps,
        }

    def _detect_reps(
        self,
        angles: List[float],
        frames: List[float],
        joint_name: str,
        target_rom: float,
    ) -> List[Dict[str, Any]]:
        config = cast(Dict[str, Any], REHAB_JOINTS[joint_name])
        direction = cast(str, config["direction"])
        rest_threshold = cast(float, config["rest_threshold"])
        active_threshold = cast(float, config["active_threshold"])
        at_rest = (
            (lambda value: value <= rest_threshold)
            if direction == "increase"
            else (lambda value: value >= rest_threshold)
        )
        at_active = (
            (lambda value: value >= active_threshold)
            if direction == "increase"
            else (lambda value: value <= active_threshold)
        )

        reps: List[Dict[str, Any]] = []
        state = "waiting_for_rest"
        # Positions index into the (compacted) angle series; frames[pos] maps each
        # position back to the originating video frame so durations reflect real time.
        start_pos = 0
        active_pos = 0
        for pos, angle in enumerate(angles):
            if state == "waiting_for_rest" and at_rest(angle):
                start_pos = pos
                state = "moving"
            elif state == "moving" and at_active(angle):
                active_pos = pos
                state = "returning"
            elif state == "returning" and at_rest(angle):
                segment = angles[start_pos : pos + 1]
                duration = (frames[pos] - frames[start_pos]) / self.fps
                achieved_rom = max(segment) - min(segment)
                duration_ok = REHAB_MIN_REP_DURATION_SEC <= duration <= REHAB_MAX_REP_DURATION_SEC
                reps.append(
                    {
                        "rep_number": len(reps) + 1,
                        "start_frame": int(frames[start_pos]),
                        "active_frame": int(frames[active_pos]),
                        "end_frame": int(frames[pos]),
                        "duration_sec": round(duration, 2),
                        "min_angle": round(min(segment), 1),
                        "max_angle": round(max(segment), 1),
                        "rom": round(achieved_rom, 1),
                        "correct": (achieved_rom >= target_rom * REHAB_CORRECT_REP_COMPLETION_RATIO and duration_ok),
                    }
                )
                start_pos = pos
                state = "moving"
        return reps

    @staticmethod
    def _with_protocol_target(metrics: Dict[str, Any], target_rom: float) -> Dict[str, Any]:
        result = dict(metrics)
        achieved_rom = float(result["rom"])
        result["target_rom"] = target_rom
        result["deficit_deg"] = round(max(0.0, target_rom - achieved_rom), 1)
        result["completion_pct"] = round(
            min(100.0, achieved_rom / target_rom * 100.0),
            1,
        )
        result["meets_target"] = achieved_rom >= target_rom
        result["correct_reps"] = sum(
            1
            for rep in result["rep_details"]
            if rep["rom"] >= target_rom * REHAB_CORRECT_REP_COMPLETION_RATIO
            and REHAB_MIN_REP_DURATION_SEC <= rep["duration_sec"] <= REHAB_MAX_REP_DURATION_SEC
        )
        return result

    @staticmethod
    def _symmetry(left_rom: float, right_rom: float) -> Dict[str, Any]:
        denominator = max(float(left_rom), float(right_rom))
        asymmetry = abs(float(left_rom) - float(right_rom)) / denominator * 100.0 if denominator > 0 else 0.0
        if asymmetry <= REHAB_ASYMMETRY_GOOD_PCT:
            status = "good"
        elif asymmetry <= REHAB_ASYMMETRY_WARNING_PCT:
            status = "moderate"
        else:
            status = "high"
        return {
            "asymmetry_index": round(asymmetry, 1),
            "score": round(max(0.0, 100.0 - asymmetry), 1),
            "status": status,
        }

    @staticmethod
    def _empty_side(target_rom: float) -> Dict[str, Any]:
        return {
            "min_angle": 0.0,
            "max_angle": 0.0,
            "average_angle": 0.0,
            "rom": 0.0,
            "target_rom": target_rom,
            "deficit_deg": target_rom,
            "completion_pct": 0.0,
            "meets_target": False,
            "reps": 0,
            "correct_reps": 0,
            "rep_details": [],
        }

    @staticmethod
    def _build_feedback(
        histories: Dict[str, List[float]],
        target_metrics: Dict[str, Dict[str, Any]],
        symmetry: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if max(len(histories["left"]), len(histories["right"])) < REHAB_MIN_VALID_FRAMES:
            return [{"code": "insufficient_pose"}]

        feedback = []
        for side in ("left", "right"):
            if target_metrics[side]["deficit_deg"] > 0:
                feedback.append(
                    {
                        "code": "rom_deficit",
                        "side": side,
                        "value": target_metrics[side]["deficit_deg"],
                    }
                )
        if symmetry["status"] == "high":
            feedback.append(
                {
                    "code": "asymmetry_high",
                    "value": symmetry["asymmetry_index"],
                }
            )
        elif symmetry["status"] == "moderate":
            feedback.append(
                {
                    "code": "asymmetry_moderate",
                    "value": symmetry["asymmetry_index"],
                }
            )
        if not feedback:
            feedback.append({"code": "target_met"})
        return feedback
