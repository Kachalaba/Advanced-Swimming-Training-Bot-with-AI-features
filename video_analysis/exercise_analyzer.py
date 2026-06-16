from __future__ import annotations

"""
Dryland exercise analyzer.

Features:
- explicit squat / lunge / push-up profiles
- full ready -> effort -> ready repetition detection
- tempo analysis
- range-of-motion evidence
- movement consistency score
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from video_analysis.base_analyzer import BaseAnalyzer
from video_analysis.constants import (
    DRYLAND_EXERCISE_PROFILES,
    DRYLAND_MAX_INTERPOLATION_GAP_FRAMES,
    DRYLAND_MAX_REP_DURATION_SEC,
    DRYLAND_MIN_REP_DURATION_SEC,
)

logger = logging.getLogger(__name__)


@dataclass
class RepData:
    """Data for a single confirmed repetition."""

    rep_number: int
    start_frame: int
    effort_frame: int
    end_frame: int
    duration_sec: float
    min_angle: float
    max_angle: float
    range_of_motion: float
    active_side: str = ""


@dataclass
class ExerciseStats:
    """Overall exercise statistics."""

    exercise_type: str
    tracked_joint: str
    total_reps: int
    avg_tempo: float
    avg_range_of_motion: float
    stability_score: float
    min_angle: float
    max_angle: float
    reps: List[RepData] = field(default_factory=list)
    angle_history: List[float] = field(default_factory=list)


class ExerciseAnalyzer(BaseAnalyzer):
    """Analyzes dryland exercises for rep counting, tempo, and form."""

    def __init__(self, fps: float = 10.0):
        super().__init__()
        self.fps = fps
        self.angle_history: List[float] = []
        self.reps: List[RepData] = []

    def analyze(
        self,
        angles_list: List[Dict],
        exercise_type: str = "squat",
        fps: Optional[float] = None,
    ) -> ExerciseStats:
        """Analyze exercise angle frames using an explicit exercise profile."""

        self.fps = float(fps or self.fps)
        self.angle_history = []
        self.reps = []

        profile = self._profile(exercise_type)
        tracked_joint = str(profile["tracked_joint"])
        angles, sides, valid = self._extract_profile_angles(angles_list, profile)
        self.angle_history = angles

        if len(angles) < 3 or sum(valid) < 3:
            return self._empty_stats(exercise_type, tracked_joint, angles)

        # Rep confirmation uses measured angles directly. Zero-padded
        # convolution can corrupt the ready phase at clip boundaries, which is
        # exactly where short demo recordings often start and end.
        self.reps = self._detect_reps(angles, sides, valid, profile)
        return self._calc_stats(exercise_type, tracked_joint, angles)

    def _profile(self, exercise_type: str) -> Dict:
        try:
            return DRYLAND_EXERCISE_PROFILES[exercise_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported dryland exercise: {exercise_type}") from exc

    def _extract_profile_angles(self, angles_list: List[Dict], profile: Dict) -> tuple[List[float], List[str], List[bool]]:
        """Extract one active side per frame without combining incomplete sides."""

        values: List[float] = []
        sides: List[str] = []
        valid: List[bool] = []

        for angles in angles_list:
            if not angles:
                values.append(0.0)
                sides.append("")
                valid.append(False)
                continue

            candidates: List[tuple[float, str]] = []
            for key in profile["angle_keys"]:
                angle = angles.get(key)
                if angle is None:
                    continue
                side = "left" if key.startswith("L.") else "right"
                candidates.append((float(angle), side))

            if not candidates:
                values.append(0.0)
                sides.append("")
                valid.append(False)
                continue

            angle, side = min(candidates, key=lambda item: item[0])
            values.append(angle)
            sides.append(side)
            valid.append(True)

        return self._interpolate_short_gaps(values, sides, valid)

    def _interpolate_short_gaps(
        self,
        values: List[float],
        sides: List[str],
        valid: List[bool],
    ) -> tuple[List[float], List[str], List[bool]]:
        """Bridge short dropouts only; long gaps stay invalid for rep logic."""

        output = values[:]
        output_sides = sides[:]
        output_valid = valid[:]
        index = 0
        while index < len(output):
            if output_valid[index]:
                index += 1
                continue

            start = index
            while index < len(output) and not output_valid[index]:
                index += 1
            end = index - 1
            gap = end - start + 1
            if start == 0 or index >= len(output) or gap > DRYLAND_MAX_INTERPOLATION_GAP_FRAMES:
                continue

            left = output[start - 1]
            right = output[index]
            step = (right - left) / (gap + 1)
            fill_side = output_sides[start - 1] or output_sides[index]
            for offset in range(gap):
                frame = start + offset
                output[frame] = left + step * (offset + 1)
                output_sides[frame] = fill_side
                output_valid[frame] = True

        fallback = next((value for value, ok in zip(output, output_valid) if ok), 0.0)
        output = [value if ok else fallback for value, ok in zip(output, output_valid)]
        return output, output_sides, output_valid

    def _detect_reps(self, angles: List[float], sides: List[str], valid: List[bool], profile: Dict) -> List[RepData]:
        """Detect confirmed ready -> effort -> ready repetitions."""

        ready_threshold = float(profile["ready_threshold"])
        effort_threshold = float(profile["effort_threshold"])
        min_rom = float(profile["min_rom"])
        reps: List[RepData] = []
        rep_num = 0
        phase = "seek_ready"
        start_frame: Optional[int] = None
        effort_frame: Optional[int] = None
        effort_side = ""
        min_angle = 0.0
        max_angle = 0.0
        effort_min_angle = 0.0

        for index, angle in enumerate(angles):
            if not valid[index]:
                phase = "seek_ready"
                start_frame = None
                effort_frame = None
                effort_side = ""
                effort_min_angle = 0.0
                continue

            if phase == "seek_ready":
                if angle >= ready_threshold:
                    start_frame = index
                    min_angle = angle
                    max_angle = angle
                    effort_side = sides[index]
                    phase = "seek_effort"
                continue

            if start_frame is None:
                phase = "seek_ready"
                continue

            min_angle = min(min_angle, angle)
            max_angle = max(max_angle, angle)

            if phase == "seek_effort":
                if angle <= effort_threshold:
                    effort_frame = index
                    effort_side = sides[index] or effort_side
                    effort_min_angle = angle
                    phase = "seek_return"
                continue

            if phase != "seek_return" or effort_frame is None:
                continue

            if angle < effort_min_angle:
                effort_min_angle = angle
                effort_frame = index
                effort_side = sides[index] or effort_side

            if angle >= ready_threshold:
                duration = (index - start_frame) / self.fps
                rom = max_angle - min_angle
                if (
                    DRYLAND_MIN_REP_DURATION_SEC <= duration <= DRYLAND_MAX_REP_DURATION_SEC
                    and rom >= min_rom
                ):
                    rep_num += 1
                    reps.append(
                        RepData(
                            rep_number=rep_num,
                            start_frame=start_frame,
                            effort_frame=effort_frame,
                            end_frame=index,
                            duration_sec=round(duration, 2),
                            min_angle=round(min_angle, 1),
                            max_angle=round(max_angle, 1),
                            range_of_motion=round(rom, 1),
                            active_side=effort_side,
                        )
                    )

                start_frame = index
                effort_frame = None
                effort_side = sides[index]
                effort_min_angle = 0.0
                min_angle = angle
                max_angle = angle
                phase = "seek_effort"

        return reps

    def _calc_stats(self, exercise_type: str, tracked_joint: str, angles: List[float]) -> ExerciseStats:
        """Calculate exercise statistics from confirmed repetitions."""

        if not self.reps:
            return self._empty_stats(exercise_type, tracked_joint, angles)

        durations = [r.duration_sec for r in self.reps]
        roms = [r.range_of_motion for r in self.reps]
        avg_tempo = float(np.mean(durations))
        avg_rom = float(np.mean(roms))

        rom_std = float(np.std(roms)) if len(roms) > 1 else 0.0
        dur_std = float(np.std(durations)) if len(durations) > 1 else 0.0
        stability = max(0.0, 100.0 - rom_std * 2.0 - dur_std * 10.0)

        return ExerciseStats(
            exercise_type=exercise_type,
            tracked_joint=tracked_joint,
            total_reps=len(self.reps),
            avg_tempo=round(avg_tempo, 2),
            avg_range_of_motion=round(avg_rom, 1),
            stability_score=round(stability, 1),
            min_angle=round(min(angles), 1),
            max_angle=round(max(angles), 1),
            reps=self.reps,
            angle_history=angles,
        )

    def _empty_stats(
        self,
        exercise_type: str,
        tracked_joint: str,
        angles: Optional[List[float]] = None,
    ) -> ExerciseStats:
        return ExerciseStats(
            exercise_type=exercise_type,
            tracked_joint=tracked_joint,
            total_reps=0,
            avg_tempo=0,
            avg_range_of_motion=0,
            stability_score=0,
            min_angle=round(min(angles), 1) if angles else 0,
            max_angle=round(max(angles), 1) if angles else 0,
            angle_history=angles or [],
        )

    def get_rep_at_frame(self, frame_idx: int) -> Optional[int]:
        """Get current rep number at frame."""
        for rep in self.reps:
            if rep.start_frame <= frame_idx <= rep.end_frame:
                return rep.rep_number
        return 0

    def draw_rep_counter(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Draw rep counter on frame."""
        h, w = frame.shape[:2]

        current_rep = self.get_rep_at_frame(frame_idx)

        cv2.rectangle(frame, (w - 150, 10), (w - 10, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (w - 150, 10), (w - 10, 80), (0, 200, 255), 2)

        cv2.putText(
            frame,
            f"{current_rep}",
            (w - 100, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "REPS",
            (w - 140, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        return frame


def generate_exercise_chart(stats: ExerciseStats, output_path: str) -> Optional[str]:
    """Generate an angle chart for exercise evidence."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.patch.set_facecolor("#1a1a2e")

        ax1 = axes[0]
        ax1.set_facecolor("#16213e")
        timestamps = [i / 10.0 for i in range(len(stats.angle_history))]
        ax1.plot(timestamps, stats.angle_history, color="#00d9ff", linewidth=2)
        for rep in stats.reps:
            t = rep.start_frame / 10.0
            ax1.axvline(x=t, color="#10b981", linestyle="--", alpha=0.7)
            ax1.text(
                t,
                stats.max_angle - 5,
                f"R{rep.rep_number}",
                color="#10b981",
                fontsize=10,
                fontweight="bold",
            )
        ax1.set_xlabel("Time (sec)", color="white", fontsize=11)
        ax1.set_ylabel("Angle (deg)", color="white", fontsize=11)
        ax1.set_title("Joint angle over time", color="white", fontsize=14, fontweight="bold")
        ax1.tick_params(colors="white")
        ax1.grid(True, alpha=0.3, color="#4a5568")

        ax2 = axes[1]
        ax2.set_facecolor("#16213e")
        if stats.reps:
            x = np.arange(len(stats.reps))
            roms = [r.range_of_motion for r in stats.reps]
            durations = [r.duration_sec for r in stats.reps]

            ax2.bar(x, roms, color="#00d9ff", alpha=0.8, label="ROM (deg)")
            ax2_twin = ax2.twinx()
            ax2_twin.plot(
                x,
                durations,
                color="#f59e0b",
                marker="o",
                linewidth=2,
                markersize=8,
                label="Tempo (sec)",
            )

            ax2.set_xlabel("Repetition", color="white", fontsize=11)
            ax2.set_ylabel("ROM (deg)", color="#00d9ff", fontsize=11)
            ax2_twin.set_ylabel("Tempo (sec)", color="#f59e0b", fontsize=11)
            ax2.set_title("Repetition evidence", color="white", fontsize=14, fontweight="bold")
            ax2.set_xticks(x)
            ax2.set_xticklabels([f"R{r.rep_number}" for r in stats.reps])
            ax2.tick_params(colors="white")
            ax2_twin.tick_params(colors="white")
            ax2.legend(loc="upper left", facecolor="#16213e", labelcolor="white")
            ax2_twin.legend(loc="upper right", facecolor="#16213e", labelcolor="white")

        plt.tight_layout()
        plt.savefig(output_path, facecolor="#1a1a2e", dpi=150, bbox_inches="tight")
        plt.close()
        return output_path

    except Exception as exc:
        logger.warning("Chart generation failed: %s", exc)
        return None
