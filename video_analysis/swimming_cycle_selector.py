"""Selection of complete, high-quality freestyle stroke cycles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from video_analysis.base_analyzer import BaseAnalyzer
from video_analysis.constants import SWIM_CONFIDENCE_MEDIUM, SWIM_TARGET_CYCLE_COUNT


@dataclass(frozen=True)
class StrokeCycle:
    id: str
    start_frame: int
    peak_frame: int
    end_frame: int
    start_sec: float
    peak_sec: float
    end_sec: float
    quality: float
    complete: bool

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


class SwimmingCycleSelector(BaseAnalyzer):
    """Detect recurring wrist phase and retain the clearest full cycles."""

    REQUIRED_LANDMARKS: Tuple[str, ...] = (
        "left_shoulder",
        "right_shoulder",
        "left_wrist",
        "right_wrist",
    )

    def __init__(self, fps: float = 30.0) -> None:
        super().__init__()
        self.fps = max(float(fps), 1.0)

    def select(
        self,
        frames: Sequence[Mapping[str, Any]],
        limit: int = SWIM_TARGET_CYCLE_COUNT,
    ) -> List[StrokeCycle]:
        if len(frames) < 3 or limit <= 0:
            return []
        raw_signal = [self._phase_value(frame) for frame in frames]
        window = max(3, int(round(self.fps * 0.2)))
        if window % 2 == 0:
            window += 1
        signal = self._smooth(raw_signal, window=window)
        peaks = self._local_peaks(signal, min_gap=max(4, int(round(self.fps * 0.7))))
        candidates = [
            self._build_cycle(frames, signal, start, end, index)
            for index, (start, end) in enumerate(zip(peaks, peaks[1:], strict=False), start=1)
        ]
        reliable = [cycle for cycle in candidates if cycle.complete and cycle.quality >= SWIM_CONFIDENCE_MEDIUM]
        selected = sorted(reliable, key=lambda cycle: cycle.quality, reverse=True)[:limit]
        return sorted(selected, key=lambda cycle: cycle.start_frame)

    def _phase_value(self, frame: Mapping[str, Any]) -> float:
        landmarks = frame.get("landmarks", frame)
        left_shoulder = self._point(landmarks, "left_shoulder")
        right_shoulder = self._point(landmarks, "right_shoulder")
        if left_shoulder is None or right_shoulder is None:
            return 0.0

        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
        shoulder_width = max(abs(right_shoulder[0] - left_shoulder[0]), 1e-6)
        left_wrist = self._point(landmarks, "left_wrist")
        right_wrist = self._point(landmarks, "right_wrist")
        if left_wrist is not None and right_wrist is not None:
            return (left_wrist[1] - right_wrist[1]) / shoulder_width
        if left_wrist is not None:
            return 2.0 * (left_wrist[1] - shoulder_y) / shoulder_width
        if right_wrist is not None:
            return -2.0 * (right_wrist[1] - shoulder_y) / shoulder_width
        return 0.0

    @staticmethod
    def _point(landmarks: Mapping[str, Any], name: str) -> Optional[Tuple[float, float]]:
        value = landmarks.get(name)
        if value is None:
            return None
        if isinstance(value, Mapping):
            if value.get("state") == "bridged":
                return None
            if float(value.get("visibility", 1.0)) < SWIM_CONFIDENCE_MEDIUM:
                return None
            if "x" in value and "y" in value:
                return float(value["x"]), float(value["y"])
            point = value.get("point")
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                return float(point[0]), float(point[1])
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return float(value[0]), float(value[1])
        return None

    @staticmethod
    def _local_peaks(signal: Sequence[float], min_gap: int) -> List[int]:
        candidates = [
            index for index in range(1, len(signal) - 1) if signal[index - 1] < signal[index] >= signal[index + 1]
        ]
        if not candidates:
            return []
        peaks: List[int] = []
        for index in candidates:
            if not peaks or index - peaks[-1] >= min_gap:
                peaks.append(index)
            elif signal[index] > signal[peaks[-1]]:
                peaks[-1] = index
        return peaks

    def _build_cycle(
        self,
        frames: Sequence[Mapping[str, Any]],
        signal: Sequence[float],
        start: int,
        end: int,
        index: int,
    ) -> StrokeCycle:
        duration_sec = (end - start) / self.fps
        segment = signal[start : end + 1]
        peak_frame = start + min(range(len(segment)), key=lambda offset: segment[offset])
        observations = frames[start : end + 1]

        phase_completeness = self._phase_completeness(segment, duration_sec)
        continuity = mean(self._landmark_coverage(frame) for frame in observations)
        tracking = self._mean_metadata(observations, "tracking_confidence")
        waterline = self._mean_metadata(observations, "waterline_confidence")
        blur = self._mean_metadata(observations, "blur_quality")
        quality = 0.25 * phase_completeness + 0.25 * continuity + 0.20 * tracking + 0.15 * waterline + 0.15 * blur
        complete = 0.7 <= duration_sec <= 3.5 and phase_completeness >= 0.7
        return StrokeCycle(
            id=f"cycle-{index}",
            start_frame=start,
            peak_frame=peak_frame,
            end_frame=end,
            start_sec=start / self.fps,
            peak_sec=peak_frame / self.fps,
            end_sec=end / self.fps,
            quality=round(max(0.0, min(1.0, quality)), 2),
            complete=complete,
        )

    @staticmethod
    def _phase_completeness(segment: Sequence[float], duration_sec: float) -> float:
        if not segment or not 0.5 <= duration_sec <= 4.0:
            return 0.0
        amplitude = max(segment) - min(segment)
        amplitude_score = max(0.0, min(1.0, amplitude / 1.5))
        duration_score = 1.0 if 0.7 <= duration_sec <= 3.5 else 0.5
        return 0.7 * amplitude_score + 0.3 * duration_score

    def _landmark_coverage(self, frame: Mapping[str, Any]) -> float:
        landmarks = frame.get("landmarks", frame)
        present = sum(self._point(landmarks, name) is not None for name in self.REQUIRED_LANDMARKS)
        return present / len(self.REQUIRED_LANDMARKS)

    @staticmethod
    def _mean_metadata(frames: Sequence[Mapping[str, Any]], key: str) -> float:
        if not frames:
            return 0.0
        values = [max(0.0, min(1.0, float(frame.get(key, 0.0)))) for frame in frames]
        return mean(values)
