"""
🏋️ Exercise Analyzer for Dryland Training

Features:
- Rep counter (automatic)
- Tempo analysis (seconds per rep)
- Range of motion (min/max angles)
- Stability score (deviation between reps)
"""

import numpy as np
import cv2
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RepData:
    """Data for a single repetition."""
    rep_number: int
    start_frame: int
    end_frame: int
    duration_sec: float
    min_angle: float
    max_angle: float
    range_of_motion: float


@dataclass
class ExerciseStats:
    """Overall exercise statistics."""
    exercise_type: str
    total_reps: int
    avg_tempo: float
    avg_range_of_motion: float
    stability_score: float
    min_angle: float
    max_angle: float
    reps: List[RepData] = field(default_factory=list)
    angle_history: List[float] = field(default_factory=list)


class ExerciseAnalyzer:
    """Analyzes dryland exercises for rep counting, tempo, and form."""
    
    # Thresholds for rep detection
    THRESHOLDS = {
        "elbow": {"low": 70, "high": 140},
        "knee": {"low": 100, "high": 165},
    }
    
    def __init__(self, fps: float = 10.0):
        self.fps = fps
        self.angle_history = []
        self.reps: List[RepData] = []
    
    def analyze(self, angles_list: List[Dict], exercise_type: str = "default") -> ExerciseStats:
        """
        Analyze exercise from angle data.
        
        Args:
            angles_list: List of angle dicts per frame
            exercise_type: Type of exercise
            
        Returns:
            ExerciseStats with metrics
        """
        # Determine joint to track
        joint = "elbow"
        if angles_list and "knee" in str(angles_list[0]).lower():
            joint = "knee"
        
        # Extract angle series
        angles = self._extract_angles(angles_list, joint)
        self.angle_history = angles
        
        if len(angles) < 10:
            return self._empty_stats(exercise_type)
        
        # Smooth angles
        smoothed = self._smooth(angles)
        
        # Detect reps
        self.reps = self._detect_reps(smoothed, joint)
        
        # Calculate stats
        return self._calc_stats(exercise_type, smoothed)
    
    def _extract_angles(self, angles_list: List[Dict], joint: str) -> List[float]:
        """Extract angle series for target joint."""
        series = []
        
        for angles in angles_list:
            if not angles:
                series.append(np.nan)
                continue
            
            # Find matching angles
            left = angles.get(f"L.{joint}")
            right = angles.get(f"R.{joint}")
            
            if left is not None and right is not None:
                series.append((left + right) / 2)
            elif left is not None:
                series.append(left)
            elif right is not None:
                series.append(right)
            else:
                series.append(np.nan)
        
        # Interpolate NaNs
        arr = np.array(series)
        nans = np.isnan(arr)
        if not nans.all():
            x = np.arange(len(arr))
            arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
        else:
            arr = np.full(len(series), 90.0)
        
        return arr.tolist()
    
    def _smooth(self, angles: List[float], window: int = 3) -> List[float]:
        """Smooth angle series."""
        if len(angles) < window:
            return angles
        kernel = np.ones(window) / window
        return np.convolve(angles, kernel, mode='same').tolist()
    
    def _detect_reps(self, angles: List[float], joint: str) -> List[RepData]:
        """Detect repetitions from angle series."""
        thresh = self.THRESHOLDS.get(joint, self.THRESHOLDS["elbow"])
        low_thresh = thresh["low"]
        high_thresh = thresh["high"]
        
        # Find peaks (extended) and valleys (contracted)
        peaks = []
        valleys = []
        
        for i in range(2, len(angles) - 2):
            # Local max
            if angles[i] > angles[i-1] and angles[i] > angles[i+1]:
                if angles[i] > high_thresh - 20:
                    peaks.append(i)
            # Local min
            if angles[i] < angles[i-1] and angles[i] < angles[i+1]:
                if angles[i] < low_thresh + 30:
                    valleys.append(i)
        
        # Match valleys -> peak -> valley = 1 rep
        reps = []
        rep_num = 0
        used_valleys = set()
        
        for i, v1 in enumerate(valleys):
            if v1 in used_valleys:
                continue
            
            # Find peak after valley
            peak = None
            for p in peaks:
                if p > v1:
                    peak = p
                    break
            
            if peak is None:
                continue
            
            # Find valley after peak
            v2 = None
            for v in valleys:
                if v > peak and v not in used_valleys:
                    v2 = v
                    break
            
            if v2 is None:
                continue
            
            # Valid rep
            rep_num += 1
            used_valleys.add(v1)
            used_valleys.add(v2)
            
            duration = (v2 - v1) / self.fps
            min_a = min(angles[v1:v2+1])
            max_a = max(angles[v1:v2+1])
            
            reps.append(RepData(
                rep_number=rep_num,
                start_frame=v1,
                end_frame=v2,
                duration_sec=duration,
                min_angle=min_a,
                max_angle=max_a,
                range_of_motion=max_a - min_a,
            ))
        
        return reps
    
    def _calc_stats(self, exercise_type: str, angles: List[float]) -> ExerciseStats:
        """Calculate exercise statistics."""
        if not self.reps:
            return self._empty_stats(exercise_type)
        
        durations = [r.duration_sec for r in self.reps]
        roms = [r.range_of_motion for r in self.reps]
        
        avg_tempo = np.mean(durations)
        avg_rom = np.mean(roms)
        
        # Stability: lower std = higher score
        rom_std = np.std(roms) if len(roms) > 1 else 0
        dur_std = np.std(durations) if len(durations) > 1 else 0
        stability = max(0, 100 - rom_std * 2 - dur_std * 10)
        
        return ExerciseStats(
            exercise_type=exercise_type,
            total_reps=len(self.reps),
            avg_tempo=round(avg_tempo, 2),
            avg_range_of_motion=round(avg_rom, 1),
            stability_score=round(stability, 1),
            min_angle=round(min(angles), 1),
            max_angle=round(max(angles), 1),
            reps=self.reps,
            angle_history=angles,
        )
    
    def _empty_stats(self, exercise_type: str) -> ExerciseStats:
        return ExerciseStats(
            exercise_type=exercise_type,
            total_reps=0, avg_tempo=0, avg_range_of_motion=0,
            stability_score=0, min_angle=0, max_angle=0,
        )
    
    def get_rep_at_frame(self, frame_idx: int) -> Optional[int]:
        """Get current rep number at frame."""
        for rep in self.reps:
            if rep.start_frame <= frame_idx <= rep.end_frame:
                return rep.rep_number
        # After last rep
        if self.reps and frame_idx > self.reps[-1].end_frame:
            return self.reps[-1].rep_number
        return 0
    
    def draw_rep_counter(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Draw rep counter on frame."""
        h, w = frame.shape[:2]
        
        current_rep = self.get_rep_at_frame(frame_idx)
        total_reps = len(self.reps)
        
        # Background
        cv2.rectangle(frame, (w - 150, 10), (w - 10, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (w - 150, 10), (w - 10, 80), (0, 200, 255), 2)
        
        # Rep count
        cv2.putText(frame, f"{current_rep}", (w - 100, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, "REPS", (w - 140, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        return frame


def generate_exercise_chart(stats: ExerciseStats, output_path: str) -> Optional[str]:
    """Generate angle chart for exercise."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.patch.set_facecolor('#1a1a2e')
        
        # Chart 1: Angle over time
        ax1 = axes[0]
        ax1.set_facecolor('#16213e')
        
        timestamps = [i / 10.0 for i in range(len(stats.angle_history))]
        ax1.plot(timestamps, stats.angle_history, color='#00d9ff', linewidth=2)
        
        # Mark reps
        for rep in stats.reps:
            t = rep.start_frame / 10.0
            ax1.axvline(x=t, color='#10b981', linestyle='--', alpha=0.7)
            ax1.text(t, stats.max_angle - 5, f'R{rep.rep_number}', 
                    color='#10b981', fontsize=10, fontweight='bold')
        
        ax1.set_xlabel('Час (сек)', color='white', fontsize=11)
        ax1.set_ylabel('Кут (°)', color='white', fontsize=11)
        ax1.set_title('📐 Кут суглоба в часі', color='white', fontsize=14, fontweight='bold')
        ax1.tick_params(colors='white')
        ax1.grid(True, alpha=0.3, color='#4a5568')
        
        # Chart 2: Rep comparison
        ax2 = axes[1]
        ax2.set_facecolor('#16213e')
        
        if stats.reps:
            x = np.arange(len(stats.reps))
            roms = [r.range_of_motion for r in stats.reps]
            durations = [r.duration_sec for r in stats.reps]
            
            bars = ax2.bar(x, roms, color='#00d9ff', alpha=0.8, label='Амплітуда (°)')
            
            ax2_twin = ax2.twinx()
            ax2_twin.plot(x, durations, color='#f59e0b', marker='o', 
                         linewidth=2, markersize=8, label='Темп (с)')
            
            ax2.set_xlabel('Повторення', color='white', fontsize=11)
            ax2.set_ylabel('Амплітуда (°)', color='#00d9ff', fontsize=11)
            ax2_twin.set_ylabel('Темп (с)', color='#f59e0b', fontsize=11)
            ax2.set_title('📊 Аналіз повторень', color='white', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels([f'R{r.rep_number}' for r in stats.reps])
            ax2.tick_params(colors='white')
            ax2_twin.tick_params(colors='white')
            
            ax2.legend(loc='upper left', facecolor='#16213e', labelcolor='white')
            ax2_twin.legend(loc='upper right', facecolor='#16213e', labelcolor='white')
        
        plt.tight_layout()
        plt.savefig(output_path, facecolor='#1a1a2e', dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
        
    except Exception as e:
        logger.warning(f"Chart generation failed: {e}")
        return None
