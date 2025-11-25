"""
🏊 Swimming Stroke Analyzer

Features:
- Stroke phase detection (Catch → Pull → Push → Recovery)
- Stroke rate (strokes per minute)
- Left/Right symmetry analysis
- Body roll measurement
- Stroke count
"""

import numpy as np
import cv2
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class StrokePhase(Enum):
    """Swimming stroke phases."""
    CATCH = "Catch"        # Hand entry, arm extended
    PULL = "Pull"          # Hand pulls water back
    PUSH = "Push"          # Hand pushes past hip
    RECOVERY = "Recovery"  # Arm exits water, moves forward
    UNKNOWN = "Unknown"


@dataclass
class StrokeData:
    """Data for a single stroke."""
    stroke_number: int
    arm: str  # "left" or "right"
    start_frame: int
    end_frame: int
    duration_sec: float
    phases: Dict[str, int] = field(default_factory=dict)  # phase -> frame count


@dataclass 
class StrokeAnalysis:
    """Complete stroke analysis results."""
    total_strokes: int
    stroke_rate: float  # strokes per minute
    avg_stroke_duration: float
    left_strokes: int
    right_strokes: int
    symmetry_score: float  # 0-100, 100 = perfect symmetry
    avg_body_roll: float  # degrees
    body_roll_left: float
    body_roll_right: float
    phases_distribution: Dict[str, float]  # phase -> percentage
    strokes: List[StrokeData] = field(default_factory=list)
    frame_phases: List[str] = field(default_factory=list)  # phase per frame
    body_roll_history: List[float] = field(default_factory=list)


class StrokeAnalyzer:
    """Analyzes swimming strokes from pose keypoints."""
    
    # Keypoint indices (MediaPipe)
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    
    def __init__(self, fps: float = 10.0):
        self.fps = fps
        self.strokes: List[StrokeData] = []
        self.frame_phases: List[str] = []
        self.body_roll_history: List[float] = []
        
        # Tracking state
        self.left_arm_state = "recovery"
        self.right_arm_state = "recovery"
        self.left_stroke_start = 0
        self.right_stroke_start = 0
        self.stroke_count = 0
        
        # Smoothing
        self.left_wrist_history = deque(maxlen=10)
        self.right_wrist_history = deque(maxlen=10)
    
    def analyze(self, keypoints_list: List[Dict], fps: float = None) -> StrokeAnalysis:
        """
        Analyze strokes from keypoints data.
        
        Args:
            keypoints_list: List of keypoints dicts per frame
            fps: Video FPS
            
        Returns:
            StrokeAnalysis with all metrics
        """
        if fps:
            self.fps = fps
        
        if not keypoints_list or len(keypoints_list) < 10:
            return self._empty_analysis()
        
        # Process each frame
        for i, kps in enumerate(keypoints_list):
            if not kps:
                self.frame_phases.append(StrokePhase.UNKNOWN.value)
                self.body_roll_history.append(0)
                continue
            
            # Detect stroke phase
            phase = self._detect_phase(kps, i)
            self.frame_phases.append(phase.value)
            
            # Calculate body roll
            roll = self._calculate_body_roll(kps)
            self.body_roll_history.append(roll)
            
            # Track strokes
            self._track_strokes(kps, i)
        
        # Calculate statistics
        return self._calculate_stats()
    
    def _detect_phase(self, kps: Dict, frame_idx: int) -> StrokePhase:
        """Detect current stroke phase based on arm position."""
        # Get keypoints
        left_shoulder = self._get_point(kps, "left_shoulder")
        right_shoulder = self._get_point(kps, "right_shoulder")
        left_wrist = self._get_point(kps, "left_wrist")
        right_wrist = self._get_point(kps, "right_wrist")
        left_hip = self._get_point(kps, "left_hip")
        right_hip = self._get_point(kps, "right_hip")
        
        if not all([left_shoulder, right_shoulder, left_wrist, right_wrist]):
            return StrokePhase.UNKNOWN
        
        # Determine dominant arm (which is doing the stroke)
        # The arm that's lower (in water) is the active one
        left_active = left_wrist[1] > left_shoulder[1]
        right_active = right_wrist[1] > right_shoulder[1]
        
        # Check arm position relative to shoulder and hip
        if left_active:
            phase = self._get_arm_phase(left_wrist, left_shoulder, left_hip)
        elif right_active:
            phase = self._get_arm_phase(right_wrist, right_shoulder, right_hip)
        else:
            # Both arms up = recovery
            phase = StrokePhase.RECOVERY
        
        return phase
    
    def _get_arm_phase(
        self, 
        wrist: Tuple[float, float], 
        shoulder: Tuple[float, float],
        hip: Optional[Tuple[float, float]]
    ) -> StrokePhase:
        """Determine stroke phase based on arm position."""
        if hip is None:
            hip = (shoulder[0], shoulder[1] + 100)
        
        # Relative positions
        wrist_x, wrist_y = wrist
        shoulder_x, shoulder_y = shoulder
        hip_x, hip_y = hip
        
        # Hand in front of shoulder (extended) = Catch
        if wrist_x < shoulder_x - 30 and wrist_y < shoulder_y + 50:
            return StrokePhase.CATCH
        
        # Hand between shoulder and hip = Pull
        if shoulder_y <= wrist_y <= hip_y and wrist_x > shoulder_x - 50:
            return StrokePhase.PULL
        
        # Hand past hip = Push
        if wrist_y > hip_y - 20:
            return StrokePhase.PUSH
        
        # Hand above water (y < shoulder) = Recovery
        if wrist_y < shoulder_y:
            return StrokePhase.RECOVERY
        
        return StrokePhase.PULL
    
    def _calculate_body_roll(self, kps: Dict) -> float:
        """Calculate body roll angle in degrees."""
        left_shoulder = self._get_point(kps, "left_shoulder")
        right_shoulder = self._get_point(kps, "right_shoulder")
        
        if not left_shoulder or not right_shoulder:
            return 0.0
        
        # Calculate angle from horizontal
        dx = right_shoulder[0] - left_shoulder[0]
        dy = right_shoulder[1] - left_shoulder[1]
        
        if abs(dx) < 1:
            return 0.0
        
        angle = np.degrees(np.arctan2(dy, dx))
        return angle
    
    def _track_strokes(self, kps: Dict, frame_idx: int) -> None:
        """Track individual strokes."""
        left_wrist = self._get_point(kps, "left_wrist")
        right_wrist = self._get_point(kps, "right_wrist")
        left_shoulder = self._get_point(kps, "left_shoulder")
        right_shoulder = self._get_point(kps, "right_shoulder")
        
        if not all([left_wrist, right_wrist, left_shoulder, right_shoulder]):
            return
        
        # Track wrist positions
        self.left_wrist_history.append(left_wrist)
        self.right_wrist_history.append(right_wrist)
        
        # Detect stroke completion (wrist moves from below hip to above shoulder)
        # Left arm
        if len(self.left_wrist_history) >= 2:
            prev_y = self.left_wrist_history[-2][1]
            curr_y = left_wrist[1]
            shoulder_y = left_shoulder[1]
            
            # Recovery detected (hand moving up past shoulder)
            if prev_y > shoulder_y and curr_y <= shoulder_y:
                if self.left_arm_state == "underwater":
                    self._complete_stroke("left", self.left_stroke_start, frame_idx)
                    self.left_stroke_start = frame_idx
                self.left_arm_state = "recovery"
            
            # Entry detected (hand moving down)
            if prev_y < shoulder_y and curr_y >= shoulder_y:
                self.left_arm_state = "underwater"
                self.left_stroke_start = frame_idx
        
        # Right arm
        if len(self.right_wrist_history) >= 2:
            prev_y = self.right_wrist_history[-2][1]
            curr_y = right_wrist[1]
            shoulder_y = right_shoulder[1]
            
            if prev_y > shoulder_y and curr_y <= shoulder_y:
                if self.right_arm_state == "underwater":
                    self._complete_stroke("right", self.right_stroke_start, frame_idx)
                    self.right_stroke_start = frame_idx
                self.right_arm_state = "recovery"
            
            if prev_y < shoulder_y and curr_y >= shoulder_y:
                self.right_arm_state = "underwater"
                self.right_stroke_start = frame_idx
    
    def _complete_stroke(self, arm: str, start_frame: int, end_frame: int) -> None:
        """Record completed stroke."""
        self.stroke_count += 1
        duration = (end_frame - start_frame) / self.fps
        
        if duration < 0.3 or duration > 5.0:  # Filter invalid strokes
            return
        
        self.strokes.append(StrokeData(
            stroke_number=self.stroke_count,
            arm=arm,
            start_frame=start_frame,
            end_frame=end_frame,
            duration_sec=duration,
        ))
    
    def _get_point(self, kps: Dict, name: str) -> Optional[Tuple[float, float]]:
        """Get keypoint coordinates by name."""
        # Handle different keypoint formats
        if name in kps:
            p = kps[name]
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                return (p[0], p[1])
            elif hasattr(p, 'x') and hasattr(p, 'y'):
                return (p.x, p.y)
        
        # Try alternative names
        alt_names = {
            "left_shoulder": ["L.shoulder", "left_shoulder", 11],
            "right_shoulder": ["R.shoulder", "right_shoulder", 12],
            "left_wrist": ["L.wrist", "left_wrist", 15],
            "right_wrist": ["R.wrist", "right_wrist", 16],
            "left_hip": ["L.hip", "left_hip", 23],
            "right_hip": ["R.hip", "right_hip", 24],
        }
        
        for alt in alt_names.get(name, []):
            if alt in kps:
                p = kps[alt]
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    return (p[0], p[1])
        
        return None
    
    def _calculate_stats(self) -> StrokeAnalysis:
        """Calculate final statistics."""
        if not self.strokes:
            return self._empty_analysis()
        
        # Basic counts
        total = len(self.strokes)
        left_strokes = sum(1 for s in self.strokes if s.arm == "left")
        right_strokes = sum(1 for s in self.strokes if s.arm == "right")
        
        # Stroke rate (strokes per minute)
        total_duration = len(self.frame_phases) / self.fps
        stroke_rate = (total / total_duration) * 60 if total_duration > 0 else 0
        
        # Average duration
        avg_duration = np.mean([s.duration_sec for s in self.strokes])
        
        # Symmetry (100% = equal left/right)
        if left_strokes + right_strokes > 0:
            symmetry = 100 - abs(left_strokes - right_strokes) / (left_strokes + right_strokes) * 100
        else:
            symmetry = 100
        
        # Body roll
        valid_rolls = [r for r in self.body_roll_history if abs(r) < 90]
        if valid_rolls:
            avg_roll = np.mean([abs(r) for r in valid_rolls])
            left_rolls = [r for r in valid_rolls if r < 0]
            right_rolls = [r for r in valid_rolls if r > 0]
            roll_left = abs(np.mean(left_rolls)) if left_rolls else 0
            roll_right = abs(np.mean(right_rolls)) if right_rolls else 0
        else:
            avg_roll = roll_left = roll_right = 0
        
        # Phase distribution
        phase_counts = {}
        for phase in self.frame_phases:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        total_frames = len(self.frame_phases)
        phases_dist = {p: (c / total_frames * 100) for p, c in phase_counts.items()}
        
        return StrokeAnalysis(
            total_strokes=total,
            stroke_rate=round(stroke_rate, 1),
            avg_stroke_duration=round(avg_duration, 2),
            left_strokes=left_strokes,
            right_strokes=right_strokes,
            symmetry_score=round(symmetry, 1),
            avg_body_roll=round(avg_roll, 1),
            body_roll_left=round(roll_left, 1),
            body_roll_right=round(roll_right, 1),
            phases_distribution=phases_dist,
            strokes=self.strokes,
            frame_phases=self.frame_phases,
            body_roll_history=self.body_roll_history,
        )
    
    def _empty_analysis(self) -> StrokeAnalysis:
        return StrokeAnalysis(
            total_strokes=0, stroke_rate=0, avg_stroke_duration=0,
            left_strokes=0, right_strokes=0, symmetry_score=100,
            avg_body_roll=0, body_roll_left=0, body_roll_right=0,
            phases_distribution={},
        )
    
    def draw_phase_overlay(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Draw current phase on frame."""
        if frame_idx >= len(self.frame_phases):
            return frame
        
        phase = self.frame_phases[frame_idx]
        h, w = frame.shape[:2]
        
        # Phase colors
        colors = {
            "Catch": (255, 200, 0),     # Cyan
            "Pull": (0, 255, 0),        # Green
            "Push": (0, 165, 255),      # Orange
            "Recovery": (255, 0, 255),  # Magenta
            "Unknown": (128, 128, 128), # Gray
        }
        color = colors.get(phase, (255, 255, 255))
        
        # Draw phase indicator
        cv2.rectangle(frame, (10, 10), (180, 60), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (180, 60), color, 2)
        cv2.putText(frame, phase, (20, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        # Draw body roll if available
        if frame_idx < len(self.body_roll_history):
            roll = self.body_roll_history[frame_idx]
            roll_text = f"Roll: {roll:.1f}°"
            cv2.putText(frame, roll_text, (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        return frame


def generate_stroke_chart(analysis: StrokeAnalysis, output_path: str) -> Optional[str]:
    """Generate stroke analysis charts."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor('#1a1a2e')
        
        # Chart 1: Phase distribution pie
        ax1 = axes[0, 0]
        ax1.set_facecolor('#16213e')
        phases = list(analysis.phases_distribution.keys())
        values = list(analysis.phases_distribution.values())
        colors = ['#00d9ff', '#10b981', '#f59e0b', '#ec4899', '#6b7280']
        if phases:
            ax1.pie(values, labels=phases, colors=colors[:len(phases)], 
                   autopct='%1.0f%%', textprops={'color': 'white'})
        ax1.set_title('🏊 Розподіл фаз гребка', color='white', fontsize=12, fontweight='bold')
        
        # Chart 2: Body roll over time
        ax2 = axes[0, 1]
        ax2.set_facecolor('#16213e')
        if analysis.body_roll_history:
            frames = range(len(analysis.body_roll_history))
            ax2.plot(frames, analysis.body_roll_history, color='#00d9ff', linewidth=1.5)
            ax2.axhline(y=0, color='#4a5568', linestyle='--', alpha=0.5)
            ax2.fill_between(frames, 0, analysis.body_roll_history, 
                           where=[r > 0 for r in analysis.body_roll_history],
                           color='#10b981', alpha=0.3, label='Право')
            ax2.fill_between(frames, 0, analysis.body_roll_history,
                           where=[r < 0 for r in analysis.body_roll_history],
                           color='#f59e0b', alpha=0.3, label='Ліво')
        ax2.set_xlabel('Кадр', color='white')
        ax2.set_ylabel('Body Roll (°)', color='white')
        ax2.set_title('📐 Body Roll у часі', color='white', fontsize=12, fontweight='bold')
        ax2.tick_params(colors='white')
        ax2.legend(facecolor='#16213e', labelcolor='white')
        
        # Chart 3: Stroke symmetry
        ax3 = axes[1, 0]
        ax3.set_facecolor('#16213e')
        arms = ['Ліва', 'Права']
        counts = [analysis.left_strokes, analysis.right_strokes]
        bars = ax3.bar(arms, counts, color=['#f59e0b', '#10b981'])
        ax3.set_ylabel('Кількість гребків', color='white')
        ax3.set_title(f'⚖️ Симетрія: {analysis.symmetry_score:.0f}%', 
                     color='white', fontsize=12, fontweight='bold')
        ax3.tick_params(colors='white')
        for bar, count in zip(bars, counts):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', color='white', fontweight='bold')
        
        # Chart 4: Stroke durations
        ax4 = axes[1, 1]
        ax4.set_facecolor('#16213e')
        if analysis.strokes:
            stroke_nums = [s.stroke_number for s in analysis.strokes]
            durations = [s.duration_sec for s in analysis.strokes]
            colors_stroke = ['#f59e0b' if s.arm == 'left' else '#10b981' for s in analysis.strokes]
            ax4.bar(stroke_nums, durations, color=colors_stroke)
            ax4.axhline(y=analysis.avg_stroke_duration, color='#00d9ff', 
                       linestyle='--', label=f'Середнє: {analysis.avg_stroke_duration:.2f}с')
        ax4.set_xlabel('Гребок #', color='white')
        ax4.set_ylabel('Тривалість (с)', color='white')
        ax4.set_title('⏱️ Тривалість гребків', color='white', fontsize=12, fontweight='bold')
        ax4.tick_params(colors='white')
        ax4.legend(facecolor='#16213e', labelcolor='white')
        
        plt.tight_layout()
        plt.savefig(output_path, facecolor='#1a1a2e', dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
        
    except Exception as e:
        logger.warning(f"Chart generation failed: {e}")
        return None
