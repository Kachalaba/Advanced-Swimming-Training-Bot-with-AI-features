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
    
    # NEW: Advanced metrics
    dps: float = 0.0  # Distance Per Stroke (meters)
    swolf: float = 0.0  # SWOLF score (strokes + seconds per lap)
    
    # Hand/Arm technique
    avg_hand_entry_angle: float = 0.0  # degrees, optimal ~40°
    hand_entry_score: float = 0.0  # 0-100
    avg_elbow_angle_catch: float = 0.0  # High elbow catch angle
    high_elbow_score: float = 0.0  # 0-100
    
    # Head position
    avg_head_position: float = 0.0  # relative to spine
    head_stability_score: float = 0.0  # 0-100
    
    # Breathing
    breathing_pattern: str = ""  # e.g., "bilateral/3", "right/2"
    breathing_regularity: float = 0.0  # 0-100
    breaths_detected: int = 0
    
    # Kick
    kick_frequency: float = 0.0  # kicks per stroke
    kick_amplitude: float = 0.0  # degrees
    kick_symmetry: float = 0.0  # 0-100


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
    
    # Additional keypoints for advanced analysis
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    NOSE = 0
    LEFT_EAR = 7
    RIGHT_EAR = 8
    
    def __init__(self, fps: float = 10.0, pool_length: float = 25.0):
        self.fps = fps
        self.pool_length = pool_length  # meters
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
        
        # NEW: Advanced tracking
        self.hand_entry_angles: List[float] = []
        self.elbow_catch_angles: List[float] = []
        self.head_positions: List[float] = []
        self.breath_frames: List[int] = []  # frames where breathing detected
        self.kick_amplitudes: List[float] = []
        self.left_kick_count = 0
        self.right_kick_count = 0
        self.prev_head_roll = 0
    
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
            
            # NEW: Advanced metrics collection
            # Hand entry angle (during Catch phase)
            if phase == StrokePhase.CATCH:
                entry_angle = self._calculate_hand_entry_angle(kps)
                if entry_angle > 0:
                    self.hand_entry_angles.append(entry_angle)
                
                elbow_angle = self._calculate_elbow_catch_angle(kps)
                if elbow_angle > 0:
                    self.elbow_catch_angles.append(elbow_angle)
            
            # Head position tracking
            head_pos = self._calculate_head_position(kps)
            self.head_positions.append(head_pos)
            
            # Breathing detection
            self._detect_breathing(kps, i)
            
            # Kick analysis
            self._analyze_kick(kps, i)
        
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
    
    # =========================================================================
    # NEW: Advanced Analysis Methods
    # =========================================================================
    
    def _calculate_hand_entry_angle(self, kps: Dict) -> float:
        """
        Calculate hand entry angle during catch phase.
        Optimal angle is ~40° (fingers first, not flat).
        """
        left_wrist = self._get_point(kps, "left_wrist")
        right_wrist = self._get_point(kps, "right_wrist")
        left_elbow = self._get_point(kps, "left_elbow")
        right_elbow = self._get_point(kps, "right_elbow")
        left_shoulder = self._get_point(kps, "left_shoulder")
        right_shoulder = self._get_point(kps, "right_shoulder")
        
        angles = []
        
        # Left arm entry
        if left_wrist and left_elbow and left_shoulder:
            # Angle of forearm relative to water surface (horizontal)
            dx = left_wrist[0] - left_elbow[0]
            dy = left_wrist[1] - left_elbow[1]
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            if 10 < angle < 80:  # Valid range
                angles.append(angle)
        
        # Right arm entry
        if right_wrist and right_elbow and right_shoulder:
            dx = right_wrist[0] - right_elbow[0]
            dy = right_wrist[1] - right_elbow[1]
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            if 10 < angle < 80:
                angles.append(angle)
        
        return np.mean(angles) if angles else 0
    
    def _calculate_elbow_catch_angle(self, kps: Dict) -> float:
        """
        Calculate elbow angle during catch (high elbow technique).
        Optimal: elbow stays high, angle 90-120°.
        """
        left_shoulder = self._get_point(kps, "left_shoulder")
        left_elbow = self._get_point(kps, "left_elbow")
        left_wrist = self._get_point(kps, "left_wrist")
        right_shoulder = self._get_point(kps, "right_shoulder")
        right_elbow = self._get_point(kps, "right_elbow")
        right_wrist = self._get_point(kps, "right_wrist")
        
        angles = []
        
        # Left arm elbow angle
        if left_shoulder and left_elbow and left_wrist:
            angle = self._angle_between_points(left_shoulder, left_elbow, left_wrist)
            if 60 < angle < 180:
                angles.append(angle)
        
        # Right arm
        if right_shoulder and right_elbow and right_wrist:
            angle = self._angle_between_points(right_shoulder, right_elbow, right_wrist)
            if 60 < angle < 180:
                angles.append(angle)
        
        return np.mean(angles) if angles else 0
    
    def _angle_between_points(self, p1, p2, p3) -> float:
        """Calculate angle at p2 between p1-p2-p3."""
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 * mag2 == 0:
            return 0
        
        cos_angle = np.clip(dot / (mag1 * mag2), -1, 1)
        return np.degrees(np.arccos(cos_angle))
    
    def _calculate_head_position(self, kps: Dict) -> float:
        """
        Calculate head position relative to spine.
        Optimal: neutral, aligned with spine (0).
        Positive = head up, Negative = head down.
        """
        nose = self._get_point(kps, "nose")
        left_shoulder = self._get_point(kps, "left_shoulder")
        right_shoulder = self._get_point(kps, "right_shoulder")
        left_hip = self._get_point(kps, "left_hip")
        right_hip = self._get_point(kps, "right_hip")
        
        if not all([nose, left_shoulder, right_shoulder]):
            return 0
        
        # Shoulder center
        shoulder_center = (
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2
        )
        
        # Expected head position (inline with shoulders)
        # Calculate deviation from expected line
        head_offset = nose[1] - shoulder_center[1]
        
        return head_offset
    
    def _detect_breathing(self, kps: Dict, frame_idx: int) -> None:
        """
        Detect breathing based on head rotation.
        Breathing = significant head turn to side.
        """
        nose = self._get_point(kps, "nose")
        left_ear = self._get_point(kps, "left_ear")
        right_ear = self._get_point(kps, "right_ear")
        left_shoulder = self._get_point(kps, "left_shoulder")
        right_shoulder = self._get_point(kps, "right_shoulder")
        
        if not nose or not (left_shoulder and right_shoulder):
            return
        
        # Calculate head rotation based on nose position relative to shoulders
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        head_offset_x = nose[0] - shoulder_center_x
        
        # Also check Y position (head coming up)
        shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
        head_up = nose[1] < shoulder_center_y - 20
        
        # Significant lateral offset + head up = breathing
        threshold = 30  # pixels
        
        if abs(head_offset_x) > threshold and head_up:
            # Check if this is a new breath (not continuation)
            if not self.breath_frames or frame_idx - self.breath_frames[-1] > 5:
                self.breath_frames.append(frame_idx)
    
    def _analyze_kick(self, kps: Dict, frame_idx: int) -> None:
        """
        Analyze kick pattern - amplitude and frequency.
        """
        left_ankle = self._get_point(kps, "left_ankle")
        right_ankle = self._get_point(kps, "right_ankle")
        left_hip = self._get_point(kps, "left_hip")
        right_hip = self._get_point(kps, "right_hip")
        
        if not all([left_ankle, right_ankle, left_hip, right_hip]):
            return
        
        # Calculate kick amplitude (vertical distance between ankles)
        ankle_diff = abs(left_ankle[1] - right_ankle[1])
        
        # Normalize by hip width
        hip_width = abs(left_hip[0] - right_hip[0])
        if hip_width > 0:
            normalized_amplitude = (ankle_diff / hip_width) * 45  # Scale to degrees
            self.kick_amplitudes.append(normalized_amplitude)
    
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
        
        # =====================================================================
        # NEW: Advanced metrics calculation
        # =====================================================================
        
        # DPS (Distance Per Stroke) - estimated
        # If we know pool length and can estimate laps
        dps = self.pool_length / total if total > 0 else 0
        
        # SWOLF (strokes + seconds per lap)
        swolf = total + total_duration if total_duration > 0 else 0
        
        # Hand entry angle
        avg_hand_entry = np.mean(self.hand_entry_angles) if self.hand_entry_angles else 0
        # Score: optimal is ~40°, score decreases as deviation increases
        if avg_hand_entry > 0:
            deviation = abs(avg_hand_entry - 40)
            hand_entry_score = max(0, 100 - deviation * 2)
        else:
            hand_entry_score = 0
        
        # High elbow catch
        avg_elbow_catch = np.mean(self.elbow_catch_angles) if self.elbow_catch_angles else 0
        # Score: optimal is 90-120°
        if avg_elbow_catch > 0:
            if 90 <= avg_elbow_catch <= 120:
                high_elbow_score = 100
            elif 80 <= avg_elbow_catch < 90 or 120 < avg_elbow_catch <= 130:
                high_elbow_score = 80
            else:
                high_elbow_score = max(0, 60 - abs(avg_elbow_catch - 105))
        else:
            high_elbow_score = 0
        
        # Head position
        valid_head_pos = [h for h in self.head_positions if h != 0]
        avg_head_pos = np.mean(valid_head_pos) if valid_head_pos else 0
        head_stability = np.std(valid_head_pos) if len(valid_head_pos) > 1 else 0
        # Score: less movement = better
        head_stability_score = max(0, 100 - head_stability * 2)
        
        # Breathing pattern
        breaths = len(self.breath_frames)
        if breaths > 0 and total > 0:
            strokes_per_breath = total / breaths
            if strokes_per_breath <= 2.5:
                breathing_pattern = f"кожні 2 гребки"
            elif strokes_per_breath <= 3.5:
                breathing_pattern = f"кожні 3 гребки (bilateral)"
            elif strokes_per_breath <= 4.5:
                breathing_pattern = f"кожні 4 гребки"
            else:
                breathing_pattern = f"кожні {strokes_per_breath:.0f} гребків"
            
            # Regularity: check consistency of breath intervals
            if len(self.breath_frames) > 1:
                intervals = np.diff(self.breath_frames)
                interval_std = np.std(intervals)
                breathing_regularity = max(0, 100 - interval_std * 2)
            else:
                breathing_regularity = 50
        else:
            breathing_pattern = "не виявлено"
            breathing_regularity = 0
        
        # Kick analysis
        avg_kick_amplitude = np.mean(self.kick_amplitudes) if self.kick_amplitudes else 0
        kick_frequency = len(self.kick_amplitudes) / total if total > 0 else 0
        # Kick symmetry (based on amplitude variance)
        if len(self.kick_amplitudes) > 2:
            kick_variance = np.std(self.kick_amplitudes)
            kick_symmetry = max(0, 100 - kick_variance * 3)
        else:
            kick_symmetry = 50
        
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
            # NEW metrics
            dps=round(dps, 2),
            swolf=round(swolf, 1),
            avg_hand_entry_angle=round(avg_hand_entry, 1),
            hand_entry_score=round(hand_entry_score, 1),
            avg_elbow_angle_catch=round(avg_elbow_catch, 1),
            high_elbow_score=round(high_elbow_score, 1),
            avg_head_position=round(avg_head_pos, 1),
            head_stability_score=round(head_stability_score, 1),
            breathing_pattern=breathing_pattern,
            breathing_regularity=round(breathing_regularity, 1),
            breaths_detected=breaths,
            kick_frequency=round(kick_frequency, 1),
            kick_amplitude=round(avg_kick_amplitude, 1),
            kick_symmetry=round(kick_symmetry, 1),
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
