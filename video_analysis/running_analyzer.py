"""
🏃 Running Biomechanics Analyzer

Features:
- Cadence (steps per minute)
- Stride length estimation
- Ground contact time
- Vertical oscillation
- Knee lift angle
- Arm swing symmetry
- Posture analysis (forward lean)
- Multi-person support
- Full 33-point MediaPipe skeleton
"""

import cv2
import numpy as np
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy

logger = logging.getLogger(__name__)


class RunPhase(Enum):
    """Running gait phases."""
    STANCE = "Stance"           # Foot on ground
    FLIGHT = "Flight"           # Both feet off ground
    LOADING = "Loading"         # Initial contact
    PUSH_OFF = "Push-off"       # Toe-off
    SWING = "Swing"             # Leg swinging forward
    UNKNOWN = "Unknown"


@dataclass
class StepData:
    """Single step data."""
    frame: int
    side: str  # "left" or "right"
    duration_sec: float = 0
    knee_angle: float = 0
    hip_angle: float = 0


@dataclass 
class RunningAnalysis:
    """Running analysis results."""
    person_id: int = 0  # For multi-person tracking
    
    total_steps: int = 0
    left_steps: int = 0
    right_steps: int = 0
    cadence: float = 0  # steps per minute
    avg_stride_length: float = 0  # meters (estimated)
    
    avg_knee_lift: float = 0  # degrees
    max_knee_lift: float = 0
    
    avg_vertical_osc: float = 0  # pixels
    
    forward_lean: float = 0  # degrees
    arm_symmetry: float = 0  # percentage
    
    ground_contact_ratio: float = 0  # percentage of time on ground
    
    steps: List[StepData] = field(default_factory=list)
    phase_distribution: Dict[str, float] = field(default_factory=dict)
    
    duration_sec: float = 0
    
    # NEW: Advanced metrics
    # Foot strike pattern
    foot_strike_type: str = ""  # "heel", "midfoot", "forefoot"
    foot_strike_angle: float = 0  # degrees
    foot_strike_score: float = 0  # 0-100
    
    # Overstriding
    overstriding_detected: bool = False
    overstriding_score: float = 0  # 0-100, 100 = no overstriding
    avg_foot_ahead_distance: float = 0  # pixels ahead of hip at contact
    
    # Hip drop (Trendelenburg)
    hip_drop_left: float = 0  # degrees
    hip_drop_right: float = 0  # degrees
    hip_drop_score: float = 0  # 0-100, 100 = stable
    
    # Arm swing
    arm_crossover_detected: bool = False
    arm_swing_range: float = 0  # degrees
    
    # Ground contact
    avg_contact_time_ms: float = 0  # milliseconds
    contact_time_left: float = 0
    contact_time_right: float = 0
    
    # Bounce (vertical oscillation efficiency)
    bounce_score: float = 0  # 0-100, higher = less wasted energy
    
    # Overall scores
    efficiency_score: float = 0  # 0-100
    injury_risk_score: float = 0  # 0-100, lower = better


@dataclass
class MultiPersonRunningAnalysis:
    """Running analysis for multiple people."""
    person_count: int = 0
    analyses: Dict[int, 'RunningAnalysis'] = field(default_factory=dict)  # person_id -> analysis
    
    def get_best_runner(self) -> Optional['RunningAnalysis']:
        """Get analysis of person with most steps detected."""
        if not self.analyses:
            return None
        return max(self.analyses.values(), key=lambda a: a.total_steps)
    
    def get_all(self) -> List['RunningAnalysis']:
        """Get all analyses."""
        return list(self.analyses.values())


class RunningAnalyzer:
    """Analyze running biomechanics from video. Supports multiple people."""
    
    # Full MediaPipe 33-point landmark indices
    LANDMARKS = {
        # Face
        "nose": 0,
        "left_eye_inner": 1,
        "left_eye": 2,
        "left_eye_outer": 3,
        "right_eye_inner": 4,
        "right_eye": 5,
        "right_eye_outer": 6,
        "left_ear": 7,
        "right_ear": 8,
        "mouth_left": 9,
        "mouth_right": 10,
        # Upper body
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_pinky": 17,
        "right_pinky": 18,
        "left_index": 19,
        "right_index": 20,
        "left_thumb": 21,
        "right_thumb": 22,
        # Lower body
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
        "left_heel": 29,
        "right_heel": 30,
        "left_foot_index": 31,  # toe
        "right_foot_index": 32,  # toe
    }
    
    # Alias for compatibility
    LANDMARKS["left_toe"] = 31
    LANDMARKS["right_toe"] = 32
    
    def __init__(self, fps: float = 30.0, multi_person: bool = True):
        self.fps = fps
        self.multi_person = multi_person
        self._reset_tracking()
    
    def _reset_tracking(self):
        """Reset all tracking variables."""
        self.steps: List[StepData] = []
        self.phases: List[RunPhase] = []
        self.hip_heights: List[float] = []
        
        # Advanced tracking
        self.foot_strike_angles: List[float] = []
        self.foot_ahead_distances: List[float] = []
        self.hip_drops_left: List[float] = []
        self.hip_drops_right: List[float] = []
        self.arm_crossovers: List[bool] = []
        self.contact_times_left: List[float] = []
        self.contact_times_right: List[float] = []
        
        self.left_contact_start = None
        self.right_contact_start = None
        
    def analyze(self, keypoints_list: Union[List[Dict], List[List[Dict]]], fps: float = None) -> Union[RunningAnalysis, MultiPersonRunningAnalysis]:
        """
        Analyze running from keypoints sequence.
        
        Args:
            keypoints_list: List of keypoints dicts per frame (single person)
                           OR List of Lists of keypoints dicts per frame (multi-person)
            fps: Frames per second
            
        Returns:
            RunningAnalysis for single person
            MultiPersonRunningAnalysis for multiple people
        """
        if fps:
            self.fps = fps
            
        if not keypoints_list or len(keypoints_list) < 10:
            return self._empty_analysis()
        
        # Check if multi-person format (list of lists)
        if keypoints_list and isinstance(keypoints_list[0], list):
            return self._analyze_multi_person(keypoints_list)
        
        # Single person analysis
        return self._analyze_single_person(keypoints_list)
    
    def _analyze_multi_person(self, keypoints_list: List[List[Dict]]) -> MultiPersonRunningAnalysis:
        """Analyze multiple people running."""
        # Reorganize data by person
        # keypoints_list[frame_idx] = [person1_kps, person2_kps, ...]
        
        # Find max number of people detected in any frame
        max_persons = max(len(frame_kps) for frame_kps in keypoints_list if frame_kps)
        
        result = MultiPersonRunningAnalysis(person_count=max_persons)
        
        for person_idx in range(max_persons):
            # Extract keypoints for this person across all frames
            person_keypoints = []
            for frame_kps in keypoints_list:
                if frame_kps and person_idx < len(frame_kps):
                    person_keypoints.append(frame_kps[person_idx])
                else:
                    person_keypoints.append({})
            
            # Analyze this person
            self._reset_tracking()
            analysis = self._analyze_single_person(person_keypoints)
            analysis.person_id = person_idx
            result.analyses[person_idx] = analysis
        
        return result
    
    def _analyze_single_person(self, keypoints_list: List[Dict]) -> RunningAnalysis:
        """Analyze single person running."""
        self._reset_tracking()
        
        # Track metrics per frame
        knee_lifts_left = []
        knee_lifts_right = []
        forward_leans = []
        arm_angles_left = []
        arm_angles_right = []
        
        prev_left_ankle_y = None
        prev_right_ankle_y = None
        left_step_start = None
        right_step_start = None
        
        for frame_idx, kps in enumerate(keypoints_list):
            if not kps:
                self.phases.append(RunPhase.UNKNOWN)
                continue
            
            # Get key points
            l_hip = self._get_point(kps, "left_hip")
            r_hip = self._get_point(kps, "right_hip")
            l_knee = self._get_point(kps, "left_knee")
            r_knee = self._get_point(kps, "right_knee")
            l_ankle = self._get_point(kps, "left_ankle")
            r_ankle = self._get_point(kps, "right_ankle")
            l_shoulder = self._get_point(kps, "left_shoulder")
            r_shoulder = self._get_point(kps, "right_shoulder")
            l_elbow = self._get_point(kps, "left_elbow")
            r_elbow = self._get_point(kps, "right_elbow")
            nose = self._get_point(kps, "nose")
            
            # Calculate hip center height for vertical oscillation
            if l_hip and r_hip:
                hip_center_y = (l_hip[1] + r_hip[1]) / 2
                self.hip_heights.append(hip_center_y)
            
            # Calculate knee lift angles
            if l_hip and l_knee and l_ankle:
                knee_angle = self._calculate_angle(l_hip, l_knee, l_ankle)
                knee_lift = 180 - knee_angle  # Higher = more lift
                knee_lifts_left.append(knee_lift)
            
            if r_hip and r_knee and r_ankle:
                knee_angle = self._calculate_angle(r_hip, r_knee, r_ankle)
                knee_lift = 180 - knee_angle
                knee_lifts_right.append(knee_lift)
            
            # Calculate forward lean
            if l_shoulder and r_shoulder and l_hip and r_hip:
                shoulder_center = ((l_shoulder[0] + r_shoulder[0]) / 2,
                                  (l_shoulder[1] + r_shoulder[1]) / 2)
                hip_center = ((l_hip[0] + r_hip[0]) / 2,
                             (l_hip[1] + r_hip[1]) / 2)
                
                # Angle from vertical
                dx = shoulder_center[0] - hip_center[0]
                dy = hip_center[1] - shoulder_center[1]  # Inverted Y
                lean = math.degrees(math.atan2(dx, dy))
                forward_leans.append(lean)
            
            # Arm swing angles
            if l_shoulder and l_elbow:
                arm_angle = self._calculate_arm_angle(l_shoulder, l_elbow, l_hip)
                arm_angles_left.append(arm_angle)
            
            if r_shoulder and r_elbow:
                arm_angle = self._calculate_arm_angle(r_shoulder, r_elbow, r_hip)
                arm_angles_right.append(arm_angle)
            
            # Step detection based on ankle height changes
            if l_ankle and r_ankle:
                # Detect left step (ankle goes down then up)
                if prev_left_ankle_y is not None:
                    if prev_left_ankle_y < l_ankle[1] and left_step_start is None:
                        # Ankle going down - starting stance
                        left_step_start = frame_idx
                    elif prev_left_ankle_y > l_ankle[1] and left_step_start is not None:
                        # Ankle going up - end of stance, step complete
                        step = StepData(
                            frame=left_step_start,
                            side="left",
                            duration_sec=(frame_idx - left_step_start) / self.fps,
                            knee_angle=knee_lifts_left[-1] if knee_lifts_left else 0
                        )
                        self.steps.append(step)
                        left_step_start = None
                
                # Detect right step
                if prev_right_ankle_y is not None:
                    if prev_right_ankle_y < r_ankle[1] and right_step_start is None:
                        right_step_start = frame_idx
                    elif prev_right_ankle_y > r_ankle[1] and right_step_start is not None:
                        step = StepData(
                            frame=right_step_start,
                            side="right",
                            duration_sec=(frame_idx - right_step_start) / self.fps,
                            knee_angle=knee_lifts_right[-1] if knee_lifts_right else 0
                        )
                        self.steps.append(step)
                        right_step_start = None
                
                prev_left_ankle_y = l_ankle[1]
                prev_right_ankle_y = r_ankle[1]
            
            # Detect phase
            phase = self._detect_phase(kps)
            self.phases.append(phase)
            
            # NEW: Advanced metrics collection
            # Foot strike analysis (at initial contact)
            if phase == RunPhase.LOADING or phase == RunPhase.STANCE:
                foot_angle = self._analyze_foot_strike(kps)
                if foot_angle != 0:
                    self.foot_strike_angles.append(foot_angle)
                
                # Overstriding check
                foot_ahead = self._check_overstriding(kps)
                if foot_ahead != 0:
                    self.foot_ahead_distances.append(foot_ahead)
            
            # Hip drop analysis
            hip_drop = self._analyze_hip_drop(kps)
            if hip_drop[0] != 0:
                self.hip_drops_left.append(hip_drop[0])
            if hip_drop[1] != 0:
                self.hip_drops_right.append(hip_drop[1])
            
            # Arm crossover check
            if l_elbow and r_elbow and l_shoulder and r_shoulder:
                crossover = self._check_arm_crossover(kps)
                self.arm_crossovers.append(crossover)
            
            # Contact time tracking
            self._track_contact_time(kps, frame_idx)
        
        # Calculate final metrics
        return self._calculate_stats(
            knee_lifts_left, knee_lifts_right,
            forward_leans, arm_angles_left, arm_angles_right,
            len(keypoints_list)
        )
    
    def _get_point(self, kps: Dict, name: str) -> Optional[Tuple[float, float]]:
        """Get point coordinates from keypoints dict."""
        if name in kps:
            pt = kps[name]
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                return (pt[0], pt[1])
            elif hasattr(pt, 'x') and hasattr(pt, 'y'):
                return (pt.x, pt.y)
        return None
    
    def _calculate_angle(self, p1, p2, p3) -> float:
        """Calculate angle at p2 between p1-p2-p3."""
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 * mag2 == 0:
            return 0
        
        cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_angle))
    
    def _calculate_arm_angle(self, shoulder, elbow, hip) -> float:
        """Calculate arm swing angle relative to torso."""
        if not hip:
            return 0
        # Angle of upper arm relative to vertical torso
        dx = elbow[0] - shoulder[0]
        dy = elbow[1] - shoulder[1]
        return abs(math.degrees(math.atan2(dx, dy)))
    
    def _detect_phase(self, kps: Dict) -> RunPhase:
        """Detect current running phase."""
        l_ankle = self._get_point(kps, "left_ankle")
        r_ankle = self._get_point(kps, "right_ankle")
        l_hip = self._get_point(kps, "left_hip")
        r_hip = self._get_point(kps, "right_hip")
        
        if not all([l_ankle, r_ankle, l_hip, r_hip]):
            return RunPhase.UNKNOWN
        
        hip_y = (l_hip[1] + r_hip[1]) / 2
        
        # Simple heuristic: if both ankles are above hip level, likely flight
        # If one ankle is significantly lower, stance phase
        l_rel = l_ankle[1] - hip_y
        r_rel = r_ankle[1] - hip_y
        
        threshold = 50  # pixels
        
        if l_rel > threshold and r_rel > threshold:
            return RunPhase.STANCE
        elif l_rel < -threshold and r_rel < -threshold:
            return RunPhase.FLIGHT
        else:
            return RunPhase.SWING
    
    # =========================================================================
    # NEW: Advanced Analysis Methods
    # =========================================================================
    
    def _analyze_foot_strike(self, kps: Dict) -> float:
        """
        Analyze foot strike pattern.
        Returns angle of foot at contact.
        Positive = heel strike, ~0 = midfoot, Negative = forefoot
        """
        l_heel = self._get_point(kps, "left_heel")
        l_toe = self._get_point(kps, "left_toe")
        r_heel = self._get_point(kps, "right_heel")
        r_toe = self._get_point(kps, "right_toe")
        l_ankle = self._get_point(kps, "left_ankle")
        r_ankle = self._get_point(kps, "right_ankle")
        
        angles = []
        
        # Left foot angle
        if l_heel and l_toe:
            dx = l_toe[0] - l_heel[0]
            dy = l_toe[1] - l_heel[1]
            angle = math.degrees(math.atan2(dy, dx))
            angles.append(angle)
        
        # Right foot
        if r_heel and r_toe:
            dx = r_toe[0] - r_heel[0]
            dy = r_toe[1] - r_heel[1]
            angle = math.degrees(math.atan2(dy, dx))
            angles.append(angle)
        
        return np.mean(angles) if angles else 0
    
    def _check_overstriding(self, kps: Dict) -> float:
        """
        Check for overstriding - foot landing ahead of hips.
        Returns distance foot is ahead of hip (positive = overstriding).
        """
        l_ankle = self._get_point(kps, "left_ankle")
        r_ankle = self._get_point(kps, "right_ankle")
        l_hip = self._get_point(kps, "left_hip")
        r_hip = self._get_point(kps, "right_hip")
        
        distances = []
        
        if l_ankle and l_hip:
            # Positive = foot ahead of hip (overstriding)
            dist = l_hip[0] - l_ankle[0]  # Assuming running left to right
            distances.append(dist)
        
        if r_ankle and r_hip:
            dist = r_hip[0] - r_ankle[0]
            distances.append(dist)
        
        return np.mean(distances) if distances else 0
    
    def _analyze_hip_drop(self, kps: Dict) -> Tuple[float, float]:
        """
        Analyze hip drop (Trendelenburg sign).
        Returns (left_drop, right_drop) in degrees.
        """
        l_hip = self._get_point(kps, "left_hip")
        r_hip = self._get_point(kps, "right_hip")
        
        if not l_hip or not r_hip:
            return (0, 0)
        
        # Calculate angle from horizontal
        dx = r_hip[0] - l_hip[0]
        dy = r_hip[1] - l_hip[1]
        
        angle = math.degrees(math.atan2(dy, dx))
        
        # Positive angle = left hip higher (right side drop)
        # Negative angle = right hip higher (left side drop)
        left_drop = max(0, -angle)
        right_drop = max(0, angle)
        
        return (left_drop, right_drop)
    
    def _check_arm_crossover(self, kps: Dict) -> bool:
        """
        Check if arms cross the body midline.
        Returns True if crossover detected.
        """
        l_elbow = self._get_point(kps, "left_elbow")
        r_elbow = self._get_point(kps, "right_elbow")
        l_wrist = self._get_point(kps, "left_wrist")
        r_wrist = self._get_point(kps, "right_wrist")
        l_shoulder = self._get_point(kps, "left_shoulder")
        r_shoulder = self._get_point(kps, "right_shoulder")
        
        if not all([l_wrist, r_wrist, l_shoulder, r_shoulder]):
            return False
        
        # Body midline
        midline_x = (l_shoulder[0] + r_shoulder[0]) / 2
        
        # Check if wrists cross midline
        left_crossed = l_wrist[0] > midline_x
        right_crossed = r_wrist[0] < midline_x
        
        return left_crossed or right_crossed
    
    def _track_contact_time(self, kps: Dict, frame_idx: int) -> None:
        """Track ground contact time for each foot."""
        l_ankle = self._get_point(kps, "left_ankle")
        r_ankle = self._get_point(kps, "right_ankle")
        l_hip = self._get_point(kps, "left_hip")
        r_hip = self._get_point(kps, "right_hip")
        
        if not all([l_ankle, r_ankle, l_hip, r_hip]):
            return
        
        hip_y = (l_hip[1] + r_hip[1]) / 2
        
        # Left foot contact
        left_on_ground = l_ankle[1] > hip_y + 50
        if left_on_ground and self.left_contact_start is None:
            self.left_contact_start = frame_idx
        elif not left_on_ground and self.left_contact_start is not None:
            contact_time = (frame_idx - self.left_contact_start) / self.fps * 1000
            if 50 < contact_time < 500:  # Valid range
                self.contact_times_left.append(contact_time)
            self.left_contact_start = None
        
        # Right foot contact
        right_on_ground = r_ankle[1] > hip_y + 50
        if right_on_ground and self.right_contact_start is None:
            self.right_contact_start = frame_idx
        elif not right_on_ground and self.right_contact_start is not None:
            contact_time = (frame_idx - self.right_contact_start) / self.fps * 1000
            if 50 < contact_time < 500:
                self.contact_times_right.append(contact_time)
            self.right_contact_start = None
    
    def _calculate_stats(self, knee_left, knee_right, forward_leans,
                        arm_left, arm_right, total_frames) -> RunningAnalysis:
        """Calculate final running statistics."""
        
        duration = total_frames / self.fps
        
        left_steps = len([s for s in self.steps if s.side == "left"])
        right_steps = len([s for s in self.steps if s.side == "right"])
        total_steps = left_steps + right_steps
        
        # Cadence (steps per minute)
        cadence = (total_steps / duration) * 60 if duration > 0 else 0
        
        # Average knee lift
        all_knee = knee_left + knee_right
        avg_knee = np.mean(all_knee) if all_knee else 0
        max_knee = max(all_knee) if all_knee else 0
        
        # Forward lean
        avg_lean = np.mean(forward_leans) if forward_leans else 0
        
        # Arm symmetry
        if arm_left and arm_right:
            left_range = max(arm_left) - min(arm_left) if arm_left else 0
            right_range = max(arm_right) - min(arm_right) if arm_right else 0
            if max(left_range, right_range) > 0:
                arm_symmetry = min(left_range, right_range) / max(left_range, right_range) * 100
            else:
                arm_symmetry = 100
        else:
            arm_symmetry = 0
        
        # Vertical oscillation
        if self.hip_heights:
            vertical_osc = max(self.hip_heights) - min(self.hip_heights)
        else:
            vertical_osc = 0
        
        # Phase distribution
        phase_counts = {}
        for phase in self.phases:
            phase_counts[phase.value] = phase_counts.get(phase.value, 0) + 1
        
        total_phases = len(self.phases)
        phase_dist = {k: (v / total_phases * 100) if total_phases > 0 else 0 
                     for k, v in phase_counts.items()}
        
        # Ground contact ratio
        stance_pct = phase_dist.get("Stance", 0) + phase_dist.get("Loading", 0) + phase_dist.get("Push-off", 0)
        
        # =====================================================================
        # NEW: Advanced metrics calculation
        # =====================================================================
        
        # Foot strike analysis
        avg_foot_angle = np.mean(self.foot_strike_angles) if self.foot_strike_angles else 0
        if avg_foot_angle > 10:
            foot_strike_type = "heel"
            foot_strike_score = max(0, 100 - abs(avg_foot_angle) * 2)
        elif avg_foot_angle < -5:
            foot_strike_type = "forefoot"
            foot_strike_score = 90  # Generally good
        else:
            foot_strike_type = "midfoot"
            foot_strike_score = 100  # Optimal
        
        # Overstriding
        avg_foot_ahead = np.mean(self.foot_ahead_distances) if self.foot_ahead_distances else 0
        overstriding = avg_foot_ahead < -30  # Foot significantly ahead
        overstriding_score = max(0, 100 + avg_foot_ahead) if avg_foot_ahead < 0 else 100
        
        # Hip drop
        avg_hip_drop_left = np.mean(self.hip_drops_left) if self.hip_drops_left else 0
        avg_hip_drop_right = np.mean(self.hip_drops_right) if self.hip_drops_right else 0
        max_hip_drop = max(avg_hip_drop_left, avg_hip_drop_right)
        hip_drop_score = max(0, 100 - max_hip_drop * 5)
        
        # Arm crossover
        crossover_count = sum(1 for c in self.arm_crossovers if c)
        crossover_pct = (crossover_count / len(self.arm_crossovers) * 100) if self.arm_crossovers else 0
        arm_crossover_detected = crossover_pct > 30
        
        # Arm swing range
        arm_range = 0
        if arm_left and arm_right:
            arm_range = (max(arm_left) - min(arm_left) + max(arm_right) - min(arm_right)) / 2
        
        # Contact time
        all_contact = self.contact_times_left + self.contact_times_right
        avg_contact_time = np.mean(all_contact) if all_contact else 0
        left_contact = np.mean(self.contact_times_left) if self.contact_times_left else 0
        right_contact = np.mean(self.contact_times_right) if self.contact_times_right else 0
        
        # Bounce score (less vertical oscillation = better efficiency)
        bounce_score = max(0, 100 - vertical_osc * 0.5) if vertical_osc > 0 else 50
        
        # Overall efficiency score
        efficiency_score = (
            foot_strike_score * 0.2 +
            overstriding_score * 0.2 +
            hip_drop_score * 0.2 +
            arm_symmetry * 0.2 +
            bounce_score * 0.2
        )
        
        # Injury risk score (lower = better)
        injury_risk = 0
        if overstriding:
            injury_risk += 30
        if max_hip_drop > 5:
            injury_risk += 25
        if avg_foot_angle > 15:  # Severe heel strike
            injury_risk += 20
        if avg_contact_time > 300:  # Long contact time
            injury_risk += 15
        if arm_crossover_detected:
            injury_risk += 10
        
        return RunningAnalysis(
            total_steps=total_steps,
            left_steps=left_steps,
            right_steps=right_steps,
            cadence=round(cadence, 1),
            avg_knee_lift=round(avg_knee, 1),
            max_knee_lift=round(max_knee, 1),
            avg_vertical_osc=round(vertical_osc, 1),
            forward_lean=round(avg_lean, 1),
            arm_symmetry=round(arm_symmetry, 1),
            ground_contact_ratio=round(stance_pct, 1),
            steps=self.steps,
            phase_distribution=phase_dist,
            duration_sec=round(duration, 1),
            # NEW metrics
            foot_strike_type=foot_strike_type,
            foot_strike_angle=round(avg_foot_angle, 1),
            foot_strike_score=round(foot_strike_score, 1),
            overstriding_detected=overstriding,
            overstriding_score=round(overstriding_score, 1),
            avg_foot_ahead_distance=round(avg_foot_ahead, 1),
            hip_drop_left=round(avg_hip_drop_left, 1),
            hip_drop_right=round(avg_hip_drop_right, 1),
            hip_drop_score=round(hip_drop_score, 1),
            arm_crossover_detected=arm_crossover_detected,
            arm_swing_range=round(arm_range, 1),
            avg_contact_time_ms=round(avg_contact_time, 1),
            contact_time_left=round(left_contact, 1),
            contact_time_right=round(right_contact, 1),
            bounce_score=round(bounce_score, 1),
            efficiency_score=round(efficiency_score, 1),
            injury_risk_score=round(injury_risk, 1),
        )
    
    def _empty_analysis(self) -> RunningAnalysis:
        """Return empty analysis."""
        return RunningAnalysis()
    
    def draw_overlay(self, frame: np.ndarray, kps: Dict, 
                    analysis: RunningAnalysis = None) -> np.ndarray:
        """Draw running metrics overlay on frame."""
        h, w = frame.shape[:2]
        
        # Draw skeleton
        connections = [
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_elbow"),
            ("right_shoulder", "right_elbow"),
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            ("left_hip", "left_knee"),
            ("right_hip", "right_knee"),
            ("left_knee", "left_ankle"),
            ("right_knee", "right_ankle"),
        ]
        
        for start, end in connections:
            p1 = self._get_point(kps, start)
            p2 = self._get_point(kps, end)
            if p1 and p2:
                pt1 = (int(p1[0]), int(p1[1]))
                pt2 = (int(p2[0]), int(p2[1]))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
        
        # Draw keypoints
        for name in self.LANDMARKS.keys():
            pt = self._get_point(kps, name)
            if pt:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
        
        # Draw metrics panel
        if analysis:
            panel_h = 120
            cv2.rectangle(frame, (10, h - panel_h - 10), (300, h - 10), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, h - panel_h - 10), (300, h - 10), (0, 255, 255), 2)
            
            y = h - panel_h + 20
            cv2.putText(frame, f"Cadence: {analysis.cadence} spm", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y += 25
            cv2.putText(frame, f"Knee Lift: {analysis.avg_knee_lift:.0f}°", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y += 25
            cv2.putText(frame, f"Forward Lean: {analysis.forward_lean:.1f}°", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y += 25
            cv2.putText(frame, f"Arm Symmetry: {analysis.arm_symmetry:.0f}%", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame


def generate_running_chart(analysis: RunningAnalysis, output_path: str):
    """Generate running analysis charts."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("🏃 Running Analysis", fontsize=14, fontweight='bold')
    
    # 1. Step distribution
    ax1 = axes[0, 0]
    steps = [analysis.left_steps, analysis.right_steps]
    colors = ['#f59e0b', '#10b981']
    ax1.bar(['Ліва нога', 'Права нога'], steps, color=colors)
    ax1.set_title('Розподіл кроків')
    ax1.set_ylabel('Кроків')
    
    # 2. Key metrics
    ax2 = axes[0, 1]
    metrics = ['Cadence\n(spm)', 'Knee Lift\n(°)', 'Arm Sym.\n(%)', 'Ground\n(%)']
    values = [analysis.cadence, analysis.avg_knee_lift, 
              analysis.arm_symmetry, analysis.ground_contact_ratio]
    bars = ax2.bar(metrics, values, color=['#3b82f6', '#10b981', '#f59e0b', '#ef4444'])
    ax2.set_title('Ключові метрики')
    
    # 3. Phase distribution
    ax3 = axes[1, 0]
    if analysis.phase_distribution:
        phases = list(analysis.phase_distribution.keys())
        pcts = list(analysis.phase_distribution.values())
        ax3.pie(pcts, labels=phases, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Фази бігу')
    
    # 4. Recommendations
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    recs = []
    if analysis.cadence < 170:
        recs.append("⚠️ Cadence низький (<170). Спробуйте частіші кроки.")
    elif analysis.cadence > 190:
        recs.append("✅ Cadence оптимальний (>190)")
    
    if analysis.avg_knee_lift < 30:
        recs.append("⚠️ Низький підйом коліна. Працюйте над технікою.")
    
    if analysis.forward_lean < 5:
        recs.append("⚠️ Занадто вертикальна постава. Нахил 8-15° оптимальний.")
    elif analysis.forward_lean > 20:
        recs.append("⚠️ Занадто великий нахил вперед.")
    
    if analysis.arm_symmetry < 80:
        recs.append("⚠️ Асиметрія рук. Слідкуйте за рівномірним махом.")
    
    if not recs:
        recs.append("✅ Техніка бігу в нормі!")
    
    rec_text = "\n".join(recs)
    ax4.text(0.1, 0.9, "Рекомендації:", fontsize=12, fontweight='bold',
            transform=ax4.transAxes, va='top')
    ax4.text(0.1, 0.75, rec_text, fontsize=10, transform=ax4.transAxes, va='top',
            wrap=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='#1a1a2e', edgecolor='none')
    plt.close()
    
    return output_path
