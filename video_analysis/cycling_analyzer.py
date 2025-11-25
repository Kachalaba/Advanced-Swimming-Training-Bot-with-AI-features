"""
🚴 Cycling Biomechanics Analyzer

Features:
- Cadence (RPM)
- Pedal stroke analysis
- Knee angle at top/bottom
- Hip angle (aero position)
- Upper body stability
- Power phase detection
- Left/Right balance
"""

import cv2
import numpy as np
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PedalPhase(Enum):
    """Pedal stroke phases."""
    POWER = "Power"           # 12-5 o'clock (pushing down)
    TRANSITION_BOTTOM = "Bottom"  # 5-7 o'clock
    RECOVERY = "Recovery"     # 7-11 o'clock (pulling up)
    TRANSITION_TOP = "Top"    # 11-12 o'clock
    UNKNOWN = "Unknown"


@dataclass
class PedalStroke:
    """Single pedal revolution data."""
    frame_start: int
    frame_end: int
    side: str  # "left" or "right"
    duration_sec: float = 0
    min_knee_angle: float = 0  # At bottom
    max_knee_angle: float = 0  # At top


@dataclass
class CyclingAnalysis:
    """Cycling analysis results."""
    total_revolutions: int = 0
    left_revolutions: int = 0
    right_revolutions: int = 0
    
    cadence: float = 0  # RPM
    
    avg_knee_angle_top: float = 0  # degrees at top of stroke
    avg_knee_angle_bottom: float = 0  # degrees at bottom
    knee_range: float = 0  # Total range of motion
    
    avg_hip_angle: float = 0  # Aero position
    
    upper_body_stability: float = 0  # percentage (less movement = better)
    
    left_right_balance: float = 50  # percentage (50 = perfect balance)
    
    power_phase_pct: float = 0  # percentage of stroke in power phase
    
    pedal_strokes: List[PedalStroke] = field(default_factory=list)
    phase_distribution: Dict[str, float] = field(default_factory=dict)
    
    duration_sec: float = 0
    
    # Position analysis
    saddle_height_score: float = 0  # Based on knee extension
    aero_score: float = 0  # Based on hip angle


class CyclingAnalyzer:
    """Analyze cycling biomechanics from video."""
    
    # MediaPipe landmark indices
    LANDMARKS = {
        "nose": 0,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
    }
    
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.pedal_strokes: List[PedalStroke] = []
        self.phases: List[PedalPhase] = []
        
    def analyze(self, keypoints_list: List[Dict], fps: float = None) -> CyclingAnalysis:
        """
        Analyze cycling from keypoints sequence.
        
        Args:
            keypoints_list: List of keypoints dicts per frame
            fps: Frames per second
            
        Returns:
            CyclingAnalysis with metrics
        """
        if fps:
            self.fps = fps
            
        if not keypoints_list or len(keypoints_list) < 10:
            return self._empty_analysis()
        
        self.pedal_strokes = []
        self.phases = []
        
        # Track metrics per frame
        knee_angles_left = []
        knee_angles_right = []
        hip_angles = []
        shoulder_positions = []  # For stability
        
        # For revolution detection
        prev_left_knee_y = None
        prev_right_knee_y = None
        left_rev_start = None
        right_rev_start = None
        left_going_down = False
        right_going_down = False
        
        left_min_knee = 180
        left_max_knee = 0
        right_min_knee = 180
        right_max_knee = 0
        
        for frame_idx, kps in enumerate(keypoints_list):
            if not kps:
                self.phases.append(PedalPhase.UNKNOWN)
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
            
            # Calculate knee angles
            if l_hip and l_knee and l_ankle:
                knee_angle = self._calculate_angle(l_hip, l_knee, l_ankle)
                knee_angles_left.append(knee_angle)
                left_min_knee = min(left_min_knee, knee_angle)
                left_max_knee = max(left_max_knee, knee_angle)
            
            if r_hip and r_knee and r_ankle:
                knee_angle = self._calculate_angle(r_hip, r_knee, r_ankle)
                knee_angles_right.append(knee_angle)
                right_min_knee = min(right_min_knee, knee_angle)
                right_max_knee = max(right_max_knee, knee_angle)
            
            # Calculate hip angle (torso angle for aero position)
            if l_shoulder and r_shoulder and l_hip and r_hip:
                shoulder_center = ((l_shoulder[0] + r_shoulder[0]) / 2,
                                  (l_shoulder[1] + r_shoulder[1]) / 2)
                hip_center = ((l_hip[0] + r_hip[0]) / 2,
                             (l_hip[1] + r_hip[1]) / 2)
                
                # Angle from horizontal (lower = more aero)
                dx = shoulder_center[0] - hip_center[0]
                dy = hip_center[1] - shoulder_center[1]
                hip_angle = math.degrees(math.atan2(dy, abs(dx) + 0.001))
                hip_angles.append(hip_angle)
                
                # Track shoulder position for stability
                shoulder_positions.append(shoulder_center)
            
            # Revolution detection based on knee Y position cycle
            if l_knee:
                if prev_left_knee_y is not None:
                    if l_knee[1] > prev_left_knee_y and not left_going_down:
                        # Starting downstroke
                        left_going_down = True
                        if left_rev_start is not None:
                            # Complete previous revolution
                            stroke = PedalStroke(
                                frame_start=left_rev_start,
                                frame_end=frame_idx,
                                side="left",
                                duration_sec=(frame_idx - left_rev_start) / self.fps,
                                min_knee_angle=left_min_knee,
                                max_knee_angle=left_max_knee
                            )
                            self.pedal_strokes.append(stroke)
                            left_min_knee = 180
                            left_max_knee = 0
                        left_rev_start = frame_idx
                    elif l_knee[1] < prev_left_knee_y and left_going_down:
                        left_going_down = False
                
                prev_left_knee_y = l_knee[1]
            
            if r_knee:
                if prev_right_knee_y is not None:
                    if r_knee[1] > prev_right_knee_y and not right_going_down:
                        right_going_down = True
                        if right_rev_start is not None:
                            stroke = PedalStroke(
                                frame_start=right_rev_start,
                                frame_end=frame_idx,
                                side="right",
                                duration_sec=(frame_idx - right_rev_start) / self.fps,
                                min_knee_angle=right_min_knee,
                                max_knee_angle=right_max_knee
                            )
                            self.pedal_strokes.append(stroke)
                            right_min_knee = 180
                            right_max_knee = 0
                        right_rev_start = frame_idx
                    elif r_knee[1] < prev_right_knee_y and right_going_down:
                        right_going_down = False
                
                prev_right_knee_y = r_knee[1]
            
            # Detect pedal phase
            phase = self._detect_phase(kps)
            self.phases.append(phase)
        
        # Calculate final metrics
        return self._calculate_stats(
            knee_angles_left, knee_angles_right,
            hip_angles, shoulder_positions,
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
    
    def _detect_phase(self, kps: Dict) -> PedalPhase:
        """Detect current pedal phase based on leg position."""
        l_knee = self._get_point(kps, "left_knee")
        l_hip = self._get_point(kps, "left_hip")
        l_ankle = self._get_point(kps, "left_ankle")
        
        if not all([l_knee, l_hip, l_ankle]):
            return PedalPhase.UNKNOWN
        
        # Determine phase based on knee position relative to hip
        knee_rel_y = l_knee[1] - l_hip[1]
        ankle_rel_y = l_ankle[1] - l_hip[1]
        
        if knee_rel_y > 50 and ankle_rel_y > 0:
            return PedalPhase.POWER
        elif knee_rel_y < -30:
            return PedalPhase.RECOVERY
        elif ankle_rel_y > knee_rel_y:
            return PedalPhase.TRANSITION_BOTTOM
        else:
            return PedalPhase.TRANSITION_TOP
    
    def _calculate_stats(self, knee_left, knee_right, hip_angles,
                        shoulder_positions, total_frames) -> CyclingAnalysis:
        """Calculate final cycling statistics."""
        
        duration = total_frames / self.fps
        
        left_revs = len([s for s in self.pedal_strokes if s.side == "left"])
        right_revs = len([s for s in self.pedal_strokes if s.side == "right"])
        total_revs = left_revs + right_revs
        
        # Cadence (RPM) - revolutions per minute
        # Each leg does half the revolutions
        cadence = (total_revs / duration) * 30 if duration > 0 else 0  # *30 because we count each leg
        
        # Knee angles
        all_knee = knee_left + knee_right
        avg_knee_bottom = min(all_knee) if all_knee else 0  # Most extended
        avg_knee_top = max(all_knee) if all_knee else 0  # Most flexed
        knee_range = avg_knee_top - avg_knee_bottom
        
        # Hip angle (aero position)
        avg_hip = np.mean(hip_angles) if hip_angles else 0
        
        # Upper body stability (less movement = better)
        if len(shoulder_positions) > 1:
            x_coords = [p[0] for p in shoulder_positions]
            y_coords = [p[1] for p in shoulder_positions]
            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)
            # Score: less movement = higher score
            movement = math.sqrt(x_range**2 + y_range**2)
            stability = max(0, 100 - movement / 2)  # Rough scaling
        else:
            stability = 0
        
        # Left/Right balance
        if total_revs > 0:
            balance = (left_revs / total_revs) * 100
        else:
            balance = 50
        
        # Phase distribution
        phase_counts = {}
        for phase in self.phases:
            phase_counts[phase.value] = phase_counts.get(phase.value, 0) + 1
        
        total_phases = len(self.phases)
        phase_dist = {k: (v / total_phases * 100) if total_phases > 0 else 0 
                     for k, v in phase_counts.items()}
        
        power_phase_pct = phase_dist.get("Power", 0)
        
        # Saddle height score (based on knee extension at bottom)
        # Optimal: 140-150 degrees at bottom
        if avg_knee_bottom > 0:
            if 140 <= avg_knee_bottom <= 150:
                saddle_score = 100
            elif 130 <= avg_knee_bottom < 140 or 150 < avg_knee_bottom <= 160:
                saddle_score = 80
            else:
                saddle_score = 60
        else:
            saddle_score = 0
        
        # Aero score (based on hip angle)
        # Lower angle = more aero, optimal 30-45 degrees
        if hip_angles:
            if 30 <= avg_hip <= 45:
                aero_score = 100
            elif 45 < avg_hip <= 60:
                aero_score = 80
            elif avg_hip < 30:
                aero_score = 90  # Very aero but might be uncomfortable
            else:
                aero_score = 60
        else:
            aero_score = 0
        
        return CyclingAnalysis(
            total_revolutions=total_revs,
            left_revolutions=left_revs,
            right_revolutions=right_revs,
            cadence=round(cadence, 1),
            avg_knee_angle_top=round(avg_knee_top, 1),
            avg_knee_angle_bottom=round(avg_knee_bottom, 1),
            knee_range=round(knee_range, 1),
            avg_hip_angle=round(avg_hip, 1),
            upper_body_stability=round(stability, 1),
            left_right_balance=round(balance, 1),
            power_phase_pct=round(power_phase_pct, 1),
            pedal_strokes=self.pedal_strokes,
            phase_distribution=phase_dist,
            duration_sec=round(duration, 1),
            saddle_height_score=round(saddle_score, 1),
            aero_score=round(aero_score, 1)
        )
    
    def _empty_analysis(self) -> CyclingAnalysis:
        """Return empty analysis."""
        return CyclingAnalysis()
    
    def draw_overlay(self, frame: np.ndarray, kps: Dict,
                    analysis: CyclingAnalysis = None) -> np.ndarray:
        """Draw cycling metrics overlay on frame."""
        h, w = frame.shape[:2]
        
        # Draw skeleton
        connections = [
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_elbow"),
            ("right_shoulder", "right_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_elbow", "right_wrist"),
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
                cv2.line(frame, pt1, pt2, (255, 165, 0), 2)  # Orange for cycling
        
        # Draw keypoints
        for name in self.LANDMARKS.keys():
            pt = self._get_point(kps, name)
            if pt:
                color = (0, 255, 0)
                if "knee" in name:
                    color = (0, 255, 255)  # Highlight knees
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, color, -1)
        
        # Draw metrics panel
        if analysis:
            panel_h = 140
            cv2.rectangle(frame, (10, h - panel_h - 10), (320, h - 10), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, h - panel_h - 10), (320, h - 10), (255, 165, 0), 2)
            
            y = h - panel_h + 20
            cv2.putText(frame, f"Cadence: {analysis.cadence:.0f} RPM", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y += 25
            cv2.putText(frame, f"Knee Range: {analysis.knee_range:.0f}°", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y += 25
            cv2.putText(frame, f"Hip Angle: {analysis.avg_hip_angle:.0f}°", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y += 25
            cv2.putText(frame, f"Stability: {analysis.upper_body_stability:.0f}%", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y += 25
            cv2.putText(frame, f"L/R Balance: {analysis.left_right_balance:.0f}%", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame


def generate_cycling_chart(analysis: CyclingAnalysis, output_path: str):
    """Generate cycling analysis charts."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("🚴 Cycling Analysis", fontsize=14, fontweight='bold')
    
    # 1. Revolution balance
    ax1 = axes[0, 0]
    revs = [analysis.left_revolutions, analysis.right_revolutions]
    colors = ['#f59e0b', '#10b981']
    ax1.bar(['Ліва нога', 'Права нога'], revs, color=colors)
    ax1.set_title('Баланс обертів')
    ax1.set_ylabel('Обертів')
    
    # 2. Key metrics
    ax2 = axes[0, 1]
    metrics = ['Cadence\n(RPM)', 'Knee Range\n(°)', 'Stability\n(%)', 'Aero\nScore']
    values = [analysis.cadence, analysis.knee_range,
              analysis.upper_body_stability, analysis.aero_score]
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']
    ax2.bar(metrics, values, color=colors)
    ax2.set_title('Ключові метрики')
    
    # 3. Phase distribution
    ax3 = axes[1, 0]
    if analysis.phase_distribution:
        phases = list(analysis.phase_distribution.keys())
        pcts = list(analysis.phase_distribution.values())
        colors_pie = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#6b7280']
        ax3.pie(pcts, labels=phases, autopct='%1.1f%%', colors=colors_pie[:len(phases)])
    ax3.set_title('Фази педалювання')
    
    # 4. Position analysis & recommendations
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    recs = []
    
    # Saddle height
    if analysis.avg_knee_angle_bottom < 140:
        recs.append("⚠️ Сідло занизько. Коліно надто зігнуте внизу.")
    elif analysis.avg_knee_angle_bottom > 155:
        recs.append("⚠️ Сідло зависоко. Нога надто випрямлена.")
    else:
        recs.append("✅ Висота сідла оптимальна")
    
    # Cadence
    if analysis.cadence < 80:
        recs.append("⚠️ Каденс низький (<80). Спробуйте легшу передачу.")
    elif analysis.cadence > 110:
        recs.append("⚠️ Каденс високий (>110). Можливо важча передача.")
    else:
        recs.append("✅ Каденс в оптимальному діапазоні (80-100)")
    
    # Stability
    if analysis.upper_body_stability < 70:
        recs.append("⚠️ Нестабільний корпус. Працюйте над core.")
    
    # Balance
    if abs(analysis.left_right_balance - 50) > 10:
        recs.append(f"⚠️ Дисбаланс L/R: {analysis.left_right_balance:.0f}%")
    
    rec_text = "\n".join(recs)
    ax4.text(0.1, 0.95, "Bike Fit & Рекомендації:", fontsize=12, fontweight='bold',
            transform=ax4.transAxes, va='top')
    ax4.text(0.1, 0.80, rec_text, fontsize=10, transform=ax4.transAxes, va='top',
            wrap=True)
    
    # Scores
    ax4.text(0.1, 0.35, f"Saddle Score: {analysis.saddle_height_score:.0f}/100", 
            fontsize=11, transform=ax4.transAxes)
    ax4.text(0.1, 0.25, f"Aero Score: {analysis.aero_score:.0f}/100",
            fontsize=11, transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    plt.close()
    
    return output_path
