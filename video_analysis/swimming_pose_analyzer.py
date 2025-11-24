"""
Swimming-specific pose analysis with rotation compensation.

Features:
- Rotated pose detection (compensates for horizontal swimmer)
- Body part segmentation (head, torso, arms, legs)
- Spine axis analysis with deviation metrics
- Underwater image enhancement
- Swimming-specific biomechanics
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

import cv2
import numpy as np
import mediapipe as mp

logger = logging.getLogger(__name__)


@dataclass
class Point2D:
    """2D point with visibility."""
    x: float
    y: float
    visibility: float = 1.0
    
    def to_tuple(self) -> Tuple[int, int]:
        return (int(self.x), int(self.y))


@dataclass
class BodyPart:
    """Represents a body part segment."""
    name: str
    keypoints: Dict[str, Point2D]
    bbox: Optional[List[int]] = None
    center: Optional[Point2D] = None
    angle: float = 0.0
    visibility: float = 0.0


@dataclass 
class SpineAnalysis:
    """Spine axis analysis results."""
    center_line: List[Point2D] = field(default_factory=list)
    total_angle: float = 0.0
    deviation: float = 0.0
    curvature: float = 0.0
    segments: List[Dict] = field(default_factory=list)


@dataclass
class SwimmingMetrics:
    """Swimming-specific biomechanical metrics."""
    body_roll: float = 0.0
    hip_drop: float = 0.0
    head_position: float = 0.0
    streamline_score: float = 0.0
    left_arm_angle: float = 0.0
    right_arm_angle: float = 0.0
    left_leg_angle: float = 0.0
    right_leg_angle: float = 0.0
    kick_amplitude: float = 0.0


class SwimmingPoseAnalyzer:
    """Advanced pose analyzer for horizontal swimmers."""
    
    LANDMARKS = {
        "nose": 0, "left_eye": 2, "right_eye": 5,
        "left_ear": 7, "right_ear": 8,
        "left_shoulder": 11, "right_shoulder": 12,
        "left_elbow": 13, "right_elbow": 14,
        "left_wrist": 15, "right_wrist": 16,
        "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26,
        "left_ankle": 27, "right_ankle": 28,
    }
    
    BODY_PARTS = {
        "head": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
        "torso": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
        "left_arm": ["left_shoulder", "left_elbow", "left_wrist"],
        "right_arm": ["right_shoulder", "right_elbow", "right_wrist"],
        "left_leg": ["left_hip", "left_knee", "left_ankle"],
        "right_leg": ["right_hip", "right_knee", "right_ankle"],
    }
    
    COLORS = {
        "head": (255, 0, 255),
        "torso": (0, 255, 255),
        "left_arm": (255, 100, 0),
        "right_arm": (255, 150, 50),
        "left_leg": (0, 255, 0),
        "right_leg": (50, 255, 50),
        "spine": (0, 200, 255),
    }
    
    def __init__(self, model_complexity: int = 2, min_confidence: float = 0.3):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=model_complexity,
            enable_segmentation=True,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        logger.info(f"SwimmingPoseAnalyzer initialized")
    
    def analyze_frame(
        self,
        frame_or_path: Any,
        swimmer_bbox: Optional[List[int]] = None,
    ) -> Dict:
        """Analyze single frame for swimming pose."""
        # Load frame
        if isinstance(frame_or_path, str):
            frame = cv2.imread(frame_or_path)
            if frame is None:
                return self._empty_result()
        else:
            frame = frame_or_path.copy()
        
        original_shape = frame.shape[:2]
        crop_offset = [0, 0]
        
        # Crop to swimmer
        if swimmer_bbox:
            x1, y1, x2, y2 = swimmer_bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 > x1 and y2 > y1:
                frame = frame[y1:y2, x1:x2]
                crop_offset = [x1, y1]
        
        # Detect pose with rotation
        keypoints, method = self._detect_with_rotation(frame, swimmer_bbox)
        
        if not keypoints:
            return self._empty_result()
        
        # Adjust for crop
        for name, kp in keypoints.items():
            kp.x += crop_offset[0]
            kp.y += crop_offset[1]
        
        # Analyze
        body_parts = self._segment_body_parts(keypoints)
        spine = self._analyze_spine(keypoints)
        metrics = self._calc_metrics(keypoints, spine)
        
        return {
            "has_pose": True,
            "method": method,
            "keypoints": {k: {"x": v.x, "y": v.y, "vis": v.visibility} for k, v in keypoints.items()},
            "body_parts": {k: self._part_to_dict(v) for k, v in body_parts.items()},
            "spine": {"angle": spine.total_angle, "deviation": spine.deviation, 
                     "curvature": spine.curvature, "points": len(spine.center_line)},
            "metrics": vars(metrics),
        }
    
    def _detect_with_rotation(self, frame: np.ndarray, bbox: Optional[List[int]]) -> Tuple:
        """Detect pose with rotation compensation."""
        best_kps, best_vis, best_method = None, 0, "none"
        
        # Estimate orientation
        angles = [0]
        if bbox:
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if w > h * 1.3:  # Horizontal
                angles.extend([90, -90])
            elif h > w * 1.3:  # Vertical
                angles.extend([0])
        else:
            angles.extend([90, -90, 180])
        
        for angle in angles:
            rotated = self._rotate(frame, angle) if angle != 0 else frame
            enhanced = self._enhance_underwater(rotated)
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            
            results = self.pose.process(rgb)
            
            if results.pose_landmarks:
                kps = self._extract_keypoints(results.pose_landmarks, rotated.shape[:2])
                avg_vis = np.mean([kp.visibility for kp in kps.values()])
                
                if avg_vis > best_vis:
                    if angle != 0:
                        kps = self._rotate_keypoints(kps, -angle, frame.shape[:2])
                    best_kps, best_vis = kps, avg_vis
                    best_method = f"rot_{angle}" if angle else "direct"
        
        return best_kps, best_method
    
    def _rotate(self, frame: np.ndarray, angle: float) -> np.ndarray:
        """Rotate frame."""
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        cos, sin = abs(M[0,0]), abs(M[0,1])
        nw, nh = int(h*sin + w*cos), int(h*cos + w*sin)
        M[0,2] += (nw-w)/2
        M[1,2] += (nh-h)/2
        return cv2.warpAffine(frame, M, (nw, nh))
    
    def _rotate_keypoints(self, kps: Dict, angle: float, shape: Tuple) -> Dict:
        """Rotate keypoints back."""
        h, w = shape
        cx, cy = w/2, h/2
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        
        rotated = {}
        for name, kp in kps.items():
            px, py = kp.x - cx, kp.y - cy
            nx = px * cos_a - py * sin_a + cx
            ny = px * sin_a + py * cos_a + cy
            rotated[name] = Point2D(nx, ny, kp.visibility)
        return rotated
    
    def _enhance_underwater(self, frame: np.ndarray) -> np.ndarray:
        """Enhance underwater frames."""
        if frame is None or frame.size == 0:
            return frame
        
        # Balance colors
        b, g, r = cv2.split(frame)
        r = cv2.add(r, 15)
        b = cv2.subtract(b, 10)
        balanced = cv2.merge([b, g, r])
        
        # CLAHE
        lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
        l, a, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b_ch]), cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _extract_keypoints(self, landmarks, shape: Tuple) -> Dict[str, Point2D]:
        """Extract keypoints from MediaPipe."""
        h, w = shape
        kps = {}
        for name, idx in self.LANDMARKS.items():
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                kps[name] = Point2D(lm.x * w, lm.y * h, lm.visibility)
        return kps
    
    def _segment_body_parts(self, keypoints: Dict) -> Dict[str, BodyPart]:
        """Segment body into parts."""
        parts = {}
        for name, kp_names in self.BODY_PARTS.items():
            part_kps = {k: keypoints[k] for k in kp_names if k in keypoints}
            if not part_kps:
                continue
            
            xs = [kp.x for kp in part_kps.values()]
            ys = [kp.y for kp in part_kps.values()]
            
            part = BodyPart(
                name=name,
                keypoints=part_kps,
                bbox=[int(min(xs))-5, int(min(ys))-5, int(max(xs))+5, int(max(ys))+5],
                center=Point2D(sum(xs)/len(xs), sum(ys)/len(ys)),
                visibility=np.mean([kp.visibility for kp in part_kps.values()]),
            )
            parts[name] = part
        return parts
    
    def _analyze_spine(self, kps: Dict) -> SpineAnalysis:
        """Analyze spine axis."""
        spine = SpineAnalysis()
        
        # Spine points
        points = []
        
        if "nose" in kps and kps["nose"].visibility > 0.3:
            points.append(kps["nose"])
        
        if "left_shoulder" in kps and "right_shoulder" in kps:
            ls, rs = kps["left_shoulder"], kps["right_shoulder"]
            if ls.visibility > 0.3 and rs.visibility > 0.3:
                points.append(Point2D((ls.x+rs.x)/2, (ls.y+rs.y)/2, (ls.visibility+rs.visibility)/2))
        
        if "left_hip" in kps and "right_hip" in kps:
            lh, rh = kps["left_hip"], kps["right_hip"]
            if lh.visibility > 0.3 and rh.visibility > 0.3:
                points.append(Point2D((lh.x+rh.x)/2, (lh.y+rh.y)/2, (lh.visibility+rh.visibility)/2))
        
        if "left_knee" in kps and "right_knee" in kps:
            lk, rk = kps["left_knee"], kps["right_knee"]
            if lk.visibility > 0.3 and rk.visibility > 0.3:
                points.append(Point2D((lk.x+rk.x)/2, (lk.y+rk.y)/2, (lk.visibility+rk.visibility)/2))
        
        spine.center_line = points
        
        if len(points) >= 2:
            dx = points[-1].x - points[0].x
            dy = points[-1].y - points[0].y
            spine.total_angle = math.degrees(math.atan2(dy, dx))
            spine.deviation = abs(spine.total_angle)
            
            # Curvature
            if len(points) >= 3:
                spine.curvature = self._calc_curvature(points)
        
        return spine
    
    def _calc_curvature(self, points: List[Point2D]) -> float:
        """Calculate spine curvature."""
        if len(points) < 3:
            return 0.0
        
        # Fit line and measure deviation
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        
        # Line from first to last point
        dx = xs[-1] - xs[0]
        dy = ys[-1] - ys[0]
        length = math.sqrt(dx*dx + dy*dy)
        
        if length < 1:
            return 0.0
        
        # Max distance from middle points to line
        max_dist = 0
        for i in range(1, len(points)-1):
            # Distance from point to line
            dist = abs(dy*xs[i] - dx*ys[i] + xs[-1]*ys[0] - ys[-1]*xs[0]) / length
            max_dist = max(max_dist, dist)
        
        return max_dist / length  # Normalized curvature
    
    def _calc_metrics(self, kps: Dict, spine: SpineAnalysis) -> SwimmingMetrics:
        """Calculate swimming metrics."""
        metrics = SwimmingMetrics()
        
        # Body roll (shoulder rotation)
        if "left_shoulder" in kps and "right_shoulder" in kps:
            ls, rs = kps["left_shoulder"], kps["right_shoulder"]
            dx = rs.x - ls.x
            dy = rs.y - ls.y
            metrics.body_roll = math.degrees(math.atan2(dy, dx))
        
        # Hip drop
        if all(k in kps for k in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
            shoulder_y = (kps["left_shoulder"].y + kps["right_shoulder"].y) / 2
            hip_y = (kps["left_hip"].y + kps["right_hip"].y) / 2
            metrics.hip_drop = hip_y - shoulder_y
        
        # Head position
        if "nose" in kps and "left_shoulder" in kps and "right_shoulder" in kps:
            head_y = kps["nose"].y
            shoulder_y = (kps["left_shoulder"].y + kps["right_shoulder"].y) / 2
            metrics.head_position = head_y - shoulder_y
        
        # Arm angles
        if all(k in kps for k in ["left_shoulder", "left_elbow", "left_wrist"]):
            metrics.left_arm_angle = self._angle_3p(
                kps["left_shoulder"], kps["left_elbow"], kps["left_wrist"]
            )
        if all(k in kps for k in ["right_shoulder", "right_elbow", "right_wrist"]):
            metrics.right_arm_angle = self._angle_3p(
                kps["right_shoulder"], kps["right_elbow"], kps["right_wrist"]
            )
        
        # Leg angles
        if all(k in kps for k in ["left_hip", "left_knee", "left_ankle"]):
            metrics.left_leg_angle = self._angle_3p(
                kps["left_hip"], kps["left_knee"], kps["left_ankle"]
            )
        if all(k in kps for k in ["right_hip", "right_knee", "right_ankle"]):
            metrics.right_leg_angle = self._angle_3p(
                kps["right_hip"], kps["right_knee"], kps["right_ankle"]
            )
        
        # Kick amplitude
        if "left_ankle" in kps and "right_ankle" in kps:
            metrics.kick_amplitude = abs(kps["left_ankle"].y - kps["right_ankle"].y)
        
        # Streamline score
        metrics.streamline_score = self._calc_streamline_score(spine, metrics)
        
        return metrics
    
    def _angle_3p(self, p1: Point2D, p2: Point2D, p3: Point2D) -> float:
        """Calculate angle at p2."""
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return math.degrees(math.acos(np.clip(cos_a, -1, 1)))
    
    def _calc_streamline_score(self, spine: SpineAnalysis, metrics: SwimmingMetrics) -> float:
        """Calculate streamline score (0-100)."""
        score = 100.0
        
        # Penalize spine deviation
        score -= min(30, spine.deviation * 1.5)
        
        # Penalize curvature
        score -= min(20, spine.curvature * 100)
        
        # Penalize hip drop
        score -= min(20, abs(metrics.hip_drop) * 0.5)
        
        # Penalize body roll (some is good, too much is bad)
        roll = abs(metrics.body_roll)
        if roll > 45:
            score -= (roll - 45) * 0.5
        
        return max(0, min(100, score))
    
    def _part_to_dict(self, part: BodyPart) -> Dict:
        """Convert BodyPart to dict."""
        return {
            "bbox": part.bbox,
            "center": {"x": part.center.x, "y": part.center.y} if part.center else None,
            "visibility": part.visibility,
        }
    
    def _empty_result(self) -> Dict:
        return {"has_pose": False, "keypoints": {}, "body_parts": {}, 
                "spine": {}, "metrics": {}}
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    def draw_analysis(
        self,
        frame_or_path: Any,
        analysis: Dict,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Draw pose analysis overlay on frame.
        
        Features:
        - Body part segmentation with colored bboxes
        - Spine axis line
        - Keypoint skeleton
        - Metrics panel
        """
        # Load frame
        if isinstance(frame_or_path, str):
            frame = cv2.imread(frame_or_path)
            if frame is None:
                return None
        else:
            frame = frame_or_path.copy()
        
        if not analysis.get("has_pose"):
            return frame
        
        keypoints = analysis["keypoints"]
        body_parts = analysis.get("body_parts", {})
        spine_data = analysis.get("spine", {})
        metrics = analysis.get("metrics", {})
        
        # Draw body part bounding boxes
        for part_name, part_data in body_parts.items():
            if part_data.get("bbox"):
                color = self.COLORS.get(part_name, (200, 200, 200))
                x1, y1, x2, y2 = part_data["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, part_name.upper(), (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw skeleton connections
        self._draw_skeleton(frame, keypoints)
        
        # Draw spine axis
        self._draw_spine(frame, keypoints, spine_data)
        
        # Draw metrics panel
        self._draw_metrics_panel(frame, spine_data, metrics)
        
        # Draw legend
        self._draw_legend(frame)
        
        # Save if path provided
        if output_path:
            cv2.imwrite(output_path, frame)
        
        return frame
    
    def _draw_skeleton(self, frame: np.ndarray, keypoints: Dict):
        """Draw skeleton connections."""
        connections = [
            # Head
            ("nose", "left_eye"), ("nose", "right_eye"),
            ("left_eye", "left_ear"), ("right_eye", "right_ear"),
            # Torso
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            # Arms
            ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
            # Legs
            ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
        ]
        
        for p1_name, p2_name in connections:
            if p1_name in keypoints and p2_name in keypoints:
                p1 = keypoints[p1_name]
                p2 = keypoints[p2_name]
                if p1.get("vis", 0) > 0.3 and p2.get("vis", 0) > 0.3:
                    pt1 = (int(p1["x"]), int(p1["y"]))
                    pt2 = (int(p2["x"]), int(p2["y"]))
                    
                    # Color by body part
                    if "shoulder" in p1_name or "hip" in p1_name:
                        color = self.COLORS["torso"]
                    elif "elbow" in p1_name or "wrist" in p1_name:
                        color = self.COLORS["left_arm"] if "left" in p1_name else self.COLORS["right_arm"]
                    elif "knee" in p1_name or "ankle" in p1_name:
                        color = self.COLORS["left_leg"] if "left" in p1_name else self.COLORS["right_leg"]
                    else:
                        color = self.COLORS["head"]
                    
                    cv2.line(frame, pt1, pt2, color, 2)
        
        # Draw keypoints
        for name, kp in keypoints.items():
            if kp.get("vis", 0) > 0.3:
                pt = (int(kp["x"]), int(kp["y"]))
                cv2.circle(frame, pt, 4, (255, 255, 255), -1)
                cv2.circle(frame, pt, 5, (0, 0, 0), 1)
    
    def _draw_spine(self, frame: np.ndarray, keypoints: Dict, spine_data: Dict):
        """Draw spine axis."""
        # Build spine points
        spine_pts = []
        
        if "nose" in keypoints and keypoints["nose"].get("vis", 0) > 0.3:
            spine_pts.append((int(keypoints["nose"]["x"]), int(keypoints["nose"]["y"])))
        
        if "left_shoulder" in keypoints and "right_shoulder" in keypoints:
            ls, rs = keypoints["left_shoulder"], keypoints["right_shoulder"]
            if ls.get("vis", 0) > 0.3 and rs.get("vis", 0) > 0.3:
                spine_pts.append((int((ls["x"]+rs["x"])/2), int((ls["y"]+rs["y"])/2)))
        
        if "left_hip" in keypoints and "right_hip" in keypoints:
            lh, rh = keypoints["left_hip"], keypoints["right_hip"]
            if lh.get("vis", 0) > 0.3 and rh.get("vis", 0) > 0.3:
                spine_pts.append((int((lh["x"]+rh["x"])/2), int((lh["y"]+rh["y"])/2)))
        
        if "left_knee" in keypoints and "right_knee" in keypoints:
            lk, rk = keypoints["left_knee"], keypoints["right_knee"]
            if lk.get("vis", 0) > 0.3 and rk.get("vis", 0) > 0.3:
                spine_pts.append((int((lk["x"]+rk["x"])/2), int((lk["y"]+rk["y"])/2)))
        
        if len(spine_pts) < 2:
            return
        
        # Draw spine line
        spine_color = self.COLORS["spine"]
        for i in range(len(spine_pts) - 1):
            cv2.line(frame, spine_pts[i], spine_pts[i+1], spine_color, 4)
        
        # Draw spine points
        for pt in spine_pts:
            cv2.circle(frame, pt, 6, spine_color, -1)
            cv2.circle(frame, pt, 7, (0, 0, 0), 2)
        
        # Draw ideal horizontal line (dashed)
        if len(spine_pts) >= 2:
            start_y = spine_pts[0][1]
            cv2.line(frame, (spine_pts[0][0], start_y), (spine_pts[-1][0], start_y),
                    (100, 100, 100), 1, cv2.LINE_AA)
        
        # Show spine angle
        angle = spine_data.get("angle", 0)
        deviation = spine_data.get("deviation", 0)
        
        mid_idx = len(spine_pts) // 2
        text_pos = (spine_pts[mid_idx][0] + 15, spine_pts[mid_idx][1])
        
        # Color based on deviation
        if deviation < 10:
            text_color = (0, 255, 0)  # Green - excellent
        elif deviation < 20:
            text_color = (0, 200, 255)  # Orange - good
        else:
            text_color = (0, 0, 255)  # Red - needs work
        
        cv2.putText(frame, f"Spine: {angle:.1f}°", text_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
    
    def _draw_metrics_panel(self, frame: np.ndarray, spine: Dict, metrics: Dict):
        """Draw metrics panel in corner."""
        h, w = frame.shape[:2]
        
        # Panel background
        panel_h = 140
        panel_w = 200
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (panel_w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Metrics text
        y = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Streamline score
        score = metrics.get("streamline_score", 0)
        color = (0, 255, 0) if score > 70 else (0, 200, 255) if score > 50 else (0, 0, 255)
        cv2.putText(frame, f"Streamline: {score:.0f}/100", (15, y), font, 0.5, color, 1)
        y += 22
        
        # Spine deviation
        dev = spine.get("deviation", 0)
        color = (0, 255, 0) if dev < 10 else (0, 200, 255) if dev < 20 else (0, 0, 255)
        cv2.putText(frame, f"Spine Dev: {dev:.1f}°", (15, y), font, 0.5, color, 1)
        y += 22
        
        # Body roll
        roll = metrics.get("body_roll", 0)
        cv2.putText(frame, f"Body Roll: {roll:.1f}°", (15, y), font, 0.5, (255, 255, 255), 1)
        y += 22
        
        # Hip drop
        hip = metrics.get("hip_drop", 0)
        cv2.putText(frame, f"Hip Drop: {hip:.1f}px", (15, y), font, 0.5, (255, 255, 255), 1)
        y += 22
        
        # Kick amplitude
        kick = metrics.get("kick_amplitude", 0)
        cv2.putText(frame, f"Kick Amp: {kick:.0f}px", (15, y), font, 0.5, (255, 255, 255), 1)
    
    def _draw_legend(self, frame: np.ndarray):
        """Draw color legend at bottom."""
        h, w = frame.shape[:2]
        y = h - 25
        x = 10
        
        legend = [
            ("Head", self.COLORS["head"]),
            ("Torso", self.COLORS["torso"]),
            ("Arms", self.COLORS["left_arm"]),
            ("Legs", self.COLORS["left_leg"]),
            ("Spine", self.COLORS["spine"]),
        ]
        
        for label, color in legend:
            cv2.rectangle(frame, (x, y), (x + 12, y + 12), color, -1)
            cv2.putText(frame, label, (x + 16, y + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            x += 60


# Convenience function
def analyze_swimming_pose(
    frame_paths: List,
    detections: Optional[List[Dict]] = None,
    output_dir: str = "./swimming_pose",
) -> Dict:
    """Analyze swimming pose for multiple frames."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    analyzer = SwimmingPoseAnalyzer()
    
    results = []
    for i, frame_info in enumerate(frame_paths):
        path = frame_info["path"] if isinstance(frame_info, dict) else frame_info
        bbox = detections[i].get("bbox") if detections and i < len(detections) else None
        
        analysis = analyzer.analyze_frame(path, bbox)
        analysis["frame_index"] = i
        results.append(analysis)
        
        # Save visualization
        if analysis["has_pose"]:
            out_file = output_path / f"pose_{i:04d}.jpg"
            analyzer.draw_analysis(path, analysis, str(out_file))
    
    # Aggregate
    valid = [r for r in results if r["has_pose"]]
    
    return {
        "frame_analyses": results,
        "detection_rate": len(valid) / len(results) if results else 0,
        "avg_streamline": np.mean([r["metrics"]["streamline_score"] for r in valid]) if valid else 0,
        "avg_deviation": np.mean([r["spine"]["deviation"] for r in valid]) if valid else 0,
    }
