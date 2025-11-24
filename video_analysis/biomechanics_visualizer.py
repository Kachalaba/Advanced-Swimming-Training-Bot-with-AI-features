"""
🦴 Biomechanics Visualizer - skeleton, angles, trajectories
"""
import cv2
import numpy as np
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import mediapipe as mp

logger = logging.getLogger(__name__)

# Colors (BGR)
COLORS = {
    "head": (255, 100, 255), "shoulder": (0, 255, 255), "elbow": (255, 150, 0),
    "wrist": (255, 200, 100), "hip": (0, 255, 0), "knee": (100, 255, 100),
    "ankle": (150, 255, 150), "spine": (0, 200, 255),
}

SKELETON = [
    ("nose", "left_eye", "head"), ("nose", "right_eye", "head"),
    ("left_shoulder", "right_shoulder", "shoulder"),
    ("left_shoulder", "left_hip", "spine"), ("right_shoulder", "right_hip", "spine"),
    ("left_hip", "right_hip", "hip"),
    ("left_shoulder", "left_elbow", "shoulder"), ("left_elbow", "left_wrist", "elbow"),
    ("right_shoulder", "right_elbow", "shoulder"), ("right_elbow", "right_wrist", "elbow"),
    ("left_hip", "left_knee", "hip"), ("left_knee", "left_ankle", "knee"),
    ("right_hip", "right_knee", "hip"), ("right_knee", "right_ankle", "knee"),
]

LANDMARKS = {
    "nose": 0, "left_eye": 2, "right_eye": 5,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14, "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26, "left_ankle": 27, "right_ankle": 28,
}


class BiomechanicsVisualizer:
    """Draw skeleton, angles, trajectories on frames."""
    
    def __init__(self, trajectory_length: int = 20, min_conf: float = 0.5):
        self.min_conf = min_conf
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, model_complexity=2,
            min_detection_confidence=min_conf, min_tracking_confidence=min_conf,
        )
        self.trajectories = {
            "left_wrist": deque(maxlen=trajectory_length),
            "right_wrist": deque(maxlen=trajectory_length),
            "left_ankle": deque(maxlen=trajectory_length),
            "right_ankle": deque(maxlen=trajectory_length),
        }
    
    def process_frame(self, frame: np.ndarray, idx: int = 0, bbox: List[int] = None):
        """Process frame, return (annotated, data)."""
        if frame is None:
            return frame, {"has_pose": False}
        
        annotated = frame.copy()
        h, w = frame.shape[:2]
        offset = [0, 0]
        
        # Crop if bbox
        if bbox:
            x1, y1, x2, y2 = [max(0, bbox[0]-20), max(0, bbox[1]-20),
                             min(w, bbox[2]+20), min(h, bbox[3]+20)]
            crop = frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else frame
            offset = [x1, y1]
        else:
            crop = frame
        
        # MediaPipe
        results = self.pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        
        if not results.pose_landmarks:
            return annotated, {"has_pose": False, "frame_index": idx}
        
        # Extract keypoints
        kps = {}
        ch, cw = crop.shape[:2]
        for name, i in LANDMARKS.items():
            lm = results.pose_landmarks.landmark[i]
            kps[name] = (int(lm.x * cw) + offset[0], int(lm.y * ch) + offset[1], lm.visibility)
        
        # Update trajectories
        for name in self.trajectories:
            if name in kps and kps[name][2] > self.min_conf:
                self.trajectories[name].append(kps[name][:2])
        
        # Calculate angles
        angles = self._calc_angles(kps)
        
        # Draw skeleton
        annotated = self._draw_skeleton(annotated, kps)
        
        # Draw angles
        annotated = self._draw_angles(annotated, angles, kps)
        
        # Draw trajectories
        annotated = self._draw_trajectories(annotated)
        
        # Draw panel
        annotated = self._draw_panel(annotated, angles, idx)
        
        return annotated, {
            "has_pose": True, "frame_index": idx,
            "keypoints": {k: {"x": v[0], "y": v[1]} for k, v in kps.items()},
            "angles": angles,
        }
    
    def _calc_angles(self, kps: Dict) -> Dict[str, float]:
        """Calculate joint angles."""
        angles = {}
        
        def angle_3p(p1, p2, p3):
            v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]])
            v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            return math.degrees(math.acos(np.clip(cos, -1, 1)))
        
        # Elbows
        if all(k in kps for k in ["left_shoulder", "left_elbow", "left_wrist"]):
            angles["L.elbow"] = angle_3p(kps["left_shoulder"][:2], kps["left_elbow"][:2], kps["left_wrist"][:2])
        if all(k in kps for k in ["right_shoulder", "right_elbow", "right_wrist"]):
            angles["R.elbow"] = angle_3p(kps["right_shoulder"][:2], kps["right_elbow"][:2], kps["right_wrist"][:2])
        
        # Knees
        if all(k in kps for k in ["left_hip", "left_knee", "left_ankle"]):
            angles["L.knee"] = angle_3p(kps["left_hip"][:2], kps["left_knee"][:2], kps["left_ankle"][:2])
        if all(k in kps for k in ["right_hip", "right_knee", "right_ankle"]):
            angles["R.knee"] = angle_3p(kps["right_hip"][:2], kps["right_knee"][:2], kps["right_ankle"][:2])
        
        # Streamline
        if all(k in kps for k in ["left_shoulder", "left_hip", "left_ankle"]):
            angles["streamline"] = angle_3p(kps["left_shoulder"][:2], kps["left_hip"][:2], kps["left_ankle"][:2])
        
        return angles
    
    def _draw_skeleton(self, frame: np.ndarray, kps: Dict) -> np.ndarray:
        # Bones
        for k1, k2, col in SKELETON:
            if k1 in kps and k2 in kps and kps[k1][2] > self.min_conf and kps[k2][2] > self.min_conf:
                cv2.line(frame, kps[k1][:2], kps[k2][:2], COLORS.get(col, (255,255,255)), 2, cv2.LINE_AA)
        # Joints
        for name, (x, y, c) in kps.items():
            if c > self.min_conf:
                col = COLORS.get("shoulder" if "shoulder" in name else "elbow" if "elbow" in name 
                                 else "wrist" if "wrist" in name else "hip" if "hip" in name
                                 else "knee" if "knee" in name else "ankle" if "ankle" in name else "head")
                cv2.circle(frame, (x, y), 6, col, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 6, (255,255,255), 1, cv2.LINE_AA)
        return frame
    
    def _draw_angles(self, frame: np.ndarray, angles: Dict, kps: Dict) -> np.ndarray:
        pos_map = {
            "L.elbow": "left_elbow", "R.elbow": "right_elbow",
            "L.knee": "left_knee", "R.knee": "right_knee",
        }
        for name, angle in angles.items():
            if name in pos_map and pos_map[name] in kps:
                x, y = kps[pos_map[name]][:2]
                color = (0,255,0) if 80 < angle < 160 else (0,200,255) if 60 < angle < 170 else (0,0,255)
                cv2.putText(frame, f"{angle:.0f}°", (x+8, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return frame
    
    def _draw_trajectories(self, frame: np.ndarray) -> np.ndarray:
        colors = {"left_wrist": (255,100,100), "right_wrist": (100,100,255),
                  "left_ankle": (100,255,100), "right_ankle": (255,255,100)}
        for name, pts in self.trajectories.items():
            if len(pts) < 2: continue
            pts_list = list(pts)
            for i in range(1, len(pts_list)):
                alpha = i / len(pts_list)
                cv2.line(frame, pts_list[i-1], pts_list[i], tuple(int(c*alpha) for c in colors[name]), max(1, int(alpha*3)), cv2.LINE_AA)
        return frame
    
    def _draw_panel(self, frame: np.ndarray, angles: Dict, idx: int) -> np.ndarray:
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (200, 100), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "BIOMECHANICS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1, cv2.LINE_AA)
        y = 50
        for name, angle in list(angles.items())[:3]:
            color = (0,255,0) if 80 < angle < 170 else (0,200,255)
            cv2.putText(frame, f"{name}: {angle:.0f}°", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            y += 16
        cv2.putText(frame, f"Frame: {idx}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150,150,150), 1, cv2.LINE_AA)
        return frame


def visualize_biomechanics(frames: List, detections: List = None, output_dir: str = "./biomech_viz") -> Dict:
    """Process frames with biomechanics visualization."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    viz = BiomechanicsVisualizer()
    results = []
    
    for i, f in enumerate(frames):
        path = f.get("path") if isinstance(f, dict) else f
        frame = cv2.imread(path)
        if frame is None: continue
        bbox = detections[i].get("bbox") if detections and i < len(detections) else None
        annotated, data = viz.process_frame(frame, i, bbox)
        cv2.imwrite(str(out / f"biomech_{i:04d}.jpg"), annotated)
        data["output_path"] = str(out / f"biomech_{i:04d}.jpg")
        results.append(data)
    
    valid = [r for r in results if r.get("has_pose")]
    return {
        "frame_analyses": results, "total": len(frames), "with_pose": len(valid),
        "rate": len(valid)/len(frames) if frames else 0, "output_dir": str(out),
    }
