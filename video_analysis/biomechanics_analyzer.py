"""Biomechanics and hydrodynamics analysis for swimming technique."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)


class BiomechanicsAnalyzer:
    """Analyze swimming biomechanics and hydrodynamics using pose estimation."""

    def __init__(self):
        """Initialize MediaPipe Pose for body keypoint detection."""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize pose detector
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,  # 0=lite, 1=full, 2=heavy (best accuracy)
            enable_segmentation=False,
            min_detection_confidence=0.5,
        )

        # Water resistance coefficients (empirical values)
        self.WATER_DENSITY = 1000  # kg/m³
        self.BASE_DRAG_COEFFICIENT = 0.4  # Streamlined swimmer

    def analyze_frame(
        self,
        frame_path: str,
        swimmer_bbox: Optional[List[int]] = None,
    ) -> Dict:
        """Analyze biomechanics for a single frame.

        Args:
            frame_path: Path to frame image
            swimmer_bbox: Optional bounding box [x1, y1, x2, y2] to crop to swimmer

        Returns:
            Dict with:
                - keypoints: 33 body landmarks (x, y, z, visibility)
                - angles: key body angles (degrees)
                - body_axes: longitudinal and transverse alignment
                - hydrodynamics: drag coefficient and frontal area
                - posture_score: 0-100 streamline quality
        """
        frame = cv2.imread(frame_path)
        if frame is None:
            logger.error(f"Cannot read frame: {frame_path}")
            return self._empty_result()

        # Crop to swimmer if bbox provided
        if swimmer_bbox:
            x1, y1, x2, y2 = swimmer_bbox
            frame = frame[y1:y2, x1:x2]

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            logger.warning(f"No pose detected in {frame_path}")
            return self._empty_result()

        # Extract keypoints
        keypoints = self._extract_keypoints(results.pose_landmarks, frame.shape)

        # Calculate body angles
        angles = self._calculate_angles(keypoints)

        # Calculate body axes alignment
        body_axes = self._calculate_body_axes(keypoints)

        # Hydrodynamics analysis
        hydrodynamics = self._analyze_hydrodynamics(keypoints, angles, body_axes)

        # Overall posture score
        posture_score = self._calculate_posture_score(angles, body_axes, hydrodynamics)

        return {
            "keypoints": keypoints,
            "angles": angles,
            "body_axes": body_axes,
            "hydrodynamics": hydrodynamics,
            "posture_score": posture_score,
            "has_pose": True,
        }

    def analyze_sequence(
        self,
        frame_paths: List[str],
        detections: List[Dict],
    ) -> Dict:
        """Analyze biomechanics across multiple frames.

        Args:
            frame_paths: List of frame paths
            detections: Swimmer detection results with bboxes

        Returns:
            Dict with:
                - frame_analyses: List of per-frame results
                - stroke_cycles: Detected stroke cycles
                - kick_cycles: Detected kick cycles
                - average_metrics: Average angles and scores
                - recommendations: List of technique improvements
        """
        frame_analyses = []

        for i, frame_path in enumerate(frame_paths):
            # Get swimmer bbox from detection
            bbox = None
            if i < len(detections) and detections[i].get("bbox"):
                bbox = detections[i]["bbox"]

            analysis = self.analyze_frame(frame_path, bbox)
            analysis["frame_index"] = i
            frame_analyses.append(analysis)

        # Detect movement cycles
        stroke_cycles = self._detect_stroke_cycles(frame_analyses)
        kick_cycles = self._detect_kick_cycles(frame_analyses)

        # Calculate averages
        average_metrics = self._calculate_average_metrics(frame_analyses)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            average_metrics, frame_analyses
        )

        return {
            "frame_analyses": frame_analyses,
            "stroke_cycles": stroke_cycles,
            "kick_cycles": kick_cycles,
            "average_metrics": average_metrics,
            "recommendations": recommendations,
        }

    def _extract_keypoints(self, landmarks, frame_shape) -> Dict:
        """Extract normalized keypoints from MediaPipe landmarks."""
        height, width = frame_shape[:2]

        keypoints = {}
        landmark_names = [
            "nose",
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_pinky",
            "right_pinky",
            "left_index",
            "right_index",
            "left_thumb",
            "right_thumb",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
        ]

        for idx, name in enumerate(landmark_names):
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                keypoints[name] = {
                    "x": int(lm.x * width),
                    "y": int(lm.y * height),
                    "z": lm.z,  # Relative depth
                    "visibility": lm.visibility,
                }

        return keypoints

    def _calculate_angles(self, keypoints: Dict) -> Dict:
        """Calculate key body angles for swimming technique."""
        angles = {}

        # Head angle (nose-shoulder alignment)
        if all(k in keypoints for k in ["nose", "left_shoulder", "right_shoulder"]):
            angles["head_elevation"] = self._calculate_angle_3points(
                keypoints["left_shoulder"],
                keypoints["nose"],
                keypoints["right_shoulder"],
            )

        # Left elbow angle
        if all(k in keypoints for k in ["left_shoulder", "left_elbow", "left_wrist"]):
            angles["left_elbow"] = self._calculate_angle_3points(
                keypoints["left_shoulder"],
                keypoints["left_elbow"],
                keypoints["left_wrist"],
            )

        # Right elbow angle
        if all(
            k in keypoints for k in ["right_shoulder", "right_elbow", "right_wrist"]
        ):
            angles["right_elbow"] = self._calculate_angle_3points(
                keypoints["right_shoulder"],
                keypoints["right_elbow"],
                keypoints["right_wrist"],
            )

        # Left knee angle
        if all(k in keypoints for k in ["left_hip", "left_knee", "left_ankle"]):
            angles["left_knee"] = self._calculate_angle_3points(
                keypoints["left_hip"],
                keypoints["left_knee"],
                keypoints["left_ankle"],
            )

        # Right knee angle
        if all(k in keypoints for k in ["right_hip", "right_knee", "right_ankle"]):
            angles["right_knee"] = self._calculate_angle_3points(
                keypoints["right_hip"],
                keypoints["right_knee"],
                keypoints["right_ankle"],
            )

        # Hip angle (torso-legs alignment)
        if all(k in keypoints for k in ["left_shoulder", "left_hip", "left_knee"]):
            angles["left_hip"] = self._calculate_angle_3points(
                keypoints["left_shoulder"],
                keypoints["left_hip"],
                keypoints["left_knee"],
            )

        # Body streamline angle (shoulder-hip-ankle line)
        if all(k in keypoints for k in ["left_shoulder", "left_hip", "left_ankle"]):
            angles["body_streamline"] = self._calculate_angle_3points(
                keypoints["left_shoulder"],
                keypoints["left_hip"],
                keypoints["left_ankle"],
            )

        return angles

    def _calculate_angle_3points(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """Calculate angle at p2 formed by p1-p2-p3."""
        # Vector p2->p1
        v1 = np.array([p1["x"] - p2["x"], p1["y"] - p2["y"]])
        # Vector p2->p3
        v2 = np.array([p3["x"] - p2["x"], p3["y"] - p2["y"]])

        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        return float(angle)

    def _calculate_body_axes(self, keypoints: Dict) -> Dict:
        """Calculate longitudinal and transverse body axes."""
        axes = {}

        # Longitudinal axis (head to hips)
        if all(k in keypoints for k in ["nose", "left_hip", "right_hip"]):
            head = np.array([keypoints["nose"]["x"], keypoints["nose"]["y"]])
            hip_center = np.array(
                [
                    (keypoints["left_hip"]["x"] + keypoints["right_hip"]["x"]) / 2,
                    (keypoints["left_hip"]["y"] + keypoints["right_hip"]["y"]) / 2,
                ]
            )

            # Angle from horizontal
            dx = hip_center[0] - head[0]
            dy = hip_center[1] - head[1]
            angle = np.degrees(np.arctan2(dy, dx))

            axes["longitudinal_angle"] = float(angle)
            axes["longitudinal_deviation"] = abs(angle)  # Deviation from horizontal

        # Transverse axis (shoulder width)
        if all(k in keypoints for k in ["left_shoulder", "right_shoulder"]):
            left_shoulder = np.array(
                [keypoints["left_shoulder"]["x"], keypoints["left_shoulder"]["y"]]
            )
            right_shoulder = np.array(
                [keypoints["right_shoulder"]["x"], keypoints["right_shoulder"]["y"]]
            )

            # Angle from horizontal
            dx = right_shoulder[0] - left_shoulder[0]
            dy = right_shoulder[1] - left_shoulder[1]
            angle = np.degrees(np.arctan2(dy, dx))

            axes["transverse_angle"] = float(angle)
            axes["shoulder_rotation"] = abs(angle)  # Rotation from horizontal

        return axes

    def _analyze_hydrodynamics(
        self, keypoints: Dict, angles: Dict, body_axes: Dict
    ) -> Dict:
        """Analyze hydrodynamic efficiency."""
        hydro = {}

        # Estimate frontal area (approximate)
        if all(
            k in keypoints
            for k in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        ):
            shoulder_width = abs(
                keypoints["right_shoulder"]["x"] - keypoints["left_shoulder"]["x"]
            )
            torso_height = abs(
                keypoints["left_shoulder"]["y"] - keypoints["left_hip"]["y"]
            )

            # Frontal area in pixels² (approximate as rectangle)
            frontal_area_px = shoulder_width * torso_height
            hydro["frontal_area_pixels"] = float(frontal_area_px)

        # Drag coefficient based on body position
        drag_coefficient = self.BASE_DRAG_COEFFICIENT

        # Head elevation increases drag
        if "head_elevation" in angles:
            head_angle = angles["head_elevation"]
            if head_angle > 140:  # Head lifted
                drag_coefficient += 0.1

        # Hip drop increases drag
        if "body_streamline" in angles:
            streamline = angles["body_streamline"]
            deviation = abs(180 - streamline)  # Deviation from straight
            if deviation > 20:
                drag_coefficient += 0.05 * (deviation / 20)

        # Body rotation increases drag
        if "shoulder_rotation" in body_axes:
            rotation = body_axes["shoulder_rotation"]
            if rotation > 15:
                drag_coefficient += 0.03

        hydro["drag_coefficient"] = float(min(drag_coefficient, 1.5))  # Cap at 1.5

        # Streamline quality (0-100)
        streamline_score = 100
        if "body_streamline" in angles:
            deviation = abs(180 - angles["body_streamline"])
            streamline_score -= deviation * 2
        if "longitudinal_deviation" in body_axes:
            streamline_score -= body_axes["longitudinal_deviation"]

        hydro["streamline_score"] = float(max(0, min(100, streamline_score)))

        return hydro

    def _calculate_posture_score(
        self, angles: Dict, body_axes: Dict, hydrodynamics: Dict
    ) -> float:
        """Calculate overall posture quality score (0-100)."""
        score = 100.0

        # Deduct for poor streamline
        if "streamline_score" in hydrodynamics:
            streamline_penalty = (100 - hydrodynamics["streamline_score"]) * 0.5
            score -= streamline_penalty

        # Deduct for high drag
        if "drag_coefficient" in hydrodynamics:
            drag_penalty = (
                hydrodynamics["drag_coefficient"] - self.BASE_DRAG_COEFFICIENT
            ) * 50
            score -= drag_penalty

        # Bonus for good body alignment
        if "body_streamline" in angles:
            deviation = abs(180 - angles["body_streamline"])
            if deviation < 10:
                score += 5

        return float(max(0, min(100, score)))

    def _detect_stroke_cycles(self, frame_analyses: List[Dict]) -> List[Dict]:
        """Detect arm stroke cycles from elbow angle patterns."""
        cycles = []

        # Extract left elbow angles
        left_angles = []
        for fa in frame_analyses:
            angle = fa.get("angles", {}).get("left_elbow")
            left_angles.append(angle if angle else 180)

        # Find peaks (extended arm) and troughs (bent arm)
        if len(left_angles) > 3:
            # Simple peak detection
            for i in range(1, len(left_angles) - 1):
                if (
                    left_angles[i] < left_angles[i - 1]
                    and left_angles[i] < left_angles[i + 1]
                ):
                    # Found a stroke (bent elbow)
                    if left_angles[i] < 120:  # Threshold for bent
                        cycles.append(
                            {
                                "type": "left_arm_stroke",
                                "frame": i,
                                "elbow_angle": left_angles[i],
                            }
                        )

        return cycles

    def _detect_kick_cycles(self, frame_analyses: List[Dict]) -> List[Dict]:
        """Detect leg kick cycles from knee angle patterns."""
        cycles = []

        # Extract knee angles
        knee_angles = []
        for fa in frame_analyses:
            left = fa.get("angles", {}).get("left_knee", 180)
            right = fa.get("angles", {}).get("right_knee", 180)
            avg = (left + right) / 2
            knee_angles.append(avg)

        # Find kicks (bent knees)
        if len(knee_angles) > 3:
            for i in range(1, len(knee_angles) - 1):
                if (
                    knee_angles[i] < knee_angles[i - 1]
                    and knee_angles[i] < knee_angles[i + 1]
                ):
                    if knee_angles[i] < 140:  # Threshold for kick
                        cycles.append(
                            {
                                "type": "kick",
                                "frame": i,
                                "knee_angle": knee_angles[i],
                            }
                        )

        return cycles

    def _calculate_average_metrics(self, frame_analyses: List[Dict]) -> Dict:
        """Calculate average biomechanical metrics."""
        valid_analyses = [fa for fa in frame_analyses if fa.get("has_pose")]

        if not valid_analyses:
            return {}

        # Average angles
        avg_angles = {}
        angle_keys = [
            "head_elevation",
            "left_elbow",
            "right_elbow",
            "left_knee",
            "right_knee",
            "body_streamline",
        ]

        for key in angle_keys:
            values = [
                fa["angles"].get(key) for fa in valid_analyses if fa["angles"].get(key)
            ]
            if values:
                avg_angles[key] = sum(values) / len(values)

        # Average hydrodynamics
        drag_values = [
            fa["hydrodynamics"]["drag_coefficient"]
            for fa in valid_analyses
            if "drag_coefficient" in fa["hydrodynamics"]
        ]
        avg_drag = sum(drag_values) / len(drag_values) if drag_values else 0.4

        streamline_values = [
            fa["hydrodynamics"]["streamline_score"]
            for fa in valid_analyses
            if "streamline_score" in fa["hydrodynamics"]
        ]
        avg_streamline = (
            sum(streamline_values) / len(streamline_values) if streamline_values else 50
        )

        # Average posture score
        posture_values = [fa["posture_score"] for fa in valid_analyses]
        avg_posture = (
            sum(posture_values) / len(posture_values) if posture_values else 50
        )

        return {
            "average_angles": avg_angles,
            "average_drag_coefficient": avg_drag,
            "average_streamline_score": avg_streamline,
            "average_posture_score": avg_posture,
            "frames_with_pose": len(valid_analyses),
            "total_frames": len(frame_analyses),
        }

    def _generate_recommendations(
        self, average_metrics: Dict, frame_analyses: List[Dict]
    ) -> List[str]:
        """Generate technique improvement recommendations."""
        recommendations = []

        if not average_metrics:
            return ["Not enough data to analyze posture."]

        avg_angles = average_metrics.get("average_angles", {})
        avg_drag = average_metrics.get("average_drag_coefficient", 0.4)
        avg_streamline = average_metrics.get("average_streamline_score", 50)

        # Head position
        if "head_elevation" in avg_angles:
            if avg_angles["head_elevation"] > 150:
                recommendations.append(
                    "⚠️ Head is raised too high. Lower your face into the water to reduce drag."
                )

        # Streamline
        if avg_streamline < 70:
            recommendations.append(
                "⚠️ Body position is suboptimal. Aim to keep the body straight and horizontal."
            )

        # Drag coefficient
        if avg_drag > 0.6:
            recommendations.append(
                f"⚠️ High drag coefficient ({avg_drag:.2f}). "
                "Improve streamlining: lower your head and align your hips."
            )

        # Elbow angles
        if "left_elbow" in avg_angles and "right_elbow" in avg_angles:
            avg_elbow = (avg_angles["left_elbow"] + avg_angles["right_elbow"]) / 2
            if avg_elbow > 160:
                recommendations.append(
                    "💡 Elbows are too straight. Bend them during the pull for more power."
                )

        # Body streamline
        if "body_streamline" in avg_angles:
            deviation = abs(180 - avg_angles["body_streamline"])
            if deviation > 20:
                recommendations.append("⚠️ Body is misaligned. Work on core stability.")

        if not recommendations:
            recommendations.append("✅ Technique looks good! Keep it up.")

        return recommendations

    def _empty_result(self) -> Dict:
        """Return empty result when no pose detected."""
        return {
            "keypoints": {},
            "angles": {},
            "body_axes": {},
            "hydrodynamics": {},
            "posture_score": 0,
            "has_pose": False,
        }

    def draw_pose_on_frame(
        self,
        frame_path: str,
        analysis: Dict,
        output_path: str = None,
    ) -> str:
        """Draw pose skeleton and metrics on frame.

        Args:
            frame_path: Input frame path
            analysis: Biomechanics analysis result
            output_path: Output image path

        Returns:
            Path to output image
        """
        frame = cv2.imread(frame_path)
        if frame is None or not analysis.get("has_pose"):
            return frame_path

        keypoints = analysis["keypoints"]
        angles = analysis["angles"]
        hydro = analysis["hydrodynamics"]

        # Draw keypoints
        for name, kp in keypoints.items():
            if kp["visibility"] > 0.5:
                cv2.circle(frame, (kp["x"], kp["y"]), 3, (0, 255, 0), -1)

        # Draw skeleton connections
        connections = [
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            ("left_hip", "left_knee"),
            ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"),
            ("right_knee", "right_ankle"),
        ]

        for p1_name, p2_name in connections:
            if p1_name in keypoints and p2_name in keypoints:
                p1 = keypoints[p1_name]
                p2 = keypoints[p2_name]
                if p1["visibility"] > 0.5 and p2["visibility"] > 0.5:
                    cv2.line(
                        frame, (p1["x"], p1["y"]), (p2["x"], p2["y"]), (255, 255, 0), 2
                    )

        # Draw metrics overlay
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Posture score
        score = analysis["posture_score"]
        color = (
            (0, 255, 0) if score > 70 else (0, 165, 255) if score > 50 else (0, 0, 255)
        )
        cv2.putText(
            frame, f"Posture: {score:.1f}/100", (10, y_offset), font, 0.7, color, 2
        )
        y_offset += 30

        # Drag coefficient
        if "drag_coefficient" in hydro:
            drag = hydro["drag_coefficient"]
            cv2.putText(
                frame,
                f"Drag Cd: {drag:.2f}",
                (10, y_offset),
                font,
                0.6,
                (255, 255, 255),
                2,
            )
            y_offset += 25

        # Streamline score
        if "streamline_score" in hydro:
            streamline = hydro["streamline_score"]
            cv2.putText(
                frame,
                f"Streamline: {streamline:.0f}%",
                (10, y_offset),
                font,
                0.6,
                (255, 255, 255),
                2,
            )

        # Save output
        if output_path is None:
            path = Path(frame_path)
            output_path = str(path.parent / f"pose_{path.name}")

        cv2.imwrite(output_path, frame)
        logger.info(f"Saved pose visualization: {output_path}")

        return output_path

    def save_analysis_json(self, analysis: Dict, output_path: str) -> str:
        """Save biomechanics analysis to JSON file."""
        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Saved biomechanics analysis to {output_path}")
        return output_path


def analyze_biomechanics(
    frame_paths: List[str],
    detections: List[Dict],
    output_dir: str = "./biomechanics",
) -> Dict:
    """Convenience function to analyze biomechanics.

    Args:
        frame_paths: List of frame paths
        detections: Swimmer detection results
        output_dir: Output directory

    Returns:
        Biomechanics analysis results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    analyzer = BiomechanicsAnalyzer()
    analysis = analyzer.analyze_sequence(frame_paths, detections)

    # Save JSON
    analyzer.save_analysis_json(analysis, str(output_path / "biomechanics.json"))

    return analysis


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    frame_paths = [f"./frames/frame_{i:04d}.jpg" for i in range(10)]
    detections = [{"bbox": [100, 100, 400, 400]}] * 10

    result = analyze_biomechanics(frame_paths, detections)
    print(f"Analyzed {len(result['frame_analyses'])} frames")
    print(
        f"Average posture score: {result['average_metrics']['average_posture_score']:.1f}"
    )
