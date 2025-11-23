"""Trajectory-based analysis for swimming when pose detection fails.

This module analyzes swimmer movement using only bounding box trajectories
from YOLO detection, without requiring pose estimation. Useful for underwater
videos where MediaPipe pose detection has low success rate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import math

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class TrajectoryAnalyzer:
    """Analyze swimming technique using only bounding box trajectories."""
    
    def __init__(self):
        """Initialize trajectory analyzer."""
        self.WATER_DENSITY = 1000  # kg/m³
        
    def analyze_trajectory(
        self,
        detections: List[Dict],
        fps: float = 2.0,
        pool_length: float = 25.0,
    ) -> Dict:
        """Analyze swimming technique from bbox trajectories.
        
        Args:
            detections: List of detection dicts with bbox coordinates
            fps: Frames per second
            pool_length: Pool length in meters
            
        Returns:
            Dict with trajectory metrics, movement quality, and recommendations
        """
        if not detections:
            logger.warning("No detections provided")
            return self._empty_result()
        
        # Extract trajectory data
        trajectory = self._extract_trajectory(detections)
        
        if len(trajectory) < 2:
            logger.warning("Not enough trajectory points")
            return self._empty_result()
        
        # Compute metrics
        velocity_profile = self._compute_velocity(trajectory, fps)
        acceleration_profile = self._compute_acceleration(velocity_profile, fps)
        body_rotation = self._analyze_rotation(trajectory)
        streamline_quality = self._analyze_streamline(trajectory)
        rhythm_metrics = self._analyze_rhythm(trajectory, fps)
        wall_touches = self._detect_wall_touches(velocity_profile, trajectory)
        
        # Overall scores
        movement_score = self._compute_movement_score(
            velocity_profile,
            streamline_quality,
            rhythm_metrics
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            velocity_profile,
            body_rotation,
            streamline_quality,
            rhythm_metrics
        )
        
        return {
            "method": "trajectory_analysis",
            "trajectory": trajectory,
            "velocity_profile": velocity_profile,
            "acceleration_profile": acceleration_profile,
            "body_rotation": body_rotation,
            "streamline_quality": streamline_quality,
            "rhythm_metrics": rhythm_metrics,
            "wall_touches": wall_touches,
            "movement_score": movement_score,
            "recommendations": recommendations,
            "summary": self._create_summary(
                velocity_profile,
                streamline_quality,
                movement_score,
                len(detections)
            )
        }
    
    def _extract_trajectory(self, detections: List[Dict]) -> List[Dict]:
        """Extract center points and bbox dimensions from detections."""
        trajectory = []
        
        for det in detections:
            if not det.get("bbox"):
                continue
            
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox
            
            # Center point
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Dimensions
            width = x2 - x1
            height = y2 - y1
            area = width * height
            aspect_ratio = width / height if height > 0 else 0
            
            trajectory.append({
                "frame": det.get("frame_index", len(trajectory)),
                "center": (center_x, center_y),
                "bbox": bbox,
                "width": width,
                "height": height,
                "area": area,
                "aspect_ratio": aspect_ratio,
            })
        
        return trajectory
    
    def _compute_velocity(
        self,
        trajectory: List[Dict],
        fps: float
    ) -> Dict:
        """Compute velocity profile from trajectory."""
        if len(trajectory) < 2:
            return {
                "velocities": [],
                "avg_velocity": 0.0,
                "max_velocity": 0.0,
                "min_velocity": 0.0,
            }
        
        velocities = []
        time_delta = 1.0 / fps
        
        for i in range(1, len(trajectory)):
            prev = trajectory[i - 1]
            curr = trajectory[i]
            
            # Distance in pixels
            dx = curr["center"][0] - prev["center"][0]
            dy = curr["center"][1] - prev["center"][1]
            distance = math.sqrt(dx**2 + dy**2)
            
            # Velocity in pixels/second
            velocity = distance / time_delta
            
            velocities.append({
                "frame": curr["frame"],
                "velocity": velocity,
                "dx": dx,
                "dy": dy,
                "distance": distance,
            })
        
        velocity_values = [v["velocity"] for v in velocities]
        
        return {
            "velocities": velocities,
            "avg_velocity": np.mean(velocity_values) if velocity_values else 0.0,
            "max_velocity": np.max(velocity_values) if velocity_values else 0.0,
            "min_velocity": np.min(velocity_values) if velocity_values else 0.0,
            "velocity_std": np.std(velocity_values) if velocity_values else 0.0,
        }
    
    def _compute_acceleration(
        self,
        velocity_profile: Dict,
        fps: float
    ) -> Dict:
        """Compute acceleration from velocity profile."""
        velocities = velocity_profile.get("velocities", [])
        
        if len(velocities) < 2:
            return {
                "accelerations": [],
                "avg_acceleration": 0.0,
            }
        
        accelerations = []
        time_delta = 1.0 / fps
        
        for i in range(1, len(velocities)):
            prev_vel = velocities[i - 1]["velocity"]
            curr_vel = velocities[i]["velocity"]
            
            accel = (curr_vel - prev_vel) / time_delta
            
            accelerations.append({
                "frame": velocities[i]["frame"],
                "acceleration": accel,
            })
        
        accel_values = [a["acceleration"] for a in accelerations]
        
        return {
            "accelerations": accelerations,
            "avg_acceleration": np.mean(accel_values) if accel_values else 0.0,
            "max_acceleration": np.max(accel_values) if accel_values else 0.0,
        }
    
    def _analyze_rotation(self, trajectory: List[Dict]) -> Dict:
        """Analyze body rotation from bbox aspect ratio changes."""
        if len(trajectory) < 3:
            return {
                "rotation_changes": [],
                "avg_rotation": 0.0,
                "rotation_stability": 100.0,
            }
        
        aspect_ratios = [t["aspect_ratio"] for t in trajectory]
        
        # Detect rotation changes
        rotation_changes = []
        for i in range(1, len(aspect_ratios)):
            change = abs(aspect_ratios[i] - aspect_ratios[i - 1])
            rotation_changes.append({
                "frame": trajectory[i]["frame"],
                "aspect_ratio": aspect_ratios[i],
                "change": change,
            })
        
        changes = [r["change"] for r in rotation_changes]
        
        # Stability: lower variance = more stable
        stability = 100.0 - min(np.std(aspect_ratios) * 100, 100.0)
        
        return {
            "rotation_changes": rotation_changes,
            "avg_aspect_ratio": np.mean(aspect_ratios),
            "aspect_ratio_std": np.std(aspect_ratios),
            "rotation_stability": stability,
        }
    
    def _analyze_streamline(self, trajectory: List[Dict]) -> Dict:
        """Analyze streamline quality from bbox dimensions."""
        if not trajectory:
            return {
                "avg_streamline_score": 0.0,
                "consistency": 0.0,
            }
        
        # Streamline indicator: lower aspect ratio = more streamlined
        # Ideal: aspect_ratio close to 1.0 (square-ish, body straight)
        aspect_ratios = [t["aspect_ratio"] for t in trajectory]
        
        # Score: penalize deviation from ideal streamline
        scores = []
        for ar in aspect_ratios:
            # Ideal range: 0.3-0.5 (horizontal swimmer)
            if 0.3 <= ar <= 0.5:
                score = 100.0
            elif ar < 0.3:
                score = max(0, 100.0 - (0.3 - ar) * 200)
            else:
                score = max(0, 100.0 - (ar - 0.5) * 100)
            scores.append(score)
        
        # Consistency: lower variance = better
        consistency = 100.0 - min(np.std(scores), 100.0)
        
        return {
            "streamline_scores": scores,
            "avg_streamline_score": np.mean(scores) if scores else 0.0,
            "consistency": consistency,
            "ideal_aspect_ratio": 0.4,
            "current_avg_aspect_ratio": np.mean(aspect_ratios),
        }
    
    def _analyze_rhythm(self, trajectory: List[Dict], fps: float) -> Dict:
        """Analyze movement rhythm and periodicity."""
        if len(trajectory) < 10:
            return {
                "rhythm_score": 0.0,
                "stroke_frequency": 0.0,
            }
        
        # Analyze area changes (indicator of stroke cycles)
        areas = [t["area"] for t in trajectory]
        
        # Detect peaks in area (arms extended)
        peaks = []
        for i in range(1, len(areas) - 1):
            if areas[i] > areas[i - 1] and areas[i] > areas[i + 1]:
                peaks.append(i)
        
        # Compute frequency
        if len(peaks) > 1:
            avg_interval = np.mean(np.diff(peaks)) / fps
            frequency = 1.0 / avg_interval if avg_interval > 0 else 0.0
        else:
            frequency = 0.0
        
        # Rhythm score: consistency of intervals
        if len(peaks) > 2:
            intervals = np.diff(peaks)
            rhythm_score = 100.0 - min(np.std(intervals) * 10, 100.0)
        else:
            rhythm_score = 0.0
        
        return {
            "rhythm_score": rhythm_score,
            "stroke_frequency": frequency,
            "detected_peaks": len(peaks),
            "peak_frames": [trajectory[p]["frame"] for p in peaks],
        }
    
    def _detect_wall_touches(
        self,
        velocity_profile: Dict,
        trajectory: List[Dict]
    ) -> Dict:
        """Detect wall touches from velocity drops."""
        velocities = velocity_profile.get("velocities", [])
        
        if len(velocities) < 5:
            return {
                "wall_touches": [],
                "count": 0,
            }
        
        wall_touches = []
        velocity_values = [v["velocity"] for v in velocities]
        threshold = np.mean(velocity_values) * 0.3  # 30% of avg velocity
        
        for i in range(2, len(velocities) - 2):
            curr_vel = velocities[i]["velocity"]
            
            # Wall touch: sudden drop then increase
            if curr_vel < threshold:
                # Check if velocity increases after
                future_vel = velocities[i + 1]["velocity"]
                if future_vel > curr_vel * 1.5:
                    wall_touches.append({
                        "frame": velocities[i]["frame"],
                        "velocity_before": velocities[i - 1]["velocity"],
                        "velocity_at_touch": curr_vel,
                        "velocity_after": future_vel,
                    })
        
        return {
            "wall_touches": wall_touches,
            "count": len(wall_touches),
        }
    
    def _compute_movement_score(
        self,
        velocity_profile: Dict,
        streamline_quality: Dict,
        rhythm_metrics: Dict
    ) -> float:
        """Compute overall movement quality score (0-100)."""
        # Components
        velocity_consistency = 100.0 - min(
            velocity_profile.get("velocity_std", 0) * 2,
            100.0
        )
        streamline_score = streamline_quality.get("avg_streamline_score", 0)
        rhythm_score = rhythm_metrics.get("rhythm_score", 0)
        
        # Weighted average
        movement_score = (
            velocity_consistency * 0.3 +
            streamline_score * 0.4 +
            rhythm_score * 0.3
        )
        
        return movement_score
    
    def _generate_recommendations(
        self,
        velocity_profile: Dict,
        body_rotation: Dict,
        streamline_quality: Dict,
        rhythm_metrics: Dict
    ) -> List[str]:
        """Generate technique recommendations based on trajectory analysis."""
        recommendations = []
        
        # Velocity consistency
        vel_std = velocity_profile.get("velocity_std", 0)
        if vel_std > 50:
            recommendations.append(
                "⚠️ Неровная скорость. Работайте над плавностью движений и постоянным темпом."
            )
        elif vel_std < 20:
            recommendations.append(
                "✅ Отличная стабильность скорости!"
            )
        
        # Streamline quality
        streamline_score = streamline_quality.get("avg_streamline_score", 0)
        if streamline_score < 60:
            recommendations.append(
                "⚠️ Положение тела не оптимально. Вытягивайтесь горизонтально для снижения сопротивления."
            )
        elif streamline_score > 80:
            recommendations.append(
                "✅ Отличная обтекаемость тела!"
            )
        
        # Body rotation stability
        rotation_stability = body_rotation.get("rotation_stability", 0)
        if rotation_stability < 70:
            recommendations.append(
                "💡 Тело вращается неравномерно. Работайте над стабилизацией core и ровным положением."
            )
        elif rotation_stability > 85:
            recommendations.append(
                "✅ Стабильное положение тела в воде!"
            )
        
        # Rhythm
        rhythm_score = rhythm_metrics.get("rhythm_score", 0)
        if rhythm_score < 50:
            recommendations.append(
                "💡 Ритм движений неравномерен. Работайте над постоянной частотой гребков."
            )
        elif rhythm_score > 75:
            recommendations.append(
                "✅ Ровный ритм движений!"
            )
        
        if not recommendations:
            recommendations.append(
                "✅ Хорошая техника! Продолжайте в том же духе."
            )
        
        return recommendations
    
    def _create_summary(
        self,
        velocity_profile: Dict,
        streamline_quality: Dict,
        movement_score: float,
        total_frames: int
    ) -> Dict:
        """Create summary statistics."""
        return {
            "total_frames": total_frames,
            "avg_velocity_pixels_per_sec": velocity_profile.get("avg_velocity", 0),
            "velocity_consistency": 100.0 - min(
                velocity_profile.get("velocity_std", 0) * 2,
                100.0
            ),
            "streamline_score": streamline_quality.get("avg_streamline_score", 0),
            "movement_quality_score": movement_score,
        }
    
    def _empty_result(self) -> Dict:
        """Return empty result structure."""
        return {
            "method": "trajectory_analysis",
            "trajectory": [],
            "velocity_profile": {},
            "body_rotation": {},
            "streamline_quality": {},
            "rhythm_metrics": {},
            "wall_touches": {"wall_touches": [], "count": 0},
            "movement_score": 0.0,
            "recommendations": ["⚠️ Недостаточно данных для анализа траектории."],
            "summary": {
                "total_frames": 0,
                "movement_quality_score": 0.0,
            }
        }
    
    def save_results(self, results: Dict, output_path: str) -> None:
        """Save trajectory analysis results to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Trajectory analysis saved to {output_path}")


def analyze_trajectory(
    detections: List[Dict],
    fps: float = 2.0,
    pool_length: float = 25.0,
    output_dir: Optional[str] = None,
) -> Dict:
    """Convenience function to analyze trajectory.
    
    Args:
        detections: List of detection dicts
        fps: Frames per second
        pool_length: Pool length in meters
        output_dir: Optional directory to save results
        
    Returns:
        Analysis results dict
    """
    analyzer = TrajectoryAnalyzer()
    results = analyzer.analyze_trajectory(detections, fps, pool_length)
    
    if output_dir:
        output_path = Path(output_dir) / "trajectory_analysis.json"
        analyzer.save_results(results, str(output_path))
    
    return results
