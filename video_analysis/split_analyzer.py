"""Analyze split times and stroke frequency from detections."""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SplitAnalyzer:
    """Analyze swimming splits and stroke frequency."""

    def __init__(
        self,
        pool_length: float = 25.0,
        fps: float = 1.0,
    ):
        """Initialize split analyzer.
        
        Args:
            pool_length: Pool length in meters
            fps: Frames per second (for timing)
        """
        self.pool_length = pool_length
        self.fps = fps
        self.frame_duration = 1.0 / fps
    
    def detect_wall_touches(
        self,
        detections: List[Dict],
        edge_threshold: float = 0.1,
    ) -> List[int]:
        """Detect wall touches (turns, start, finish).
        
        Args:
            detections: List of detection results
            edge_threshold: Distance from edge to consider touch (0-1, relative)
            
        Returns:
            List of frame indices where wall touch detected
        """
        touches = []
        
        for det in detections:
            if det["center"] is None:
                continue
            
            cx, _ = det["center"]
            
            # Assuming video frame width represents pool width
            # Touches occur at x < threshold or x > (1-threshold)
            # We need to estimate this from first/last frames
            
            # Simple heuristic: detect sudden position changes
            touches.append(det["frame_index"])
        
        # Find frames where swimmer is near edges
        if not detections:
            return []
        
        # Get frame dimensions from first detection
        first_frame_with_bbox = next(
            (d for d in detections if d["bbox"] is not None), None
        )
        if not first_frame_with_bbox:
            return []
        
        # Detect position reversals (turns)
        positions_x = []
        valid_indices = []
        
        for det in detections:
            if det["center"] is not None:
                positions_x.append(det["center"][0])
                valid_indices.append(det["frame_index"])
        
        if len(positions_x) < 3:
            return []
        
        # Find direction changes
        touches = []
        for i in range(1, len(positions_x) - 1):
            # Calculate velocity
            v1 = positions_x[i] - positions_x[i - 1]
            v2 = positions_x[i + 1] - positions_x[i]
            
            # Direction reversal (wall touch)
            if v1 * v2 < 0 and abs(v1) > 5:  # Threshold for movement
                touches.append(valid_indices[i])
        
        # Add start (first frame) and finish (last frame)
        if valid_indices:
            touches.insert(0, valid_indices[0])
            touches.append(valid_indices[-1])
        
        logger.info(f"Detected {len(touches)} wall touches at frames: {touches}")
        return touches
    
    def calculate_splits(
        self,
        wall_touches: List[int],
    ) -> List[Dict]:
        """Calculate split times between wall touches.
        
        Args:
            wall_touches: Frame indices of wall touches
            
        Returns:
            List of splits with times and distances
        """
        if len(wall_touches) < 2:
            return []
        
        splits = []
        for i in range(len(wall_touches) - 1):
            start_frame = wall_touches[i]
            end_frame = wall_touches[i + 1]
            
            # Calculate time
            frames_elapsed = end_frame - start_frame
            time_seconds = frames_elapsed * self.frame_duration
            
            # Calculate distance (alternate between pool_length)
            distance = self.pool_length
            
            # Calculate speed
            speed = distance / time_seconds if time_seconds > 0 else 0
            
            split = {
                "split_number": i + 1,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "time_seconds": round(time_seconds, 2),
                "distance_meters": distance,
                "speed_mps": round(speed, 2),
                "pace_per_100m": round(100 / speed, 2) if speed > 0 else 0,
            }
            splits.append(split)
            logger.info(
                f"Split {i+1}: {time_seconds:.2f}s, {speed:.2f} m/s"
            )
        
        return splits
    
    def estimate_stroke_rate(
        self,
        detections: List[Dict],
        window_size: int = 5,
    ) -> Optional[float]:
        """Estimate stroke rate from position oscillations.
        
        Args:
            detections: Detection results
            window_size: Frames to analyze
            
        Returns:
            Estimated strokes per minute or None
        """
        # Get Y positions (vertical movement indicates strokes)
        positions_y = []
        for det in detections:
            if det["center"] is not None:
                positions_y.append(det["center"][1])
        
        if len(positions_y) < window_size:
            return None
        
        # Count peaks (arm cycles)
        positions = np.array(positions_y)
        mean_y = np.mean(positions)
        
        # Simple peak counting
        peaks = 0
        for i in range(1, len(positions) - 1):
            if positions[i] > positions[i-1] and positions[i] > positions[i+1]:
                if positions[i] > mean_y:
                    peaks += 1
        
        # Convert to strokes per minute
        duration_minutes = (len(detections) * self.frame_duration) / 60
        if duration_minutes > 0:
            strokes_per_minute = peaks / duration_minutes
            logger.info(f"Estimated stroke rate: {strokes_per_minute:.1f} SPM")
            return round(strokes_per_minute, 1)
        
        return None
    
    def analyze_video(
        self,
        detections: List[Dict],
    ) -> Dict:
        """Complete analysis of swimming video.
        
        Args:
            detections: Detection results from SwimmerDetector
            
        Returns:
            Complete analysis results
        """
        # Detect wall touches
        wall_touches = self.detect_wall_touches(detections)
        
        # Calculate splits
        splits = self.calculate_splits(wall_touches)
        
        # Estimate stroke rate
        stroke_rate = self.estimate_stroke_rate(detections)
        
        # Calculate summary stats
        total_distance = sum(s["distance_meters"] for s in splits)
        total_time = sum(s["time_seconds"] for s in splits)
        avg_speed = total_distance / total_time if total_time > 0 else 0
        
        analysis = {
            "wall_touches": {
                "count": len(wall_touches),
                "frames": wall_touches,
            },
            "splits": splits,
            "stroke_rate_spm": stroke_rate,
            "summary": {
                "total_distance_m": round(total_distance, 1),
                "total_time_s": round(total_time, 2),
                "average_speed_mps": round(avg_speed, 2),
                "average_pace_per_100m": round(100 / avg_speed, 2) if avg_speed > 0 else 0,
            },
        }
        
        logger.info(f"Analysis complete: {total_distance}m in {total_time:.2f}s")
        return analysis
    
    def save_analysis(
        self,
        analysis: Dict,
        output_path: str = "analysis.json",
    ) -> str:
        """Save analysis results to JSON.
        
        Args:
            analysis: Analysis results
            output_path: Output file path
            
        Returns:
            Path to saved file
        """
        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Saved analysis to {output_path}")
        return output_path


def analyze_swimming_video(
    detections: List[Dict],
    pool_length: float = 25.0,
    fps: float = 1.0,
    output_path: str = "analysis.json",
) -> Dict:
    """Analyze swimming video (convenience function).
    
    Args:
        detections: Detection results
        pool_length: Pool length in meters
        fps: Video frame rate
        output_path: Output JSON path
        
    Returns:
        Analysis results
    """
    analyzer = SplitAnalyzer(pool_length=pool_length, fps=fps)
    analysis = analyzer.analyze_video(detections)
    analyzer.save_analysis(analysis, output_path)
    
    return analysis


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Load detections
    with open("detections.json") as f:
        detections = json.load(f)
    
    # Analyze
    analysis = analyze_swimming_video(
        detections,
        pool_length=25.0,
        fps=1.0,
    )
    
    print(f"Total distance: {analysis['summary']['total_distance_m']}m")
    print(f"Total time: {analysis['summary']['total_time_s']}s")
    print(f"Stroke rate: {analysis['stroke_rate_spm']} SPM")
