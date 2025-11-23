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
        self.pixels_per_meter = None  # Calibrated from video
    
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
        
        # Find direction changes (wall touches)
        touches = []
        min_movement_threshold = 20  # Минимальное движение для детекции (пикселей)
        min_frames_between_touches = max(3, int(self.fps * 2))  # Минимум 2 секунды между касаниями
        
        for i in range(1, len(positions_x) - 1):
            # Skip if too close to previous touch
            if touches and (valid_indices[i] - touches[-1]) < min_frames_between_touches:
                continue
            
            # Calculate velocity
            v1 = positions_x[i] - positions_x[i - 1]
            v2 = positions_x[i + 1] - positions_x[i]
            
            # Direction reversal (wall touch) with stricter threshold
            if v1 * v2 < 0 and abs(v1) > min_movement_threshold:
                # Additional check: verify significant movement before reversal
                if i >= 2:
                    # Check movement over last few frames
                    movement = abs(positions_x[i] - positions_x[max(0, i-3)])
                    if movement > min_movement_threshold * 2:
                        touches.append(valid_indices[i])
                else:
                    touches.append(valid_indices[i])
        
        # Add start (first frame) and finish (last frame)
        if valid_indices:
            touches.insert(0, valid_indices[0])
            touches.append(valid_indices[-1])
        
        logger.info(f"Detected {len(touches)} wall touches at frames: {touches}")
        return touches
    
    def calibrate_pixels_to_meters(self, detections: List[Dict]) -> None:
        """Calibrate pixels to meters using pool length and video width.
        
        Assumes the video captures the full pool width.
        """
        if not detections:
            return
        
        # Find the range of X positions (pool width in pixels)
        x_positions = []
        for det in detections:
            if det.get("center"):
                x_positions.append(det["center"][0])
        
        if len(x_positions) < 10:
            # Not enough data for calibration
            logger.warning("Not enough data for pixel calibration")
            return
        
        # Calculate pool width in pixels (90th percentile range to avoid outliers)
        x_min = np.percentile(x_positions, 5)
        x_max = np.percentile(x_positions, 95)
        pool_width_pixels = x_max - x_min
        
        if pool_width_pixels > 50:  # Sanity check
            self.pixels_per_meter = pool_width_pixels / self.pool_length
            logger.info(
                f"Calibrated: {self.pixels_per_meter:.1f} pixels/meter "
                f"(pool: {pool_width_pixels:.0f} pixels = {self.pool_length}m)"
            )
        else:
            logger.warning(f"Invalid pool width: {pool_width_pixels} pixels")
    
    def calculate_real_distance(self, detections: List[Dict], start_frame: int, end_frame: int) -> float:
        """Calculate real distance traveled between frames using calibration.
        
        Args:
            detections: List of detections
            start_frame: Start frame index
            end_frame: End frame index
            
        Returns:
            Distance in meters
        """
        # Find detections in range
        positions = []
        for det in detections:
            frame = det.get("frame_index", -1)
            if start_frame <= frame <= end_frame and det.get("center"):
                positions.append(det["center"][0])
        
        if len(positions) < 2:
            # Fallback to pool_length
            return self.pool_length
        
        # Calculate total distance traveled (sum of movements)
        total_distance_pixels = 0
        for i in range(1, len(positions)):
            total_distance_pixels += abs(positions[i] - positions[i-1])
        
        # Convert to meters
        if self.pixels_per_meter and self.pixels_per_meter > 0:
            distance_meters = total_distance_pixels / self.pixels_per_meter
            # Sanity check: distance should be reasonable (0.5x to 2x pool_length)
            if 0.5 * self.pool_length <= distance_meters <= 2.0 * self.pool_length:
                return distance_meters
        
        # Fallback to pool_length if calibration failed or unreasonable
        return self.pool_length
    
    def calculate_splits(
        self,
        wall_touches: List[int],
        detections: List[Dict] = None,
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
            
            # Calculate time using real video timestamp if available
            if detections and start_frame < len(detections) and end_frame < len(detections):
                start_det = detections[start_frame]
                end_det = detections[end_frame]
                
                # Используем реальный timestamp из видео если доступен
                if start_det.get("timestamp") is not None and end_det.get("timestamp") is not None:
                    time_seconds = end_det["timestamp"] - start_det["timestamp"]
                    logger.debug(f"Split {i+1}: using video timestamps ({start_det['timestamp']:.3f}s - {end_det['timestamp']:.3f}s)")
                else:
                    # Fallback: расчёт по frame_index
                    frame_diff = end_frame - start_frame
                    time_seconds = frame_diff * self.frame_duration
                    logger.debug(f"Split {i+1}: using frame_duration fallback")
            else:
                # Fallback: расчёт по frame_index
                frame_diff = end_frame - start_frame
                time_seconds = frame_diff * self.frame_duration
            
            # Calculate distance using calibration if available
            if detections and self.pixels_per_meter:
                distance = self.calculate_real_distance(detections, start_frame, end_frame)
            else:
                distance = self.pool_length
            
            # Calculate speed
            speed = distance / time_seconds if time_seconds > 0 else 0
            
            # Filter anomalous speeds (sanity check)
            # Elite swimmers: ~2.0-2.5 m/s, recreational: 0.5-1.5 m/s
            # Allow range: 0.3 - 4.0 m/s
            if speed < 0.3 or speed > 4.0:
                logger.warning(f"Anomalous speed detected: {speed:.2f} m/s, using pool length estimate")
                distance = self.pool_length
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
    
    def analyze(
        self,
        detections: List[Dict],
    ) -> Dict:
        """Run full analysis on detections.
        
        Args:
            detections: List of detection results
            
        Returns:
            Analysis results dict
        """
        # Calibrate pixels to meters
        self.calibrate_pixels_to_meters(detections)
        
        # Detect wall touches
        wall_touches = self.detect_wall_touches(detections)
        
        # Calculate splits
        splits = self.calculate_splits(wall_touches, detections)
        
        # Estimate stroke rate
        stroke_rate = self.estimate_stroke_rate(detections)
        
        # Calculate summary stats
        total_distance = sum(s["distance_meters"] for s in splits)
        total_time = sum(s["time_seconds"] for s in splits)
        avg_speed = total_distance / total_time if total_time > 0 else 0
        
        # Проверяем используются ли реальные timestamps из видео
        uses_video_timestamps = False
        if detections and len(detections) > 0:
            uses_video_timestamps = detections[0].get("timestamp") is not None
        
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
                "uses_video_timestamps": uses_video_timestamps,  # Флаг: используем реальное время видео
                "timing_source": "video_timestamps" if uses_video_timestamps else "frame_extraction_rate",
            },
        }
        
        if uses_video_timestamps:
            logger.info("✅ Using real video timestamps for accurate speed/split analysis")
        else:
            logger.warning("⚠️ Using frame extraction rate for timing (less accurate)")
        
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
    analysis = analyzer.analyze(detections)
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
