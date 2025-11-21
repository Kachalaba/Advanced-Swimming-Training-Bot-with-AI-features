"""Extract frames from swimming video using OpenCV."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import cv2

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extract frames from video at specified intervals."""

    def __init__(self, output_dir: str = "./frames"):
        """Initialize frame extractor.
        
        Args:
            output_dir: Directory to save extracted frames
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_frames(
        self,
        video_path: str,
        fps: int = 1,
        max_duration: int = 60,
    ) -> List[str]:
        """Extract frames from video.
        
        Args:
            video_path: Path to input video (mp4)
            fps: Frames per second to extract (default: 1)
            max_duration: Maximum video duration in seconds
            
        Returns:
            List of paths to saved frames
            
        Raises:
            ValueError: If video cannot be opened or is too long
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps
        
        if duration > max_duration:
            cap.release()
            raise ValueError(f"Video too long: {duration:.1f}s (max: {max_duration}s)")
        
        logger.info(f"Video: {duration:.1f}s, {video_fps:.1f} fps, {total_frames} frames")
        
        # Calculate frame interval
        frame_interval = int(video_fps / fps)
        
        saved_frames = []
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame at intervals
            if frame_count % frame_interval == 0:
                frame_path = self.output_dir / f"frame_{saved_count:04d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                saved_frames.append(str(frame_path))
                saved_count += 1
                logger.debug(f"Saved frame {saved_count} at {frame_count / video_fps:.1f}s")
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {saved_count} frames from {video_path}")
        
        return saved_frames


def extract_frames_from_video(
    video_path: str,
    output_dir: str = "./frames",
    fps: int = 1,
) -> dict:
    """Extract frames from video (convenience function).
    
    Args:
        video_path: Path to video file
        output_dir: Output directory
        fps: Frames per second to extract
        
    Returns:
        Dict with:
            - frames: List of frame paths
            - count: Number of frames
            - output_dir: Output directory
    """
    extractor = FrameExtractor(output_dir)
    frames = extractor.extract_frames(video_path, fps=fps)
    
    return {
        "frames": frames,
        "count": len(frames),
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    result = extract_frames_from_video(
        "test_video.mp4",
        output_dir="./test_frames",
        fps=1,
    )
    
    print(f"Extracted {result['count']} frames to {result['output_dir']}")
