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
        fps: float = 1.0,
        max_duration: int = 60,
    ) -> List[str]:
        """Extract frames from video.
        
        Args:
            video_path: Path to input video (mp4)
            fps: Frames per second to extract (float, default: 1.0)
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
        logger.info(f"Target extraction rate: {fps:.2f} fps")
        
        # Используем накопление времени для точного извлечения кадров
        # Вместо округления frame_interval
        time_per_extracted_frame = 1.0 / fps  # Время между извлекаемыми кадрами (сек)
        time_per_video_frame = 1.0 / video_fps  # Время одного кадра видео (сек)
        
        saved_frames = []
        frame_count = 0
        saved_count = 0
        next_save_time = 0.0  # Время когда нужно сохранить следующий кадр
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_count * time_per_video_frame
            
            # Сохраняем кадр если текущее время >= времени следующего сохранения
            if current_time >= next_save_time:
                frame_path = self.output_dir / f"frame_{saved_count:04d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                
                # Сохраняем как словарь с timestamp из реального видео
                saved_frames.append({
                    "path": str(frame_path),
                    "timestamp": current_time,  # Реальное время в видео (сек)
                    "video_frame": frame_count,  # Номер кадра в исходном видео
                    "extracted_index": saved_count,  # Индекс среди извлечённых
                })
                
                logger.debug(f"Saved frame {saved_count} at {current_time:.3f}s (video frame {frame_count})")
                saved_count += 1
                
                # Планируем следующий кадр для сохранения
                next_save_time += time_per_extracted_frame
            
            frame_count += 1
        
        cap.release()
        
        actual_fps = saved_count / duration if duration > 0 else 0
        logger.info(f"Extracted {saved_count} frames from {video_path}")
        logger.info(f"Actual extraction rate: {actual_fps:.2f} fps (target: {fps:.2f} fps)")
        
        return saved_frames


def extract_frames_from_video(
    video_path: str,
    output_dir: str = "./frames",
    fps: float = 1.0,
) -> dict:
    """Extract frames from video (convenience function).
    
    Args:
        video_path: Path to video file
        output_dir: Output directory
        fps: Frames per second to extract (float for precise timing)
        
    Returns:
        Dict with:
            - frames: List of frame paths
            - count: Number of frames
            - output_dir: Output directory
            - actual_fps: Actual extraction rate
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
