"""
🎬 Video Tools for Analysis

Features:
- Side-by-side video comparison
- Highlight clip extraction
- Zoom on specific regions
- Slow-motion export
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Video metadata."""
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float


def get_video_info(video_path: str) -> Optional[VideoInfo]:
    """Get video metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    info = VideoInfo(
        path=video_path,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=cap.get(cv2.CAP_PROP_FPS),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        duration_sec=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    )
    cap.release()
    return info


# ============================================================================
# SIDE-BY-SIDE COMPARISON
# ============================================================================

def create_side_by_side(
    video1_path: str,
    video2_path: str,
    output_path: str,
    labels: Tuple[str, str] = ("Відео 1", "Відео 2"),
    sync_start: bool = True,
) -> Optional[str]:
    """
    Create side-by-side comparison video.
    
    Args:
        video1_path: Path to first video
        video2_path: Path to second video
        output_path: Path for output video
        labels: Labels for each video
        sync_start: Sync videos from start
        
    Returns:
        Output path if successful
    """
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    if not cap1.isOpened() or not cap2.isOpened():
        logger.error("Failed to open videos")
        return None
    
    # Get video properties
    w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    
    # Use average FPS
    fps = (fps1 + fps2) / 2
    
    # Normalize heights
    target_height = max(h1, h2)
    scale1 = target_height / h1
    scale2 = target_height / h2
    
    new_w1 = int(w1 * scale1)
    new_w2 = int(w2 * scale2)
    
    # Output dimensions
    output_width = new_w1 + new_w2 + 20  # 20px gap
    output_height = target_height + 50  # Space for labels
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    if not out.isOpened():
        logger.error("Failed to create output video")
        return None
    
    frame_count = 0
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 and not ret2:
            break
        
        # Create output frame
        output_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        output_frame[:] = (30, 30, 40)  # Dark background
        
        # Resize and place frames
        if ret1:
            frame1_resized = cv2.resize(frame1, (new_w1, target_height))
            output_frame[50:50+target_height, 0:new_w1] = frame1_resized
        
        if ret2:
            frame2_resized = cv2.resize(frame2, (new_w2, target_height))
            output_frame[50:50+target_height, new_w1+20:new_w1+20+new_w2] = frame2_resized
        
        # Draw labels
        cv2.putText(output_frame, labels[0], (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(output_frame, labels[1], (new_w1 + 30, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 200), 2, cv2.LINE_AA)
        
        # Draw center line
        cv2.line(output_frame, (new_w1 + 10, 50), (new_w1 + 10, output_height),
                (100, 100, 100), 2)
        
        out.write(output_frame)
        frame_count += 1
    
    cap1.release()
    cap2.release()
    out.release()
    
    logger.info(f"Side-by-side video created: {output_path} ({frame_count} frames)")
    return output_path


# ============================================================================
# HIGHLIGHT CLIPS
# ============================================================================

def extract_highlight(
    video_path: str,
    output_path: str,
    start_sec: float,
    end_sec: float,
    add_text: str = None,
    slow_factor: float = 1.0,
) -> Optional[str]:
    """
    Extract highlight clip from video.
    
    Args:
        video_path: Source video path
        output_path: Output path
        start_sec: Start time in seconds
        end_sec: End time in seconds
        add_text: Optional text overlay
        slow_factor: Slow motion factor (0.5 = half speed)
        
    Returns:
        Output path if successful
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    
    # Output FPS adjusted for slow motion
    output_fps = fps * slow_factor
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_num = start_frame
    while frame_num < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add text overlay if provided
        if add_text:
            cv2.rectangle(frame, (10, height - 50), (width - 10, height - 10), (0, 0, 0), -1)
            cv2.putText(frame, add_text, (20, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        out.write(frame)
        frame_num += 1
    
    cap.release()
    out.release()
    
    logger.info(f"Highlight extracted: {output_path}")
    return output_path


def find_highlights(
    analysis_data: Dict,
    video_path: str,
    fps: float,
) -> List[Dict]:
    """
    Find highlight moments in video based on analysis.
    
    Returns list of highlight suggestions with timestamps.
    """
    highlights = []
    
    # Check for stroke analysis
    stroke_data = analysis_data.get("stroke_analysis")
    if stroke_data:
        strokes = getattr(stroke_data, 'strokes', [])
        
        # Best symmetry moments
        # Fastest strokes
        if strokes:
            fastest = min(strokes, key=lambda s: s.duration_sec)
            highlights.append({
                "type": "fastest_stroke",
                "title": "🚀 Найшвидший гребок",
                "start_frame": fastest.start_frame,
                "end_frame": fastest.end_frame,
                "start_sec": fastest.start_frame / fps,
                "end_sec": fastest.end_frame / fps,
            })
    
    # Check for pose data
    pose_data = analysis_data.get("swimming_pose", {})
    frame_analyses = pose_data.get("frame_analyses", [])
    
    if frame_analyses:
        # Best streamline moment
        best_streamline_idx = 0
        best_streamline = 0
        
        for i, fa in enumerate(frame_analyses):
            sl = fa.get("streamline", 0)
            if sl > best_streamline:
                best_streamline = sl
                best_streamline_idx = i
        
        if best_streamline > 0:
            highlights.append({
                "type": "best_streamline",
                "title": f"✨ Найкращий streamline ({best_streamline:.0f}/100)",
                "start_frame": max(0, best_streamline_idx - 10),
                "end_frame": best_streamline_idx + 10,
                "start_sec": max(0, (best_streamline_idx - 10) / fps),
                "end_sec": (best_streamline_idx + 10) / fps,
            })
    
    return highlights


# ============================================================================
# ZOOM VIDEO
# ============================================================================

def create_zoom_video(
    video_path: str,
    output_path: str,
    region: Tuple[int, int, int, int],  # x, y, width, height
    zoom_factor: float = 2.0,
    output_size: Tuple[int, int] = None,
) -> Optional[str]:
    """
    Create zoomed video focusing on specific region.
    
    Args:
        video_path: Source video
        output_path: Output path
        region: (x, y, width, height) region to zoom
        zoom_factor: Zoom multiplier
        output_size: Output video dimensions
        
    Returns:
        Output path if successful
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    x, y, w, h = region
    
    # Calculate crop region
    crop_x = max(0, x - w // 2)
    crop_y = max(0, y - h // 2)
    crop_w = min(w * int(zoom_factor), orig_width - crop_x)
    crop_h = min(h * int(zoom_factor), orig_height - crop_y)
    
    # Output size
    if output_size:
        out_w, out_h = output_size
    else:
        out_w, out_h = orig_width, orig_height
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop region
        cropped = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        
        # Resize to output
        zoomed = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        
        # Add zoom indicator
        cv2.putText(zoomed, f"ZOOM {zoom_factor}x", (out_w - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        out.write(zoomed)
    
    cap.release()
    out.release()
    
    logger.info(f"Zoom video created: {output_path}")
    return output_path


def create_tracked_zoom(
    video_path: str,
    output_path: str,
    detections: List[Dict],
    zoom_factor: float = 1.5,
    smooth_tracking: bool = True,
) -> Optional[str]:
    """
    Create zoom video that tracks detected person.
    
    Args:
        video_path: Source video
        output_path: Output path
        detections: List of detection dicts with bbox
        zoom_factor: Zoom multiplier
        smooth_tracking: Smooth camera movement
        
    Returns:
        Output path if successful
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Calculate crop dimensions
    crop_w = int(width / zoom_factor)
    crop_h = int(height / zoom_factor)
    
    prev_cx, prev_cy = width // 2, height // 2
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get center from detection
        if frame_idx < len(detections) and detections[frame_idx].get("bbox"):
            bbox = detections[frame_idx]["bbox"]
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
        else:
            cx, cy = prev_cx, prev_cy
        
        # Smooth tracking
        if smooth_tracking:
            cx = int(prev_cx * 0.7 + cx * 0.3)
            cy = int(prev_cy * 0.7 + cy * 0.3)
        
        prev_cx, prev_cy = cx, cy
        
        # Calculate crop region
        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        x2 = min(width, x1 + crop_w)
        y2 = min(height, y1 + crop_h)
        
        # Adjust if near edges
        if x2 - x1 < crop_w:
            x1 = max(0, x2 - crop_w)
        if y2 - y1 < crop_h:
            y1 = max(0, y2 - crop_h)
        
        # Crop and resize
        cropped = frame[y1:y2, x1:x2]
        zoomed = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # Add indicator
        cv2.putText(zoomed, f"TRACKING ZOOM {zoom_factor}x", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        out.write(zoomed)
        frame_idx += 1
    
    cap.release()
    out.release()
    
    logger.info(f"Tracked zoom video created: {output_path}")
    return output_path
