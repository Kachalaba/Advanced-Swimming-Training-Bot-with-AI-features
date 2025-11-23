"""Generate annotated swimming videos with overlays for axes and events."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence

import cv2

logger = logging.getLogger(__name__)


class VideoOverlayGenerator:
    """Create annotated videos from extracted frames and analysis results."""

    def __init__(self, output_dir: str = "./overlays", fps: float = 10.0):
        """Configure output directory and playback speed.

        Args:
            output_dir: Directory where the annotated video will be written.
            fps: Frames per second to encode the resulting video (default: 10).
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = max(5.0, fps)  # Minimum 5 fps for smooth playback

    def generate_annotated_video(
        self,
        frame_paths: Sequence[str],
        detections: List[Dict],
        analysis: Dict | None = None,
        output_path: str | None = None,
    ) -> str:
        """Build an annotated mp4 with axes, boxes, and key moments.

        Args:
            frame_paths: Ordered list of extracted frame paths.
            detections: Detection results that include ``frame_index`` and bbox data.
            analysis: Optional analysis dict that may contain wall touch frames.
            output_path: Optional custom path for the resulting video.

        Returns:
            Path to the saved annotated video.

        Raises:
            ValueError: If frames are missing or the encoder cannot be initialized.
        """
        if not frame_paths:
            raise ValueError("No frames provided for overlay generation")

        # Поддержка нового формата (dict) и старого (str)
        first_frame_info = frame_paths[0]
        if isinstance(first_frame_info, dict):
            first_frame_path = first_frame_info["path"]
        else:
            first_frame_path = first_frame_info
        
        first_frame = cv2.imread(first_frame_path)
        if first_frame is None:
            raise ValueError(f"Cannot read first frame: {first_frame_path}")

        height, width = first_frame.shape[:2]
        output_path = output_path or str(self.output_dir / "annotated_video.mp4")

        # Try better codecs for quality
        # h264 > avc1 > mp4v
        for codec in ["avc1", "mp4v"]:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))
            if writer.isOpened():
                logger.info(f"Using codec: {codec} at {self.fps} fps")
                break
        
        if not writer.isOpened():
            raise ValueError(f"Unable to open video writer for {output_path}")

        detection_map = {det.get("frame_index"): det for det in detections}
        touch_frames = self._extract_touch_frames(analysis)

        for frame_index, frame_info in enumerate(frame_paths):
            # Поддержка нового формата (dict) и старого (str)
            if isinstance(frame_info, dict):
                frame_path = frame_info["path"]
            else:
                frame_path = frame_info
            
            frame = cv2.imread(frame_path)
            if frame is None:
                logger.warning("Skipping unreadable frame: %s", frame_path)
                continue

            detection = detection_map.get(frame_index)
            if detection:
                self._draw_detection(frame, detection)
                self._draw_axes(frame, detection)

            if touch_frames:
                label = self._label_for_frame(frame_index, touch_frames)
                if label:
                    self._draw_moment_badge(frame, label)

            self._draw_timestamp(frame, frame_index)
            writer.write(frame)

        writer.release()
        logger.info("Saved annotated video: %s", output_path)
        return output_path

    def _draw_detection(self, frame, detection: Dict) -> None:
        """Overlay bounding box and lane label on a frame."""
        bbox = detection.get("bbox")
        if not bbox:
            return

        x1, y1, x2, y2 = bbox
        confidence = detection.get("confidence")
        lane = detection.get("lane")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 176, 80), 2)
        label = "Swimmer"
        if lane:
            label += f" | Lane {lane}"
        if confidence is not None:
            label += f" | {confidence:.2f}"

        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(
            frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 176, 80), -1
        )
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

    def _draw_axes(self, frame, detection: Dict) -> None:
        """Draw longitudinal and transverse axes through the detected center."""
        center = detection.get("center")
        if not center:
            return

        cx, cy = center
        height, width = frame.shape[:2]

        # Longitudinal axis (horizontal)
        cv2.line(frame, (0, cy), (width, cy), (255, 199, 0), 2)
        cv2.putText(
            frame,
            "Longitudinal axis",
            (10, max(20, cy - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (30, 30, 30),
            2,
        )

        # Transverse axis (vertical)
        cv2.line(frame, (cx, 0), (cx, height), (178, 102, 255), 2)
        cv2.putText(
            frame,
            "Transverse axis",
            (min(width - 200, cx + 10), 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (30, 30, 30),
            2,
        )

        # Center marker
        cv2.circle(frame, (cx, cy), 6, (0, 92, 255), -1)

    def _draw_timestamp(self, frame, frame_index: int) -> None:
        """Render relative timestamp in the bottom-left corner."""
        timestamp = frame_index / self.fps if self.fps else frame_index
        text = f"t = {timestamp:.1f}s"
        cv2.putText(
            frame,
            text,
            (12, frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (240, 240, 240),
            2,
        )

    def _draw_moment_badge(self, frame, label: str) -> None:
        """Draw a highlighted badge describing a key moment."""
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        padding = 10
        x1, y1 = 10, 10
        x2, y2 = x1 + text_w + padding * 2, y1 + text_h + padding * 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (53, 152, 219), 2)
        cv2.putText(
            frame,
            label,
            (x1 + padding, y2 - padding),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 40, 40),
            2,
        )

    def _label_for_frame(self, frame_index: int, touch_frames: List[int]) -> str | None:
        """Return a user-friendly label for start/turn/finish frames."""
        if frame_index not in touch_frames:
            return None

        position = touch_frames.index(frame_index)
        if position == 0:
            return "Start"
        if position == len(touch_frames) - 1:
            return "Finish"
        return f"Turn {position}"

    def _extract_touch_frames(self, analysis: Dict | None) -> List[int]:
        """Extract sorted wall-touch frames from analysis results."""
        if not analysis:
            return []
        wall_touches = analysis.get("wall_touches", {})
        frames = (
            wall_touches.get("frames", []) if isinstance(wall_touches, dict) else []
        )
        return sorted({int(frame) for frame in frames})


def generate_annotated_video(
    frame_paths: Sequence[str],
    detections: List[Dict],
    analysis: Dict | None = None,
    output_dir: str = "./overlays",
    fps: float = 1.0,
) -> str:
    """Convenience wrapper around :class:`VideoOverlayGenerator`.

    Args:
        frame_paths: Ordered list of frame file paths.
        detections: Detection results with frame indices and centers.
        analysis: Optional analysis dict containing wall touches.
        output_dir: Where to write the resulting mp4.
        fps: Frames per second for the encoded video.

    Returns:
        Path to the saved annotated video.
    """
    generator = VideoOverlayGenerator(output_dir=output_dir, fps=fps)
    return generator.generate_annotated_video(
        frame_paths,
        detections,
        analysis=analysis,
    )
