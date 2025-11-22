"""Detect swimmer and lane in video frames using YOLO."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import cv2
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class SwimmerDetector:
    """Detect swimmer position in swimming pool frames."""

    def __init__(self, model_name: str = "yolov8n.pt"):
        """Initialize swimmer detector.

        Args:
            model_name: YOLO model name (n/s/m/l/x)
        """
        self.model = YOLO(model_name)
        # Person class ID in COCO dataset
        self.person_class_id = 0

    def detect_swimmer(
        self,
        frame_path: str,
        confidence_threshold: float = 0.5,
    ) -> Dict:
        """Detect swimmer in a single frame.

        Args:
            frame_path: Path to frame image
            confidence_threshold: Minimum confidence for detection

        Returns:
            Dict with:
                - bbox: [x1, y1, x2, y2] or None
                - confidence: float or None
                - center: [cx, cy] or None
                - lane: estimated lane number (1-8) or None
        """
        # Load frame
        frame = cv2.imread(frame_path)
        if frame is None:
            logger.error(f"Cannot read frame: {frame_path}")
            return {"bbox": None, "confidence": None, "center": None, "lane": None}

        height, width = frame.shape[:2]

        # Run detection
        results = self.model(frame, verbose=False)

        # Find person with highest confidence
        best_detection = None
        best_conf = 0

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == self.person_class_id and conf >= confidence_threshold:
                    if conf > best_conf:
                        best_conf = conf
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        best_detection = {
                            "bbox": [x1, y1, x2, y2],
                            "confidence": conf,
                        }

        if best_detection is None:
            return {"bbox": None, "confidence": None, "center": None, "lane": None}

        # Calculate center
        bbox = best_detection["bbox"]
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        best_detection["center"] = [cx, cy]

        # Estimate lane (divide width into 8 lanes)
        lane = min(8, max(1, int((cx / width) * 8) + 1))
        best_detection["lane"] = lane

        return best_detection

    def detect_batch(
        self,
        frame_paths: List[str],
        confidence_threshold: float = 0.5,
    ) -> List[Dict]:
        """Detect swimmer in multiple frames.

        Args:
            frame_paths: List of frame paths
            confidence_threshold: Minimum confidence

        Returns:
            List of detection results for each frame
        """
        results = []
        for i, frame_path in enumerate(frame_paths):
            logger.info(f"Processing frame {i+1}/{len(frame_paths)}")
            detection = self.detect_swimmer(frame_path, confidence_threshold)
            detection["frame_index"] = i
            detection["frame_path"] = frame_path
            results.append(detection)

        return results

    def draw_detections(
        self,
        frame_path: str,
        detection: Dict,
        output_path: str = None,
    ) -> str:
        """Draw bounding box on frame.

        Args:
            frame_path: Input frame path
            detection: Detection result
            output_path: Output path (default: add _detected suffix)

        Returns:
            Path to output image
        """
        frame = cv2.imread(frame_path)

        if detection["bbox"] is not None:
            x1, y1, x2, y2 = detection["bbox"]
            conf = detection["confidence"]
            lane = detection["lane"]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"Swimmer (Lane {lane}): {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                (0, 255, 0),
                -1,
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

            # Draw center point
            cx, cy = detection["center"]
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Save output
        if output_path is None:
            path = Path(frame_path)
            output_path = str(path.parent / f"{path.stem}_detected{path.suffix}")

        cv2.imwrite(output_path, frame)
        logger.info(f"Saved detection image: {output_path}")

        return output_path

    def save_detections_json(
        self,
        detections: List[Dict],
        output_path: str = "detections.json",
    ) -> str:
        """Save detections to JSON file.

        Args:
            detections: List of detection results
            output_path: Output JSON path

        Returns:
            Path to saved JSON
        """
        # Remove non-serializable frame objects
        clean_detections = []
        for det in detections:
            clean_det = {
                "frame_index": det["frame_index"],
                "bbox": det["bbox"],
                "confidence": det["confidence"],
                "center": det["center"],
                "lane": det["lane"],
            }
            clean_detections.append(clean_det)

        with open(output_path, "w") as f:
            json.dump(clean_detections, f, indent=2)

        logger.info(f"Saved detections to {output_path}")
        return output_path


def detect_swimmer_in_frames(
    frame_paths: List[str],
    output_dir: str = "./detections",
    draw_boxes: bool = True,
) -> Dict:
    """Detect swimmer in frames (convenience function).

    Args:
        frame_paths: List of frame paths
        output_dir: Output directory for results
        draw_boxes: Whether to draw bounding boxes

    Returns:
        Dict with detection results and paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    detector = SwimmerDetector()
    detections = detector.detect_batch(frame_paths)

    # Save JSON
    json_path = detector.save_detections_json(
        detections,
        str(output_path / "detections.json"),
    )

    # Draw boxes
    detected_images = []
    if draw_boxes:
        for detection in detections:
            if detection["bbox"] is not None:
                img_path = detector.draw_detections(
                    detection["frame_path"],
                    detection,
                    str(output_path / f"detected_{detection['frame_index']:04d}.jpg"),
                )
                detected_images.append(img_path)

    return {
        "detections": detections,
        "json_path": json_path,
        "detected_images": detected_images,
        "output_dir": str(output_path),
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Test on sample frames
    frame_paths = [f"./frames/frame_{i:04d}.jpg" for i in range(10)]

    result = detect_swimmer_in_frames(
        frame_paths,
        output_dir="./detection_results",
    )

    print(f"Detected in {len(result['detections'])} frames")
    print(f"Results saved to {result['output_dir']}")
