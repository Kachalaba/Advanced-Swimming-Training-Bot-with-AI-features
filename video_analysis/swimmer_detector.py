"""Detect swimmer and lane in video frames using YOLO."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
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
        enable_tracking: bool = True,
    ) -> List[Dict]:
        """Detect swimmer in multiple frames with optional tracking.

        Args:
            frame_paths: List of frame paths
            confidence_threshold: Minimum confidence
            enable_tracking: If True, track same swimmer across frames

        Returns:
            List of detection results for each frame
        """
        if not enable_tracking:
            # Simple per-frame detection
            results = []
            for i, frame_path in enumerate(frame_paths):
                detection = self.detect_swimmer(frame_path, confidence_threshold)
                detection["frame_index"] = i
                detection["frame_path"] = frame_path
                results.append(detection)
            return results

        # Tracking mode: maintain swimmer identity across frames
        return self._detect_with_tracking(frame_paths, confidence_threshold)

    def _detect_with_tracking(
        self,
        frame_paths: List[str],
        confidence_threshold: float = 0.5,
    ) -> List[Dict]:
        """Detect and track the same swimmer across frames.
        
        Uses IoU and distance to maintain swimmer identity when multiple people present.
        
        Args:
            frame_paths: List of frame paths
            confidence_threshold: Minimum confidence
            
        Returns:
            List of detection results tracking the same swimmer
        """
        results = []
        prev_bbox = None
        prev_center = None
        target_lane = None
        
        for i, frame_info in enumerate(frame_paths):
            # Support both dict format {"path": ..., "timestamp": ...} and plain string
            if isinstance(frame_info, dict):
                frame_path = frame_info.get("path", frame_info)
                timestamp = frame_info.get("timestamp", i / 30.0)
                video_frame = frame_info.get("video_frame", i)
            else:
                frame_path = frame_info
                timestamp = i / 30.0
                video_frame = i
            
            # Get all person detections in frame
            all_detections = self._detect_all_persons(frame_path, confidence_threshold)
            
            if not all_detections:
                # No detection found
                results.append({
                    "frame_index": i,
                    "frame_path": frame_path,
                    "timestamp": timestamp,
                    "video_frame": video_frame,
                    "bbox": None,
                    "confidence": None,
                    "center": None,
                    "lane": None,
                })
                continue
            
            # First frame: pick largest/most centered detection
            if prev_bbox is None:
                best_det = self._pick_initial_swimmer(all_detections)
                target_lane = best_det.get("lane")
            else:
                # Track: pick detection closest to previous
                best_det = self._pick_tracked_swimmer(
                    all_detections, prev_bbox, prev_center, target_lane
                )
            
            best_det["frame_index"] = i
            best_det["frame_path"] = frame_path
            best_det["timestamp"] = timestamp
            best_det["video_frame"] = video_frame
            # Store all bboxes for multi-person analysis
            best_det["all_boxes"] = [d["bbox"] for d in all_detections]
            best_det["all_detections"] = all_detections
            results.append(best_det)
            
            # Update tracking state
            prev_bbox = best_det.get("bbox")
            prev_center = best_det.get("center")
            if best_det.get("lane"):
                target_lane = best_det["lane"]
        
        return results
    
    def _detect_all_persons(
        self, frame_path: str, confidence_threshold: float
    ) -> List[Dict]:
        """Get all person detections in a frame.
        
        Args:
            frame_path: Path to frame
            confidence_threshold: Minimum confidence
            
        Returns:
            List of all person detections with bbox, confidence, center, lane
        """
        frame = cv2.imread(frame_path)
        if frame is None:
            return []
        
        height, width = frame.shape[:2]
        results = self.model(frame, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls == self.person_class_id and conf >= confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    lane = min(8, max(1, int((cx / width) * 8) + 1))
                    
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "center": [cx, cy],
                        "lane": lane,
                    })
        
        return detections
    
    def _pick_initial_swimmer(self, detections: List[Dict]) -> Dict:
        """Pick the most likely swimmer from first frame.
        
        Strategy: largest bbox (likely the main subject).
        
        Args:
            detections: All person detections
            
        Returns:
            Best detection
        """
        best_det = None
        best_area = 0
        
        for det in detections:
            bbox = det["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area > best_area:
                best_area = area
                best_det = det
        
        return best_det or detections[0]
    
    def _pick_tracked_swimmer(
        self,
        detections: List[Dict],
        prev_bbox: List[int],
        prev_center: List[int],
        target_lane: int = None,
    ) -> Dict:
        """Pick detection that best matches previous swimmer.
        
        Uses IoU, center distance, and lane consistency.
        
        Args:
            detections: All person detections in current frame
            prev_bbox: Previous frame bbox
            prev_center: Previous frame center
            target_lane: Expected lane number
            
        Returns:
            Best matching detection
        """
        best_det = None
        best_score = -1
        
        for det in detections:
            score = 0.0
            
            # IoU similarity (0-1)
            iou = self._calculate_iou(prev_bbox, det["bbox"])
            score += iou * 3.0  # Weight: 3x
            
            # Center distance (inverse, normalized)
            if prev_center and det["center"]:
                dx = det["center"][0] - prev_center[0]
                dy = det["center"][1] - prev_center[1]
                dist = np.sqrt(dx**2 + dy**2)
                # Max expected movement ~200px between frames
                dist_score = max(0, 1.0 - dist / 200.0)
                score += dist_score * 2.0  # Weight: 2x
            
            # Lane consistency
            if target_lane and det["lane"] == target_lane:
                score += 1.0  # Weight: 1x
            
            # Confidence bonus
            score += det["confidence"] * 0.5  # Weight: 0.5x
            
            if score > best_score:
                best_score = score
                best_det = det
        
        # Fallback: return highest confidence if tracking failed
        if best_det is None:
            best_det = max(detections, key=lambda d: d["confidence"])
        
        return best_det
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union between two bboxes.
        
        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]
            
        Returns:
            IoU value (0-1)
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

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
    enable_tracking: bool = True,
) -> Dict:
    """Detect swimmer in frames (convenience function).

    Args:
        frame_paths: List of frame paths
        output_dir: Output directory for results
        draw_boxes: Whether to draw bounding boxes
        enable_tracking: If True, track same swimmer across frames (recommended for multi-person scenes)

    Returns:
        Dict with detection results and paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    detector = SwimmerDetector()
    detections = detector.detect_batch(frame_paths, enable_tracking=enable_tracking)

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
