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
        confidence_threshold: float = 0.3,
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
        confidence_threshold: float = 0.3,
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
        frame_paths: List,
        confidence_threshold: float = 0.5,
    ) -> List[Dict]:
        """Detect and track the same swimmer across frames.
        
        Uses IoU and distance to maintain swimmer identity when multiple people present.
        
        Args:
            frame_paths: List of frame paths (str) or frame dicts with 'path' and 'timestamp'
            confidence_threshold: Minimum confidence
            
        Returns:
            List of detection results tracking the same swimmer
        """
        results = []
        prev_bbox = None
        prev_center = None
        prev_prev_center = None  # Для расчёта velocity
        target_lane = None
        velocity = [0, 0]  # Скорость движения [vx, vy] в пикселях/кадр
        frame_width = None  # Ширина кадра (получим из первого кадра)
        
        for i, frame_info in enumerate(frame_paths):
            # Поддержка старого формата (str) и нового (dict)
            if isinstance(frame_info, dict):
                frame_path = frame_info["path"]
                timestamp = frame_info.get("timestamp")
                video_frame = frame_info.get("video_frame")
            else:
                frame_path = frame_info
                timestamp = None
                video_frame = None
            
            # Получаем размер кадра если ещё не получили
            if frame_width is None:
                temp_frame = cv2.imread(frame_path)
                if temp_frame is not None:
                    frame_width = temp_frame.shape[1]
                else:
                    frame_width = 1920  # Fallback
            
            # Get all person detections in frame
            all_detections = self._detect_all_persons(frame_path, confidence_threshold)
            
            if not all_detections:
                # YOLO не нашёл пловца - пытаемся underwater detection
                underwater_det = None
                
                # Попытка 1: Motion-based detection (если есть предыдущий кадр)
                if prev_bbox and i > 0:
                    # Расширяем search bbox на основе velocity
                    search_margin = 100
                    if velocity and (velocity[0] != 0 or velocity[1] != 0):
                        # Предсказываем где искать
                        predicted_center = [
                            prev_center[0] + velocity[0],
                            prev_center[1] + velocity[1]
                        ]
                        search_bbox = [
                            predicted_center[0] - (prev_bbox[2] - prev_bbox[0]) // 2 - search_margin,
                            predicted_center[1] - (prev_bbox[3] - prev_bbox[1]) // 2 - search_margin,
                            predicted_center[0] + (prev_bbox[2] - prev_bbox[0]) // 2 + search_margin,
                            predicted_center[1] + (prev_bbox[3] - prev_bbox[1]) // 2 + search_margin,
                        ]
                    else:
                        # Без velocity - ищем вокруг предыдущей позиции
                        search_bbox = [
                            prev_bbox[0] - search_margin,
                            prev_bbox[1] - search_margin,
                            prev_bbox[2] + search_margin,
                            prev_bbox[3] + search_margin,
                        ]
                    
                    prev_frame_info = frame_paths[i - 1]
                    prev_frame_path_str = prev_frame_info["path"] if isinstance(prev_frame_info, dict) else prev_frame_info
                    
                    underwater_det = self._detect_underwater_with_motion(
                        frame_path,
                        prev_frame_path_str,
                        search_bbox
                    )
                    
                    if underwater_det:
                        logger.info(f"Frame {i}: 🌊 Underwater motion detection SUCCESS")
                
                # Попытка 2: YOLO с пониженным confidence (0.15 вместо 0.3)
                if not underwater_det:
                    ultra_low_detections = self._detect_all_persons(frame_path, confidence_threshold=0.15)
                    if ultra_low_detections:
                        # Используем ближайшую к predicted position
                        if prev_center and velocity:
                            predicted_center = [
                                prev_center[0] + velocity[0],
                                prev_center[1] + velocity[1]
                            ]
                            best_det = min(ultra_low_detections, key=lambda d: 
                                np.sqrt((d["center"][0] - predicted_center[0])**2 + 
                                       (d["center"][1] - predicted_center[1])**2))
                            underwater_det = best_det
                            underwater_det["ultra_low_confidence"] = True
                            logger.info(f"Frame {i}: 🌊 Ultra-low confidence YOLO SUCCESS (conf={best_det['confidence']:.2f})")
                
                # Попытка 3: Aggressive extrapolation по velocity
                if not underwater_det and prev_bbox and prev_center and velocity:
                    if velocity[0] != 0 or velocity[1] != 0:
                        # Считаем сколько кадров подряд пропущено
                        frames_missing = 1  # Пока просто 1
                        underwater_det = self._aggressive_extrapolate(
                            prev_bbox, prev_center, velocity, frames_missing,
                            target_lane=target_lane,  # Передаём известную дорожку
                            frame_width=frame_width   # Передаём ширину кадра
                        )
                        logger.info(f"Frame {i}: ⚡ Aggressive extrapolation USED")
                
                # Если всё равно ничего не нашли - пустая детекция
                if not underwater_det:
                    logger.warning(f"Frame {i}: ❌ No detection (YOLO, motion, extrapolation all failed)")
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
                
                # Используем underwater detection
                best_det = underwater_det
            else:
                # YOLO нашёл детекции - используем обычный трекинг
                # First frame: pick largest/most centered detection
                if prev_bbox is None:
                    best_det = self._pick_initial_swimmer(all_detections)
                    target_lane = best_det.get("lane")
                else:
                    # Вычисляем velocity если есть история движения
                    if prev_center and prev_prev_center:
                        velocity = [
                            prev_center[0] - prev_prev_center[0],  # vx
                            prev_center[1] - prev_prev_center[1],  # vy
                        ]
                    
                    # Предсказываем следующую позицию используя velocity
                    predicted_center = None
                    if prev_center and (velocity[0] != 0 or velocity[1] != 0):
                        predicted_center = [
                            prev_center[0] + velocity[0],  # expected x
                            prev_center[1] + velocity[1],  # expected y
                        ]
                        logger.debug(f"Frame {i}: velocity=[{velocity[0]:.1f}, {velocity[1]:.1f}], predicted={predicted_center}")
                    
                    # Track: pick detection closest to predicted position
                    best_det = self._pick_tracked_swimmer(
                        all_detections, prev_bbox, prev_center, target_lane,
                        predicted_center=predicted_center,  # Передаём предсказание!
                        velocity=velocity
                    )
            
            best_det["frame_index"] = i
            best_det["frame_path"] = frame_path
            best_det["timestamp"] = timestamp  # Реальное время из видео (сек)
            best_det["video_frame"] = video_frame  # Номер кадра в оригинальном видео
            results.append(best_det)
            
            # Update tracking state with velocity
            prev_prev_center = prev_center  # Сохраняем для расчёта velocity
            prev_bbox = best_det.get("bbox")
            prev_center = best_det.get("center")
            if best_det.get("lane"):
                target_lane = best_det["lane"]
        
        # Интерполяция пропущенных кадров (важно для подводных сцен)
        results = self._interpolate_missing_detections(results)
        
        # Сглаживание траектории (убирает "пьяный" эффект)
        results = self._smooth_trajectory(results, window_size=5)
        
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
        target_lane: int,
        predicted_center: List[int] = None,
        velocity: List[float] = None,
    ) -> Dict:
        """Pick detection that best matches the previously tracked swimmer.
        
        Uses IoU, center distance, bbox size similarity, and lane consistency.
        НОВОЕ: использует velocity prediction для трекинга быстрых пловцов.
        
        Args:
            detections: All person detections in current frame
            prev_bbox: Previous frame bbox
            prev_center: Previous frame center
            target_lane: Expected lane number
            predicted_center: Predicted center position (from velocity)
            velocity: Movement velocity [vx, vy] in pixels/frame
            
        Returns:
            Best matching detection
        """
        best_det = None
        best_score = -1
        best_iou = 0
        
        # Вычисляем размер предыдущего bbox для сравнения
        prev_width = prev_bbox[2] - prev_bbox[0]
        prev_height = prev_bbox[3] - prev_bbox[1]
        prev_area = prev_width * prev_height
        
        # Вычисляем скорость движения для адаптивного search radius
        speed = 0
        if velocity:
            speed = np.sqrt(velocity[0]**2 + velocity[1]**2)  # Пикселей/кадр
        
        # Адаптивный max distance в зависимости от скорости
        # Быстрые пловцы: больше search radius
        base_max_dist = 150
        adaptive_max_dist = base_max_dist + speed * 1.5  # Расширяем зону поиска
        logger.debug(f"Speed: {speed:.1f}px/frame, search radius: {adaptive_max_dist:.1f}px")
        
        for det in detections:
            score = 0.0
            
            # 1. IoU similarity (0-1) - САМЫЙ ВАЖНЫЙ для одного объекта!
            iou = self._calculate_iou(prev_bbox, det["bbox"])
            score += iou * 10.0  # Weight: 10x (было 3x) ← увеличено!
            
            # 2. Center distance (inverse, normalized)
            # НОВОЕ: используем predicted_center если доступен!
            reference_center = predicted_center if predicted_center else prev_center
            
            if reference_center and det["center"]:
                dx = det["center"][0] - reference_center[0]
                dy = det["center"][1] - reference_center[1]
                dist = np.sqrt(dx**2 + dy**2)
                
                # Используем адаптивный max distance (расширяется для быстрых пловцов)
                dist_score = max(0, 1.0 - dist / adaptive_max_dist)
                score += dist_score * 5.0  # Weight: 5x
                
                # Бонус если детекция близко к предсказанной позиции
                if predicted_center and dist < speed * 0.5:  # В пределах половины скорости
                    score += 2.0  # Дополнительный бонус за точное предсказание!
            
            # 3. Размер bbox (похожий размер = тот же объект)
            det_width = det["bbox"][2] - det["bbox"][0]
            det_height = det["bbox"][3] - det["bbox"][1]
            det_area = det_width * det_height
            
            # Соотношение площадей (должно быть близко к 1.0)
            area_ratio = min(prev_area, det_area) / max(prev_area, det_area) if max(prev_area, det_area) > 0 else 0
            # Соотношение сторон (aspect ratio)
            prev_aspect = prev_width / prev_height if prev_height > 0 else 1
            det_aspect = det_width / det_height if det_height > 0 else 1
            aspect_similarity = min(prev_aspect, det_aspect) / max(prev_aspect, det_aspect) if max(prev_aspect, det_aspect) > 0 else 0
            
            # Bonus за похожий размер и форму
            size_score = (area_ratio + aspect_similarity) / 2.0
            score += size_score * 3.0  # Weight: 3x
            
            # 4. Lane consistency
            if target_lane and det["lane"] == target_lane:
                score += 2.0  # Weight: 2x (было 1x) ← увеличено!
            
            # 5. Confidence bonus (меньший вес - не главное)
            score += det["confidence"] * 0.3  # Weight: 0.3x (было 0.5x)
            
            if score > best_score:
                best_score = score
                best_iou = iou
                best_det = det
        
        # ЖЕСТКИЙ ПОРОГ: если лучший IoU слишком низкий, возможно это другой объект!
        # Не переключаемся на другого человека если overlap < 20%
        if best_iou < 0.2 and best_det is not None:
            logger.warning(f"Low IoU {best_iou:.3f} - possible object switch! Keeping previous bbox.")
            # Возвращаем "синтетическую" детекцию с предыдущим bbox
            # Это предотвратит перескакивание
            return {
                "bbox": prev_bbox,
                "center": prev_center,
                "confidence": 0.5,
                "lane": target_lane,
                "tracking_lost": True  # Помечаем что трекинг ненадежен
            }
        
        # Fallback: return highest confidence if tracking failed
        if best_det is None:
            best_det = max(detections, key=lambda d: d["confidence"])
            logger.warning("Tracking fallback: using highest confidence detection")
        
        logger.debug(f"Tracking score: {best_score:.2f}, IoU: {best_iou:.3f}")
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
    
    def _interpolate_missing_detections(self, detections: List[Dict]) -> List[Dict]:
        """Интерполировать bbox для кадров где детекция не сработала.
        
        Особенно важно для подводных сцен где YOLO может не детектировать пловца,
        но мы знаем что он там есть по соседним кадрам.
        
        Args:
            detections: Список детекций с пропусками (bbox=None)
            
        Returns:
            Список детекций с заполненными пропусками
        """
        if not detections:
            return detections
        
        # Находим все gaps (последовательности None)
        for i in range(len(detections)):
            if detections[i]["bbox"] is not None:
                continue
            
            # Ищем предыдущий валидный bbox
            prev_idx = None
            for j in range(i - 1, -1, -1):
                if detections[j]["bbox"] is not None:
                    prev_idx = j
                    break
            
            # Ищем следующий валидный bbox
            next_idx = None
            for j in range(i + 1, len(detections)):
                if detections[j]["bbox"] is not None:
                    next_idx = j
                    break
            
            # Если есть оба - интерполируем
            if prev_idx is not None and next_idx is not None:
                prev_bbox = detections[prev_idx]["bbox"]
                next_bbox = detections[next_idx]["bbox"]
                
                # Линейная интерполяция
                gap_size = next_idx - prev_idx
                position = i - prev_idx
                ratio = position / gap_size
                
                interpolated_bbox = [
                    int(prev_bbox[0] + (next_bbox[0] - prev_bbox[0]) * ratio),
                    int(prev_bbox[1] + (next_bbox[1] - prev_bbox[1]) * ratio),
                    int(prev_bbox[2] + (next_bbox[2] - prev_bbox[2]) * ratio),
                    int(prev_bbox[3] + (next_bbox[3] - prev_bbox[3]) * ratio),
                ]
                
                cx = (interpolated_bbox[0] + interpolated_bbox[2]) // 2
                cy = (interpolated_bbox[1] + interpolated_bbox[3]) // 2
                
                detections[i]["bbox"] = interpolated_bbox
                detections[i]["center"] = [cx, cy]
                detections[i]["confidence"] = 0.5  # Средняя уверенность для интерполяции
                detections[i]["interpolated"] = True  # Помечаем как интерполированный
                
                logger.debug(f"Interpolated bbox for frame {i} (underwater gap)")
            
            # Если есть только prev - экстраполируем вперёд (до 3 кадров)
            elif prev_idx is not None and (i - prev_idx) <= 3:
                detections[i]["bbox"] = detections[prev_idx]["bbox"]
                detections[i]["center"] = detections[prev_idx]["center"]
                detections[i]["confidence"] = 0.4
                detections[i]["interpolated"] = True
                logger.debug(f"Extrapolated bbox forward for frame {i}")
            
            # Если есть только next - экстраполируем назад (до 3 кадров)
            elif next_idx is not None and (next_idx - i) <= 3:
                detections[i]["bbox"] = detections[next_idx]["bbox"]
                detections[i]["center"] = detections[next_idx]["center"]
                detections[i]["confidence"] = 0.4
                detections[i]["interpolated"] = True
                logger.debug(f"Extrapolated bbox backward for frame {i}")
        
        return detections
    
    def _smooth_trajectory(
        self, detections: List[Dict], window_size: int = 5
    ) -> List[Dict]:
        """Сглаживание траектории bbox для устранения дрожания.
        
        Использует скользящее среднее для координат bbox,
        что убирает "пьяный" эффект и запаздывания.
        
        Args:
            detections: Список детекций
            window_size: Размер окна для скользящего среднего (нечётное число)
            
        Returns:
            Список детекций со сглаженными bbox
        """
        if not detections or window_size < 3:
            return detections
        
        # Убеждаемся что window_size нечётный
        if window_size % 2 == 0:
            window_size += 1
        
        half_window = window_size // 2
        smoothed_detections = []
        
        for i, detection in enumerate(detections):
            if detection["bbox"] is None:
                smoothed_detections.append(detection)
                continue
            
            # Собираем bbox из окна вокруг текущего кадра
            window_bboxes = []
            for j in range(max(0, i - half_window), min(len(detections), i + half_window + 1)):
                if detections[j]["bbox"] is not None:
                    window_bboxes.append(detections[j]["bbox"])
            
            if not window_bboxes:
                smoothed_detections.append(detection)
                continue
            
            # Вычисляем среднее значение для каждой координаты
            avg_bbox = [
                int(sum(bbox[0] for bbox in window_bboxes) / len(window_bboxes)),  # x1
                int(sum(bbox[1] for bbox in window_bboxes) / len(window_bboxes)),  # y1
                int(sum(bbox[2] for bbox in window_bboxes) / len(window_bboxes)),  # x2
                int(sum(bbox[3] for bbox in window_bboxes) / len(window_bboxes)),  # y2
            ]
            
            # Обновляем center
            cx = (avg_bbox[0] + avg_bbox[2]) // 2
            cy = (avg_bbox[1] + avg_bbox[3]) // 2
            
            # Создаём сглаженную детекцию
            smoothed_det = detection.copy()
            smoothed_det["bbox"] = avg_bbox
            smoothed_det["center"] = [cx, cy]
            smoothed_det["smoothed"] = True
            
            smoothed_detections.append(smoothed_det)
            
            if i % 50 == 0:
                logger.debug(f"Smoothed bbox for frame {i}: window={len(window_bboxes)} bboxes")
        
        logger.info(f"Trajectory smoothing applied (window_size={window_size})")
        return smoothed_detections
    
    def _estimate_lane(self, center_x: int, frame_width: int) -> int:
        """Оценить номер дорожки по горизонтальной позиции.
        
        Args:
            center_x: X координата центра пловца
            frame_width: Ширина кадра
            
        Returns:
            Lane number (1-8)
        """
        # Делим ширину кадра на 8 дорожек
        lane = min(8, max(1, int((center_x / frame_width) * 8) + 1))
        return lane
    
    def _detect_underwater_with_motion(
        self,
        frame_path: str,
        prev_frame_path: str = None,
        search_bbox: List[int] = None,
    ) -> Dict:
        """Детекция под водой используя motion detection.
        
        Когда YOLO не видит пловца под водой, используем разницу кадров
        для детекции движения в ожидаемой области.
        
        Args:
            frame_path: Текущий кадр
            prev_frame_path: Предыдущий кадр для сравнения
            search_bbox: Область поиска [x1, y1, x2, y2]
            
        Returns:
            Detection dict или None
        """
        if not prev_frame_path:
            return None
        
        frame = cv2.imread(frame_path)
        prev_frame = cv2.imread(prev_frame_path)
        
        if frame is None or prev_frame is None:
            return None
        
        # Конвертируем в grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Если есть search_bbox, ищем только там
        if search_bbox:
            x1, y1, x2, y2 = search_bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            gray_roi = gray[y1:y2, x1:x2]
            prev_gray_roi = prev_gray[y1:y2, x1:x2]
        else:
            gray_roi = gray
            prev_gray_roi = prev_gray
            x1, y1 = 0, 0
        
        # Frame difference для детекции движения
        diff = cv2.absdiff(gray_roi, prev_gray_roi)
        
        # Threshold для бинаризации
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Морфологические операции для удаления шума
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Находим контуры движущихся объектов
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Берём самый большой контур (вероятно пловец)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Минимальная площадь для детекции (фильтруем шум)
        min_area = 5000  # пикселей
        if area < min_area:
            logger.debug(f"Motion detected but area too small: {area}px < {min_area}px")
            return None
        
        # Получаем bounding box контура
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Преобразуем в координаты полного кадра
        bbox = [x1 + x, y1 + y, x1 + x + w, y1 + y + h]
        center = [bbox[0] + (bbox[2] - bbox[0]) // 2, bbox[1] + (bbox[3] - bbox[1]) // 2]
        
        logger.info(f"🌊 Underwater motion detected: area={area}px, bbox={bbox}")
        
        return {
            "bbox": bbox,
            "center": center,
            "confidence": 0.5,  # Medium confidence для motion-based
            "lane": self._estimate_lane(center[0], frame.shape[1]),
            "underwater_detection": True,  # Флаг что это underwater detection
        }
    
    def _aggressive_extrapolate(
        self,
        prev_bbox: List[int],
        prev_center: List[int],
        velocity: List[float],
        frames_missing: int = 1,
        target_lane: int = None,
        frame_width: int = 1920,
    ) -> Dict:
        """Агрессивная экстраполяция bbox по velocity для подводных участков.
        
        Args:
            prev_bbox: Последний известный bbox
            prev_center: Последний известный center
            velocity: Скорость движения [vx, vy]
            frames_missing: Сколько кадров пропущено
            target_lane: Известная дорожка (если есть)
            frame_width: Ширина кадра (для оценки lane если target_lane=None)
            
        Returns:
            Extrapolated detection dict
        """
        # Экстраполируем center
        new_center = [
            int(prev_center[0] + velocity[0] * frames_missing),
            int(prev_center[1] + velocity[1] * frames_missing),
        ]
        
        # Экстраполируем bbox (сохраняем размер)
        bbox_width = prev_bbox[2] - prev_bbox[0]
        bbox_height = prev_bbox[3] - prev_bbox[1]
        
        new_bbox = [
            new_center[0] - bbox_width // 2,
            new_center[1] - bbox_height // 2,
            new_center[0] + bbox_width // 2,
            new_center[1] + bbox_height // 2,
        ]
        
        # Используем известную lane если есть, иначе оцениваем
        lane = target_lane if target_lane else self._estimate_lane(new_center[0], frame_width)
        
        logger.debug(f"⚡ Aggressive extrapolation: center={new_center}, velocity={velocity}")
        
        return {
            "bbox": new_bbox,
            "center": new_center,
            "confidence": 0.3,  # Low confidence для extrapolated
            "lane": lane,
            "extrapolated": True,  # Флаг что это экстраполяция
        }


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
