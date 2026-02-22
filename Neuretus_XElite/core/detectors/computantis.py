import torch
import cv2
import numpy as np
import json
import os
from ultralytics import YOLO
from typing import Dict, Tuple, Optional
from datetime import datetime

from .base import DocumentDetector
from .utils import reorder_corners


class ComputantisDetector(DocumentDetector):
    """YOLO-pose детектор документов (ключевые точки)"""
    
    def __init__(self, model_path: str, output_dir: str, device: Optional[str] = None):
        """
        Args:
            model_path: путь к .pt файлу YOLO-pose модели
            output_dir: директория для сохранения результатов
            device: 'cuda', 'mps', 'cpu' или None (автоопределение)
        """
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.backends.cudnn.is_available() else "cpu"
        
        self.device = device
        self.model = YOLO(model_path)
        self.output_dir = output_dir
        
        self.detector_output_dir = os.path.join(output_dir, "computantis_detector_output")
        os.makedirs(self.detector_output_dir, exist_ok=True)
    
    @property
    def name(self) -> str:
        return "Computantis (YOLO-pose)"
    
    def _save_detection_results(self, image: np.ndarray, corners: Dict[str, Tuple[int, int]], 
                                bbox: Tuple[int, int, int, int], conf_th: float):
        """
        Сохраняет входное изображение и JSON с результатами детекции.
        
        Args:
            image: исходное изображение
            corners: словарь с углами
            bbox: bounding box документа
            conf_th: порог уверенности, использованный при детекции
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        image_filename = f"detector_input_{timestamp}.jpg"
        json_filename = f"detector_results_{timestamp}.json"
        
        
        image_path = os.path.join(self.detector_output_dir, image_filename)
        json_path = os.path.join(self.detector_output_dir, json_filename)
        
        
        cv2.imwrite(image_path, image)
        
        
        detection_data = {
            "timestamp": timestamp,
            "detector_name": self.name,
            "device": self.device,
            "confidence_threshold": conf_th,
            "image_info": {
                "filename": image_filename,
                "shape": {
                    "height": image.shape[0],
                    "width": image.shape[1],
                    "channels": image.shape[2] if len(image.shape) == 3 else 1
                }
            },
            "detection_results": {
                "corners": {
                    "TL": {"x": int(corners["TL"][0]), "y": int(corners["TL"][1])},
                    "TR": {"x": int(corners["TR"][0]), "y": int(corners["TR"][1])},
                    "BR": {"x": int(corners["BR"][0]), "y": int(corners["BR"][1])},
                    "BL": {"x": int(corners["BL"][0]), "y": int(corners["BL"][1])}
                },
                "bbox": {
                    "x1": int(bbox[0]),
                    "y1": int(bbox[1]),
                    "x2": int(bbox[2]),
                    "y2": int(bbox[3]),
                    "width": int(bbox[2] - bbox[0]),
                    "height": int(bbox[3] - bbox[1])
                }
            }
        }
        
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detection_data, f, indent=2, ensure_ascii=False)
        
        print(f"Результаты детекции сохранены в: {self.detector_output_dir}")
        print(f"  Изображение: {image_filename}")
        print(f"  JSON: {json_filename}")
        
        self._save_visualization(image, corners, bbox, timestamp)
    
    def _save_visualization(self, image: np.ndarray, corners: Dict[str, Tuple[int, int]], 
                            bbox: Tuple[int, int, int, int], timestamp: str):
        """
        Сохраняет визуализацию с отмеченными углами и bounding box.
        
        Args:
            image: исходное изображение
            corners: словарь с углами
            bbox: bounding box документа
            timestamp: временная метка для имени файла
        """
        
        vis_image = image.copy()
        
        
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        
        colors = {
            "TL": (0, 0, 255),  # Красный
            "TR": (0, 255, 0),  # Зеленый
            "BR": (255, 0, 0),  # Синий
            "BL": (255, 255, 0)  # Желтый
        }
        
        for corner_name, (x, y) in corners.items():
            cv2.circle(vis_image, (x, y), 8, colors.get(corner_name, (255, 255, 255)), -1)
            
            cv2.putText(vis_image, corner_name, (x + 10, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        
        vis_filename = f"detector_visualization_{timestamp}.jpg"
        vis_path = os.path.join(self.detector_output_dir, vis_filename)
        cv2.imwrite(vis_path, vis_image)
        print(f"  Визуализация: {vis_filename}")
    
    def detect(self, image: np.ndarray, conf_th: float = 0.3, **kwargs) -> Tuple[Dict[str, Tuple[int, int]], Tuple[int, int, int, int]]:
        """
        Детектирует документ через ключевые точки.
        
        Args:
            image: BGR изображение
            conf_th: порог уверенности для ключевых точек
            
        Returns:
            corners: словарь с 4 углами в порядке TL, TR, BR, BL
            bbox: (x1, y1, x2, y2) - bounding box документа
        """
        
        results = self.model(image, device=self.device)[0]
        
        if results.boxes is None or results.keypoints is None:
            raise RuntimeError("Document not found by Computantis detector")
        
        box = results.boxes.xyxy[0].cpu().numpy().astype(int)
        kpts = results.keypoints.xy[0].cpu().numpy()
        kpts_conf = results.keypoints.conf[0].cpu().numpy()
        
        labels = ["TL", "TR", "BR", "BL"]
        raw_corners = {}
        
        for i, ((x, y), conf) in enumerate(zip(kpts, kpts_conf)):
            if conf < conf_th:
                continue
            raw_corners[labels[i]] = (int(x), int(y))
        
        if len(raw_corners) != 4:
            raise RuntimeError(f"Not all corners detected: got {len(raw_corners)}")
        
        corners = reorder_corners(raw_corners)
        
        
        self._save_detection_results(image, corners, tuple(box), conf_th)
        
        return corners, tuple(box)
    
    def detect_batch(self, images: list[np.ndarray], conf_th: float = 0.3, **kwargs) -> list[Tuple[Dict[str, Tuple[int, int]], Tuple[int, int, int, int]]]:
        """
        Детектирует документы на нескольких изображениях.
        
        Args:
            images: список BGR изображений
            conf_th: порог уверенности для ключевых точек
            
        Returns:
            Список кортежей (corners, bbox) для каждого изображения
        """
        results = []
        for i, image in enumerate(images):
            try:
                corners, bbox = self.detect(image, conf_th=conf_th, **kwargs)
                results.append((corners, bbox))
            except RuntimeError as e:
                print(f"Ошибка детекции на изображении {i}: {e}")
                results.append((None, None))
        
        return results
    
    def get_output_directory(self) -> str:
        """Возвращает путь к директории, где сохраняются результаты."""
        return self.detector_output_dir