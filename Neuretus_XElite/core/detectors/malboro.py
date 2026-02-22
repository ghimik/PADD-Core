import torch
import cv2
import numpy as np
import json
import os
from ultralytics import YOLO
from typing import Dict, Tuple, Optional
from datetime import datetime

from .base import DocumentDetector
from .utils import mask_to_quad


class MalboroDetector(DocumentDetector):
    """YOLO-SEG детектор документов (маска → четырёхугольник)"""
    
    def __init__(self, model_path: str, output_dir: str, device: Optional[str] = None):
        """
        Args:
            model_path: путь к .pt файлу YOLO-SEG модели
            output_dir: директория для сохранения результатов
            device: 'cuda', 'mps', 'cpu' или None (автоопределение)
        """
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.backends.cudnn.is_available() else "cpu"
        
        self.device = device
        self.model = YOLO(model_path)
        self.output_dir = output_dir
        
        
        self.detector_output_dir = os.path.join(output_dir, "malboro_detector_output")
        os.makedirs(self.detector_output_dir, exist_ok=True)
    
    @property
    def name(self) -> str:
        return "Malboro (YOLO-SEG)"
    
    def _save_detection_results(self, image: np.ndarray, corners: Dict[str, Tuple[int, int]], 
                                bbox: Tuple[int, int, int, int], conf_th: float, mask: Optional[np.ndarray] = None):
        """
        Сохраняет входное изображение и JSON с результатами детекции.
        
        Args:
            image: исходное изображение
            corners: словарь с углами
            bbox: bounding box документа
            conf_th: порог уверенности, использованный при детекции
            mask: бинарная маска документа (опционально)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        
        image_filename = f"detector_input_malboro_{timestamp}.jpg"
        json_filename = f"detector_results_malboro_{timestamp}.json"
        
        
        image_path = os.path.join(self.detector_output_dir, image_filename)
        json_path = os.path.join(self.detector_output_dir, json_filename)
        
        
        cv2.imwrite(image_path, image)
        
        mask_filename = None
        if mask is not None:
            mask_filename = f"detector_mask_malboro_{timestamp}.png"
            mask_path = os.path.join(self.detector_output_dir, mask_filename)
            cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
        
        
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
        
        if mask_filename:
            detection_data["detection_results"]["mask"] = {
                "filename": mask_filename,
                "shape": mask.shape
            }
        
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detection_data, f, indent=2, ensure_ascii=False)
        
        print(f"Результаты детекции (Malboro) сохранены в: {self.detector_output_dir}")
        print(f"  Изображение: {image_filename}")
        print(f"  JSON: {json_filename}")
        if mask_filename:
            print(f"  Маска: {mask_filename}")
        
        
        self._save_visualization(image, corners, bbox, timestamp, mask)
    
    def _save_visualization(self, image: np.ndarray, corners: Dict[str, Tuple[int, int]], 
                            bbox: Tuple[int, int, int, int], timestamp: str, mask: Optional[np.ndarray] = None):
        """
        Сохраняет визуализацию с отмеченными углами, bounding box и маской.
        
        Args:
            image: исходное изображение
            corners: словарь с углами
            bbox: bounding box документа
            timestamp: временная метка для имени файла
            mask: бинарная маска документа (опционально)
        """
        
        vis_image = image.copy()
        
        if mask is not None:
            
            colored_mask = np.zeros_like(image)
            colored_mask[:, :, 1] = (mask * 255).astype(np.uint8)  
            
            vis_image = cv2.addWeighted(vis_image, 0.7, colored_mask, 0.3, 0)
        
        
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        
        colors = {
            "TL": (0, 0, 255),     # Красный
            "TR": (0, 255, 0),     # Зеленый
            "BR": (255, 0, 0),     # Синий
            "BL": (255, 255, 0)    # Желтый
        }
        
        for corner_name, (x, y) in corners.items():
            
            cv2.circle(vis_image, (x, y), 8, colors.get(corner_name, (255, 255, 255)), -1)
            
            cv2.putText(vis_image, corner_name, (x + 10, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        
        cv2.putText(vis_image, "Malboro (YOLO-SEG)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        
        vis_filename = f"detector_visualization_malboro_{timestamp}.jpg"
        vis_path = os.path.join(self.detector_output_dir, vis_filename)
        cv2.imwrite(vis_path, vis_image)
        print(f"  Визуализация: {vis_filename}")
    
    def detect(self, image: np.ndarray, conf_th: float = 0.3, **kwargs) -> Tuple[Dict[str, Tuple[int, int]], Tuple[int, int, int, int]]:
        """
        Детектирует документ через сегментацию.
        
        Args:
            image: BGR изображение
            conf_th: порог уверенности для детекции
            
        Returns:
            corners: словарь с 4 углами в порядке TL, TR, BR, BL
            bbox: (x1, y1, x2, y2) - bounding box маски
        """
        H, W = image.shape[:2]
        
        
        results = self.model(image, conf=conf_th, device=self.device)[0]
        
        if results.masks is None:
            raise RuntimeError("Document not found by Malboro detector")
        
        masks = results.masks.data.cpu().numpy()
        areas = masks.sum(axis=(1, 2))
        idx = np.argmax(areas)
        
        raw_mask = masks[idx]
        mask = cv2.resize(raw_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.uint8)
        
        quad = mask_to_quad(mask)
        if quad is None:
            raise RuntimeError("Failed to extract quad from mask")
        
        
        ys, xs = np.where(mask == 1)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        bbox = (x1, y1, x2, y2)
        
        
        corners = {
            "TL": tuple(quad[0].astype(int)),
            "TR": tuple(quad[1].astype(int)),
            "BR": tuple(quad[2].astype(int)),
            "BL": tuple(quad[3].astype(int)),
        }
        
        
        self._save_detection_results(image, corners, bbox, conf_th, mask)
        
        return corners, bbox
    
    def detect_batch(self, images: list[np.ndarray], conf_th: float = 0.3, **kwargs) -> list[Tuple[Dict[str, Tuple[int, int]], Tuple[int, int, int, int]]]:
        """
        Детектирует документы на нескольких изображениях.
        
        Args:
            images: список BGR изображений
            conf_th: порог уверенности для детекции
            
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