import numpy as np
import cv2
import os
from typing import Dict, Tuple, Optional
from datetime import datetime


class HomographyCorrector:
    """
    Корректор перспективы документа на основе четырех углов.
    Выпрямляет документ, сохраняя пропорции сторон.
    Сохраняет входное и выходное изображения в папку homography_output.
    """
    
    def __init__(self, output_dir: str, target_size: Optional[Tuple[int, int]] = None):
        """
        Args:
            output_dir: директория для сохранения результатов
            target_size: опциональный целевой размер (ширина, высота) после выпрямления.
                        Если None, размер определяется автоматически из исходных углов.
        """
        self.output_dir = output_dir
        self.target_size = target_size
        
        
        self.homography_output_dir = os.path.join(output_dir, "homography_output")
        os.makedirs(self.homography_output_dir, exist_ok=True)
    
    def _save_images(self, input_image: np.ndarray, output_image: np.ndarray, suffix: str = ""):
        """
        Сохраняет входное и выходное изображения в папку homography_output.
        
        Args:
            input_image: исходное изображение
            output_image: обработанное изображение
            suffix: суффикс для имени файла (например, информация о коррекции)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        
        if suffix:
            input_filename = f"input_homography_{suffix}_{timestamp}.jpg"
            output_filename = f"output_homography_{suffix}_{timestamp}.jpg"
        else:
            input_filename = f"input_homography_{timestamp}.jpg"
            output_filename = f"output_homography_{timestamp}.jpg"
        
        
        input_path = os.path.join(self.homography_output_dir, input_filename)
        output_path = os.path.join(self.homography_output_dir, output_filename)
        
        
        cv2.imwrite(input_path, input_image)
        cv2.imwrite(output_path, output_image)
        
        print(f"Изображения после гомографии сохранены в: {self.homography_output_dir}")
        print(f"  Входное: {input_filename}")
        print(f"  Выходное: {output_filename}")
    
    def _format_corners_for_suffix(self, corners: Dict[str, Tuple[int, int]]) -> str:
        """
        Форматирует информацию об углах для использования в суффиксе.
        
        Args:
            corners: словарь углов
            
        Returns:
            Строка с информацией об углах
        """
        try:
            
            tl = corners["TL"]
            tr = corners["TR"]
            br = corners["BR"]
            bl = corners["BL"]
            
            
            return f"tl_{tl[0]}_{tl[1]}_tr_{tr[0]}_{tr[1]}_br_{br[0]}_{br[1]}_bl_{bl[0]}_{bl[1]}"
        except:
            return "corners"
    
    def correct(self, image: np.ndarray, corners: Dict[str, Tuple[int, int]]) -> np.ndarray:
        """
        Выпрямляет документ по углам.
        
        Args:
            image: BGR изображение
            corners: словарь углов {"TL": (x,y), "TR": (x,y), "BR": (x,y), "BL": (x,y)}
            
        Returns:
            Выпрямленное изображение
        """
        
        pts_src = np.array([
            corners["TL"],
            corners["TR"],
            corners["BR"],
            corners["BL"]
        ], dtype=np.float32)
        
        
        if self.target_size is None:
            
            w1 = np.linalg.norm(pts_src[0] - pts_src[1])  # TL -> TR
            w2 = np.linalg.norm(pts_src[3] - pts_src[2])  # BL -> BR
            h1 = np.linalg.norm(pts_src[0] - pts_src[3])  # TL -> BL
            h2 = np.linalg.norm(pts_src[1] - pts_src[2])  # TR -> BR
            
            W = int(max(w1, w2))
            H = int(max(h1, h2))
            size_info = f"auto_{W}x{H}"
        else:
            W, H = self.target_size
            size_info = f"fixed_{W}x{H}"
        
        
        pts_dst = np.array([
            [0, 0],
            [W - 1, 0],
            [W - 1, H - 1],
            [0, H - 1]
        ], dtype=np.float32)
        
        
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped_image = cv2.warpPerspective(image, M, (W, H))
        
        
        corners_suffix = self._format_corners_for_suffix(corners)
        
        if len(corners_suffix) > 100:
            corners_suffix = corners_suffix[:100]
        
        suffix = f"{size_info}_{corners_suffix}"
        
        
        self._save_images(image, warped_image, suffix=suffix)
        
        return warped_image
    
    def correct_with_bbox(self, image: np.ndarray, corners: Dict[str, Tuple[int, int]], 
                          bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Выпрямляет документ и обрезает по bounding box.
        Полезно, если углы немного выходят за пределы документа.
        
        Args:
            image: BGR изображение
            corners: словарь углов
            bbox: (x1, y1, x2, y2) - bounding box для обрезки после выпрямления
            
        Returns:
            Выпрямленное и обрезанное изображение
        """
        
        warped = self.correct(image, corners)
        
        
        x1, y1, x2, y2 = bbox
        
        
        h, w = warped.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        cropped_image = warped[y1:y2, x1:x2]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cropped_filename = f"output_homography_cropped_{timestamp}.jpg"
        cropped_path = os.path.join(self.homography_output_dir, cropped_filename)
        cv2.imwrite(cropped_path, cropped_image)
        print(f"  Обрезанная версия: {cropped_filename}")
        
        return cropped_image
    
    def get_output_directory(self) -> str:
        """Возвращает путь к директории, где сохраняются изображения."""
        return self.homography_output_dir