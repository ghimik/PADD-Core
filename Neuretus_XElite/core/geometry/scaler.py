import cv2
import numpy as np
import os
from typing import Tuple, Optional, Literal
from datetime import datetime


class DocumentScaler:
    """
    Масштабирование документа до заданных пропорций.
    По умолчанию использует пропорции A4 (210×297 мм).
    Сохраняет входное и выходное изображения в папку scale_output.
    """
    
    # A4 в дюймах (ширина, высота)
    A4_INCHES_LANDSCAPE = (11.69, 8.27)
    A4_INCHES_PORTRAIT = (8.27, 11.69)
    
    def __init__(self, output_dir: str, target_width_inches: Optional[float] = None, 
                 target_height_inches: Optional[float] = None):
        """
        Args:
            output_dir: директория для сохранения результатов
            target_width_inches: целевая ширина в дюймах (если None, используется A4)
            target_height_inches: целевая высота в дюймах (если None, используется A4)
        """
        self.output_dir = output_dir
        self.target_width_inches = target_width_inches
        self.target_height_inches = target_height_inches
        
        
        self.scale_output_dir = os.path.join(output_dir, "scale_output")
        os.makedirs(self.scale_output_dir, exist_ok=True)
    
    def _save_images(self, input_image: np.ndarray, output_image: np.ndarray, suffix: str = ""):
        """
        Сохраняет входное и выходное изображения в папку scale_output.
        
        Args:
            input_image: исходное изображение
            output_image: обработанное изображение
            suffix: суффикс для имени файла (например, метод масштабирования)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        
        if suffix:
            input_filename = f"input_{suffix}_{timestamp}.jpg"
            output_filename = f"output_{suffix}_{timestamp}.jpg"
        else:
            input_filename = f"input_{timestamp}.jpg"
            output_filename = f"output_{timestamp}.jpg"
        
        
        input_path = os.path.join(self.scale_output_dir, input_filename)
        output_path = os.path.join(self.scale_output_dir, output_filename)
        
        
        cv2.imwrite(input_path, input_image)
        cv2.imwrite(output_path, output_image)
        
        print(f"Изображения сохранены в: {self.scale_output_dir}")
        print(f"  Входное: {input_filename}")
        print(f"  Выходное: {output_filename}")
    
    def scale_to_a4(self, image: np.ndarray, dpi: int = 300, 
                   orientation: Literal['auto', 'portrait', 'landscape'] = 'auto') -> np.ndarray:
        """
        Масштабирует изображение до размера A4 при заданном DPI.
        
        Args:
            image: BGR изображение
            dpi: точек на дюйм (стандарт для печати - 300)
            orientation: 
                - 'auto': определяется из соотношения сторон
                - 'portrait': принудительно портретная
                - 'landscape': принудительно альбомная
                
        Returns:
            Отмасштабированное изображение
        """
        h, w = image.shape[:2]
        
        
        if orientation == 'auto':
            is_landscape = w >= h
        else:
            is_landscape = orientation == 'landscape'
        
        if is_landscape:
            width_inches, height_inches = self.A4_INCHES_LANDSCAPE
        else:
            width_inches, height_inches = self.A4_INCHES_PORTRAIT
        
        
        if self.target_width_inches is not None:
            width_inches = self.target_width_inches
        if self.target_height_inches is not None:
            height_inches = self.target_height_inches
        
        
        target_w = int(width_inches * dpi)
        target_h = int(height_inches * dpi)
        
        
        scaled_image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        
        
        self._save_images(image, scaled_image, suffix=f"a4_{orientation}")
        
        return scaled_image
    
    def scale_to_fixed_size(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Масштабирует до фиксированного размера в пикселях.
        
        Args:
            image: BGR изображение
            width: целевая ширина
            height: целевая высота
            
        Returns:
            Отмасштабированное изображение
        """
        scaled_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        
        
        self._save_images(image, scaled_image, suffix=f"fixed_{width}x{height}")
        
        return scaled_image
    
    def scale_by_factor(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Масштабирует изображение на заданный коэффициент.
        
        Args:
            image: BGR изображение
            factor: коэффициент масштабирования (>0)
            
        Returns:
            Отмасштабированное изображение
        """
        h, w = image.shape[:2]
        new_w = int(w * factor)
        new_h = int(h * factor)
        
        scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        
        self._save_images(image, scaled_image, suffix=f"factor_{factor:.2f}")
        
        return scaled_image
    
    def scale_to_max_size(self, image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
        """
        Масштабирует изображение так, чтобы оно вписывалось в максимальные размеры,
        сохраняя пропорции.
        
        Args:
            image: BGR изображение
            max_width: максимальная ширина
            max_height: максимальная высота
            
        Returns:
            Отмасштабированное изображение
        """
        h, w = image.shape[:2]
        
        
        scale_w = max_width / w
        scale_h = max_height / h
        
        
        scale = min(scale_w, scale_h)
        
        if scale >= 1.0:
            
            self._save_images(image, image.copy(), suffix=f"maxsize_{max_width}x{max_height}_noresize")
            return image
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        
        self._save_images(image, scaled_image, suffix=f"maxsize_{max_width}x{max_height}")
        
        return scaled_image
    
    def get_output_directory(self) -> str:
        """Возвращает путь к директории, где сохраняются изображения."""
        return self.scale_output_dir