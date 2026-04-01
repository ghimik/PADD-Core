import cv2
import numpy as np
import os
from typing import Tuple, Optional, Union
from pathlib import Path

class DocumentEnhancer:
    """
    Улучшение качества скана документа.
    Задача: сделать фон светлым/белым, текст контрастным, убрать тени.
    """
    
    def __init__(self, 
                 brightness: float = 1.15,
                 contrast: float = 1.2,
                 whitening: float = 0.85,
                 shadow_removal: bool = True,
                 sharpen: bool = True,
                 output_dir: Optional[str] = None):
        """
        Args:
            brightness: коэффициент яркости
            contrast: коэффициент контраста
            whitening: отбеливание фона (0.7-0.9, 1.0 = без изменений)
            shadow_removal: применять ли удаление теней (CLAHE)
            sharpen: применять ли мягкую резкость
            output_dir: директория для сохранения результатов (опционально)
        """
        self.brightness = brightness
        self.contrast = contrast
        self.whitening = whitening
        self.shadow_removal = shadow_removal
        self.sharpen = sharpen
        self.output_dir = os.path.join(output_dir, "enchaner_output") if output_dir else None
        print(f"DocumentEnhancer: output_dir set to {self.output_dir}")
        
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
    
    def enhance(self, image: np.ndarray, save: bool = True, 
                prefix: str = "enhanced") -> np.ndarray:
        """
        Основной метод: улучшение документа.
        
        Args:
            image: входное изображение (BGR)
            save: сохранять ли результат
            prefix: префикс для имени файла
        
        Returns:
            улучшенное изображение
        """
        if save and self.output_dir:
            input_path = os.path.join(self.output_dir, f"{prefix}_input.jpg")
            print(f"Saving input image to {input_path}")
            cv2.imwrite(input_path, image)
        
        result = image.copy()
        
        print(f"Applying brightness/contrast adjustment: brightness={self.brightness}, contrast={self.contrast}")
        result = self._adjust_brightness_contrast(result)
        
        result = self._whiten_background(result)
        
        if self.shadow_removal:
            print("Applying shadow removal")
            result = self._remove_shadows(result)
        
        if self.sharpen:
            print("Applying sharpening")
            result = self._sharpen_soft(result)
        
        if save and self.output_dir:
            print(f"Saving enhanced image to {self.output_dir}")
            output_path = os.path.join(self.output_dir, f"{prefix}_output.jpg")
            cv2.imwrite(output_path, result)
        
        print("Enhancement completed")
        return result
    
    
    def _adjust_brightness_contrast(self, img: np.ndarray) -> np.ndarray:
        """Мягкая коррекция яркости и контраста"""
        img = cv2.convertScaleAbs(img, alpha=self.contrast, beta=0)
        beta = int(50 * (self.brightness - 1.0))
        img = cv2.convertScaleAbs(img, alpha=1.0, beta=beta)
        return img
    
    def _whiten_background(self, img: np.ndarray) -> np.ndarray:
        """
        Отбеливание фона: светлые участки делаем ещё светлее,
        тёмные (текст) почти не трогаем.
        """
        img_float = img.astype(np.float32) / 255.0
        gamma = self.whitening
        img_gamma = np.power(img_float, gamma)
        img_whitened = (img_gamma * 255).astype(np.uint8)
        return img_whitened
    
    def _remove_shadows(self, img: np.ndarray) -> np.ndarray:
        """Удаление теней через CLAHE (мягкий режим)"""
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return img
    
    def _sharpen_soft(self, img: np.ndarray) -> np.ndarray:
        """Мягкое повышение резкости"""
        kernel = np.array([[0, -0.25, 0],
                           [-0.25, 2, -0.25],
                           [0, -0.25, 0]])
        return cv2.filter2D(img, -1, kernel)
    
    def quick_enhance(self, image: np.ndarray, save: bool = False) -> np.ndarray:
        """Быстрое улучшение с предустановленными параметрами"""
        return self.enhance(image, save=save)

