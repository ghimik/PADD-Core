import cv2
import numpy as np
from typing import Tuple, Optional

class DocumentEnhancer:
    """
    Улучшение качества скана документа.
    Задача: сделать фон светлым/белым, текст контрастным, убрать тени.
    """
    
    def __init__(self, 
                 brightness: float = 1.15,
                 contrast: float = 1.2,
                 whitening: float = 0.85,      # новый параметр: отбеливание фона
                 shadow_removal: bool = True,
                 sharpen: bool = True):
        """
        Args:
            brightness: коэффициент яркости
            contrast: коэффициент контраста
            whitening: отбеливание фона (0.7-0.9, 1.0 = без изменений)
            shadow_removal: применять ли удаление теней (CLAHE)
            sharpen: применять ли мягкую резкость
        """
        self.brightness = brightness
        self.contrast = contrast
        self.whitening = whitening
        self.shadow_removal = shadow_removal
        self.sharpen = sharpen
    
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Основной метод: улучшение документа.
        """
        result = image.copy()
        
        # 1. Мягкая коррекция яркости и контраста
        result = self._adjust_brightness_contrast(result)
        
        # 2. Отбеливание фона
        result = self._whiten_background(result)
        
        # 3. Удаление теней (локальное выравнивание)
        if self.shadow_removal:
            result = self._remove_shadows(result)
        
        # 4. Мягкое повышение резкости
        if self.sharpen:
            result = self._sharpen_soft(result)
        
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
        # Переводим в float для точности
        img_float = img.astype(np.float32) / 255.0
        
        # Гамма-коррекция: осветляет светлые участки, почти не трогает тёмные
        gamma = self.whitening
        img_gamma = np.power(img_float, gamma)
        
        # Возвращаем в uint8
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
    
    def quick_enhance(self, image: np.ndarray) -> np.ndarray:
        """Быстрое улучшение с предустановленными параметрами"""
        return self.enhance(image)
