from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, Optional
import numpy as np


class DocumentDetector(ABC):
    """Базовый интерфейс для всех детекторов документов"""
    
    @abstractmethod
    def detect(self, image: np.ndarray, **kwargs) -> Tuple[Dict[str, Tuple[int, int]], Tuple[int, int, int, int]]:
        """
        Детектирует документ на изображении.
        
        Args:
            image: BGR изображение (numpy array)
            **kwargs: параметры детекции (conf_th, etc)
            
        Returns:
            corners: словарь {"TL": (x,y), "TR": (x,y), "BR": (x,y), "BL": (x,y)}
            bbox: кортеж (x1, y1, x2, y2)
            
        Raises:
            RuntimeError: если документ не найден
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Имя детектора"""
        pass