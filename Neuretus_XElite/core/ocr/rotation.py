import json
import os
from pathlib import Path
from typing import Union, Optional, Tuple
import time
import numpy as np
from PIL import Image
from paddleocr import DocImgOrientationClassification


class RotationDetector:
    """
    Детектор ориентации документа на основе PaddleOCR PP-LCNet.
    Определяет угол поворота (0°, 90°, 180°, 270°), необходимый для приведения
    документа к правильной ориентации для чтения.
    ВСЁ СОХРАНЯЕТСЯ В rotation_output ВНУТРИ output_dir.
    """
    
    
    ANGLE_MAP = {
        "0": 0,
        "90": 90, 
        "180": 180,
        "270": 270
    }
    
    def __init__(self, output_dir: str, model_name: str = "PP-LCNet_x1_0_doc_ori"):
        """
        Args:
            output_dir: основная директория для сохранения результатов
            model_name: название модели ориентации из PaddleOCR
        """
        self.model_name = model_name
        self.detector = DocImgOrientationClassification(model_name=model_name)
        
        
        self.rotation_dir = os.path.join(output_dir, "rotation_output")
        os.makedirs(self.rotation_dir, exist_ok=True)
        print(f"Rotation результаты будут сохраняться в: {self.rotation_dir}")
    
    def _save_input_image(self, image: Union[str, Path, np.ndarray, Image.Image], base_name: str) -> str:
        """
        Сохраняет входное изображение в rotation_output.
        
        Args:
            image: входное изображение
            base_name: базовое имя файла
            
        Returns:
            путь к сохранённому изображению
        """
        input_path = os.path.join(self.rotation_dir, f"{base_name}_input.jpg")
        
        if isinstance(image, (str, Path)):
            
            img = Image.open(str(image))
            img.save(input_path)
        elif isinstance(image, np.ndarray):
            
            img = Image.fromarray(image[..., ::-1])
            img.save(input_path)
        elif isinstance(image, Image.Image):
            image.save(input_path)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        print(f"Входное изображение сохранено: {input_path}")
        return input_path
    
    def _save_output_image(self, image: Image.Image, base_name: str, angle: int) -> str:
        """
        Сохраняет выходное изображение (после поворота) в rotation_output.
        
        Args:
            image: PIL Image после поворота
            base_name: базовое имя файла
            angle: угол поворота
            
        Returns:
            путь к сохранённому изображению
        """
        output_path = os.path.join(self.rotation_dir, f"{base_name}_output_rotated_{angle}.jpg")
        image.save(output_path)
        print(f"Выходное изображение сохранено: {output_path}")
        return output_path
    
    def detect_angle(self, image: Union[str, Path, np.ndarray, Image.Image]) -> int:
        """
        Определяет угол поворота документа.
        Сохраняет входное изображение и JSON с результатом.
        
        Args:
            image: изображение (путь к файлу, numpy array BGR, или PIL Image)
            
        Returns:
            угол поворота в градусах (0, 90, 180, 270)
        """
        
        if isinstance(image, (str, Path)):
            base_name = Path(str(image)).stem
        else:
            base_name = f"image_{int(time.time())}"
        
        
        saved_input_path = self._save_input_image(image, base_name)
        
        
        if isinstance(image, (str, Path)):
            img_path = saved_input_path  
        else:
            
            img_path = saved_input_path
        
        
        print("Определяем ориентацию документа...")
        results = self.detector.predict(input=img_path, batch_size=1)
        
        
        json_path = os.path.join(self.rotation_dir, f"{base_name}_result.json")
        results[0].save_to_json(json_path)
        print(f"JSON результат сохранён: {json_path}")
        
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        label_names = data.get("label_names")
        if not label_names:
            raise RuntimeError("Failed to get orientation label from model")
        
        label = label_names[0]
        angle = self.ANGLE_MAP.get(label, 0)
        
        print(f"Определён угол поворота: {angle}°")
        
        return angle
    
    def rotate_to_correct(self, image: Union[str, Path, np.ndarray, Image.Image]) -> Image.Image:
        """
        Определяет ориентацию и поворачивает изображение для правильного чтения.
        Сохраняет входное изображение, JSON с результатом и выходное изображение.
        
        Args:
            image: входное изображение
            
        Returns:
            PIL Image с правильной ориентацией
        """
        
        if isinstance(image, (str, Path)):
            img = Image.open(image)
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image[..., ::-1])  # BGR -> RGB
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        
        angle = self.detect_angle(image)
        
        
        if angle != 0:
            print(f"Поворачиваем на {angle}°")
            rotated_img = img.rotate(angle, expand=True)
        else:
            rotated_img = img
        
        
        if isinstance(image, (str, Path)):
            base_name = Path(str(image)).stem
        else:
            import time
            base_name = f"image_{int(time.time())}"
        
        
        self._save_output_image(rotated_img, base_name, angle)
        
        return rotated_img