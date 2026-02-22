import os
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from paddleocr import PaddleOCRVL
from typing import Union

class OCRProcessor:
    def __init__(self, output_dir: str):
        self.pipeline = PaddleOCRVL()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def recognize(self, image: Union[str, Path, np.ndarray, Image.Image]) -> str:
        save_dir = os.path.join(self.output_dir, "ocr_output")
        os.makedirs(save_dir, exist_ok=True)
        
        json_path = os.path.join(save_dir, "result.json")
        md_path = os.path.join(save_dir, "result.md")
        
        output = self.pipeline.predict(image)
        for res in output:
            res.save_to_json(save_path=json_path)
            res.save_to_markdown(save_path=md_path)
            
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            
            self.visualize_blocks(image, data, save_dir)
                    
        return json_path
    
    def visualize_blocks(self, image, data: dict, save_dir: str):
        """Рисует все блоки на изображении"""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        elif isinstance(image, np.ndarray):
            img = image.copy()
        elif isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            return
        
        h, w = img.shape[:2]
        
        
        for block in data.get("parsing_res_list", []):
            label = block.get("block_label", "unknown")
            bbox = block.get("block_bbox")
            if not bbox:
                continue
                
            x1, y1, x2, y2 = map(int, bbox)
            
            
            if "header" in label:
                color = (255, 0, 0)  # синий
            elif "text" in label:
                color = (0, 255, 0)  # зеленый
            elif "table" in label:
                color = (0, 0, 255)  # красный
            elif "seal" in label:
                color = (255, 255, 0)  # желтый
            elif "footer" in label:
                color = (255, 0, 255)  # розовый
            else:
                color = (128, 128, 128)  # серый
            
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            
            cv2.putText(img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        
        vis_path = os.path.join(save_dir, "blocks_visualization.jpg")
        cv2.imwrite(vis_path, img)
        print(f"Визуализация сохранена: {vis_path}")