import json
import os
from typing import Union, Dict, Any, Optional
from pathlib import Path

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from .block_drawers import get_block_drawer


class PDFEngine:
    """
    Движок для реконструкции PDF из JSON-результатов OCR.
    """
    
    def __init__(self, font_path: str, font_name: str = "CustomFont"):
        """
        Args:
            font_path: путь к TTF файлу шрифта
            font_name: имя для регистрации шрифта
        """
        self.font_path = font_path
        self.font_name = font_name
        
        
        pdfmetrics.registerFont(TTFont(font_name, font_path))
    
    def reconstruct(self, json_path: Union[str, Path], output_pdf: Union[str, Path], 
                   image_dir: Optional[str] = None) -> None:
        """
        Реконструирует PDF из JSON файла.
        
        Args:
            json_path: путь к JSON с результатами OCR
            output_pdf: путь для сохранения PDF
            image_dir: директория с изображениями (если есть)
        """
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        
        page_w = data.get("width", 595)  # A4 width по умолчанию
        page_h = data.get("height", 842)  # A4 height по умолчанию
        

        c = canvas.Canvas(str(output_pdf), pagesize=(page_w, page_h))
        
        
        for block in data.get("parsing_res_list", []):
            block_type = block.get("block_label", "")
            
            
            drawer = get_block_drawer(block_type, self.font_name, image_dir)
            
            
            drawer.draw(c, block, page_h)
        
        c.save()
        print(f"PDF saved to {output_pdf}")
    
    def reconstruct_from_dict(self, data: Dict[str, Any], output_pdf: Union[str, Path],
                             image_dir: Optional[str] = None) -> None:
        """
        Реконструирует PDF из словаря с данными.
        
        Args:
            data: словарь с данными (как из JSON)
            output_pdf: путь для сохранения PDF
            image_dir: директория с изображениями (если есть)
        """
        
        page_w = data.get("width", 595)
        page_h = data.get("height", 842)
        
        
        c = canvas.Canvas(str(output_pdf), pagesize=(page_w, page_h))
        
        
        for block in data.get("parsing_res_list", []):
            block_type = block.get("block_label", "")
            
            
            drawer = get_block_drawer(block_type, self.font_name, image_dir)
            
            
            drawer.draw(c, block, page_h)
        
        c.save()
        print(f"PDF saved to {output_pdf}")