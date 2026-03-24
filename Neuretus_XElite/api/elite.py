import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Optional, Tuple, Dict
import uuid

from ..core.ocr.whitenner import DocumentEnhancer


from .processed_document import ProcessedDocument
from ..core.ocr.recognition import OCRProcessor
from ..core.ocr.rotation import RotationDetector
from ..core.detectors import MalboroDetector, ComputantisDetector, CornerBaneRefiner
from ..core.geometry import HomographyCorrector, DocumentScaler
from ..core.pdfyer import PDFEngine


class NeuretusXElite:
    def __init__(self, models_dir: str, output_dir: str = "./results"):
        self.models_dir = models_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.pdf_engine = PDFEngine(
            font_path=os.path.join(models_dir, "fonts", "DejaVuSans.ttf"),
            font_name="DejaVuSans"
        )
    
    def _get_doc_output_dir(self, doc_id: str) -> str:
        return os.path.join(self.output_dir, doc_id)
    
    def _init_components_for_doc(self, doc_id: str, ignore_ocr: bool = False):
        """Создает компоненты с конкретной выходной директорией для документа"""
        doc_dir = self._get_doc_output_dir(doc_id)
        os.makedirs(doc_dir, exist_ok=True)
        
        return {
            "rotation": RotationDetector(output_dir=doc_dir),
            "malboro": MalboroDetector(
                model_path=os.path.join(self.models_dir, "sychok_bygarety.pt"),
                output_dir=doc_dir
            ),
            "computantis": ComputantisDetector(
                model_path=os.path.join(self.models_dir, "computantis.pt"),
                output_dir=doc_dir
            ),
            "refiner": CornerBaneRefiner(
                model_path=os.path.join(self.models_dir, "corner_bane.pth"),
                output_dir=doc_dir
            ),
            "homography": HomographyCorrector(output_dir=doc_dir),
            "scaler": DocumentScaler(output_dir=doc_dir),
            "ocr": OCRProcessor(output_dir=doc_dir),
            "enhancer": DocumentEnhancer(brightness=1.15, contrast=1.2, 
                                 whitening=0.85, shadow_removal=True, sharpen=True)
        }
    
    def enhance(self, image: np.ndarray, doc_id: Optional[str] = None) -> np.ndarray:
        """Предобработка: улучшение качества документа"""
        if doc_id is None:
            doc_id = str(uuid.uuid4())[:8]
        return self.enhancer.enhance(image)
    
    def define_rotation_angle(self, image: Union[str, np.ndarray, Image.Image], 
                              doc_id: Optional[str] = None) -> int:
        """0) Определяет угол поворота"""
        if doc_id is None:
            doc_id = str(uuid.uuid4())[:8]
        components = self._init_components_for_doc(doc_id)
        return components["rotation"].detect_angle(image)
    
    def rotate(self, image: Union[str, np.ndarray, Image.Image], angle: int,
               doc_id: Optional[str] = None) -> np.ndarray:
        """0.5) Поворачивает изображение на заданный угол"""
        if doc_id is None:
            doc_id = str(uuid.uuid4())[:8]
        
        if isinstance(image, str):
            img = Image.open(image)
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            img = image
        
        rotated = img.rotate(angle, expand=True)
        return cv2.cvtColor(np.array(rotated), cv2.COLOR_RGB2BGR)
    
    def find_bbox(self, image: np.ndarray, doc_id: Optional[str] = None) -> Tuple[int, int, int, int]:
        """1) Находит bounding box документа"""
        if doc_id is None:
            doc_id = str(uuid.uuid4())[:8]
        components = self._init_components_for_doc(doc_id, ignore_ocr=True)
        
        try:
            _, bbox = components["malboro"].detect(image)
        except:
            _, bbox = components["computantis"].detect(image)
        
        return bbox
    
    def find_corners(self, image: np.ndarray, doc_id: Optional[str] = None) -> Dict[str, Tuple[int, int]]:
        """2) Находит углы документа"""
        if doc_id is None:
            doc_id = str(uuid.uuid4())[:8]
        components = self._init_components_for_doc(doc_id)
        
        try:
            corners, _ = components["malboro"].detect(image, ignore_ocr=True)
        except:
            corners, _ = components["computantis"].detect(image)
        
        return corners
    

    def find_corners_and_bbox(self, image: np.ndarray, doc_id: Optional[str] = None) -> Tuple[Dict[str, Tuple[int, int]], Tuple[int, int, int, int]]:
        """2Б) Находит bbox и углы документа"""
        if doc_id is None:
            doc_id = str(uuid.uuid4())[:8]
        components = self._init_components_for_doc(doc_id, ignore_ocr=True)
        
        try:
            corners, bbox = components["malboro"].detect(image)
        except:
            corners, bbox = components["computantis"].detect(image)
        
        return corners, bbox
    
    def refine_corners(self, image: np.ndarray, corners: Dict[str, Tuple[int, int]], 
                       bbox: Optional[Tuple[int, int, int, int]] = None,
                       doc_id: Optional[str] = None) -> Dict[str, Tuple[int, int]]:
        """3) Уточняет углы"""
        if doc_id is None:
            doc_id = str(uuid.uuid4())[:8]
        components = self._init_components_for_doc(doc_id, ignore_ocr=True)
        
        return components["refiner"].refine(image, corners, bbox)
    
    def warp_perspective(self, image: np.ndarray, corners: Dict[str, Tuple[int, int]],
                         doc_id: Optional[str] = None) -> np.ndarray:
        """4) Исправляет перспективу"""
        if doc_id is None:
            doc_id = str(uuid.uuid4())[:8]
        components = self._init_components_for_doc(doc_id, ignore_ocr=True)
        
        return components["homography"].correct(image, corners)
    
    def scale(self, image: np.ndarray, doc_id: Optional[str] = None) -> np.ndarray:
        """5) Масштабирует под A4"""
        if doc_id is None:
            doc_id = str(uuid.uuid4())[:8]
        components = self._init_components_for_doc(doc_id)
        
        return components["scaler"].scale_to_a4(image, orientation="portrait")
    
    def do_ocr(self, image: np.ndarray, doc_id: Optional[str] = None) -> ProcessedDocument:
        """6) Делает OCR, возвращает объект документа"""
        if doc_id is None:
            doc_id = str(uuid.uuid4())[:8]
        components = self._init_components_for_doc(doc_id)
        
        components["ocr"].recognize(image)
        return ProcessedDocument(doc_id, self.output_dir)
    
    def process_full(self, image_path: str, doc_id: Optional[str] = None) -> ProcessedDocument:
        """Полный пайплайн от картинки до PDF"""
        if doc_id is None:
            doc_id = str(uuid.uuid4())[:8]
        components = self._init_components_for_doc(doc_id)
        
        # Загружаем
        image = cv2.imread(image_path)
        
        # 0-0.5 Поворот
        rotated_pil = components["rotation"].rotate_to_correct(image_path)
        rotated = cv2.cvtColor(np.array(rotated_pil), cv2.COLOR_RGB2BGR)
        
        # 1-2 Детекция
        try:
            corners, bbox = components["malboro"].detect(rotated)
        except:
            try: 
                corners, bbox = components["computantis"].detect(rotated)
            except:
                raise ValueError("Failed to detect document in the image")
        
        # 3 Уточнение
        refined = components["refiner"].refine(rotated, corners, bbox)
        
        # 4 Гомография
        warped = components["homography"].correct(rotated, refined)
        
        # 5 Масштабирование
        scaled = components["scaler"].scale_to_a4(warped, orientation="portrait")
        
        # 6 OCR
        # components["ocr"].recognize(scaled)
        
        # # 7 PDF
        # json_path = os.path.join(self._get_doc_output_dir(doc_id), "ocr_output", "result.json")
        # pdf_path = os.path.join(self._get_doc_output_dir(doc_id), "output.pdf")
        
        # img_dir = os.path.join(self._get_doc_output_dir(doc_id), "ocr_output", "imgs")
        # if os.path.exists(img_dir):
        #     self.pdf_engine.reconstruct(json_path, pdf_path, image_dir=img_dir)
        # else:
        #     self.pdf_engine.reconstruct(json_path, pdf_path)
        
        return ProcessedDocument(doc_id, self.output_dir)