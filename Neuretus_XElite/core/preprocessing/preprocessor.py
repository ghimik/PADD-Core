import os
import cv2
import numpy as np
from PIL import Image
from typing import Union, Optional, List, Dict, Any, Tuple
import uuid


class Preprocessor:
    def __init__(self, output_dir: str):
        self.output_dir = os.path.join(output_dir, "preprocessing")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.default_config = {
            # geometry
            "deskew": True,
            "deskew_max_angle": 15,

            "remove_borders": True,
            "border_margin": 5,

            # denoise
            "denoise": True,
            "denoise_strength": 7,

            # contrast
            "contrast_enhance": True,
            "contrast_clip_limit": 2.5,
            "contrast_grid_size": (8, 8),

            # grayscale
            "grayscale": True,

            # binarization (Sauvola-like)
            "threshold": True,
            "threshold_method": "sauvola",
            "threshold_block_size": 31,
            "threshold_k": 0.34,

            # morphology
            "morphology": True,
            "morph_kernel": 2,

            # sharpen
            "sharpen": True,

            # DPI normalization
            "resize": True,
            "resize_max_size": 3000,

            # normalize
            "normalize": True
        }

    def _sauvola(self, img, window=31, k=0.34):
        img = img.astype(np.float32)
        mean = cv2.boxFilter(img, -1, (window, window))
        sqmean = cv2.boxFilter(img**2, -1, (window, window))
        std = np.sqrt(sqmean - mean**2)

        R = 128
        thresh = mean * (1 + k * ((std / R) - 1))
        return (img > thresh).astype(np.uint8) * 255


    def _morphology_cleanup(self, img, k=2):
        kernel = np.ones((k, k), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        return img


    def _sharpen(self, img):
        kernel = np.array([[0, -1, 0],
                        [-1, 5,-1],
                        [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)
    
    def preprocess(self, image: Union[str, np.ndarray, Image.Image], 
                   config: Optional[Dict[str, Any]] = None,
                   doc_id: Optional[str] = None) -> np.ndarray:
        
        if doc_id is None:
            doc_id = str(uuid.uuid4())[:8]
        
        if config is None:
            config = self.default_config.copy()
        
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"cannot read image: {image}")
        elif isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image.copy()
        
        original = img.copy()
        
        if config.get("resize") and config.get("resize_max_size"):
            h, w = img.shape[:2]
            if max(h, w) > config["resize_max_size"]:
                scale = config["resize_max_size"] / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h))
        
        if config.get("grayscale") and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if config.get("denoise"):
            if len(img.shape) == 3:
                img = cv2.fastNlMeansDenoisingColored(img, None, 
                                                      config["denoise_strength"],
                                                      config["denoise_strength"], 7, 21)
            else:
                img = cv2.fastNlMeansDenoising(img, None, 
                                               config["denoise_strength"], 7, 21)
        
        if config.get("contrast_enhance"):
            if len(img.shape) == 3:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=config["contrast_clip_limit"],
                                         tileGridSize=config["contrast_grid_size"])
                l = clahe.apply(l)
                lab = cv2.merge([l, a, b])
                img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=config["contrast_clip_limit"],
                                         tileGridSize=config["contrast_grid_size"])
                img = clahe.apply(img)
        
        if config.get("deskew"):
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            coords = np.column_stack(np.where(binary > 0))
            
            if len(coords) > 0:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = 90 + angle
                if abs(angle) > config["deskew_max_angle"]:
                    angle = 0
                
                if abs(angle) > 0.5:
                    h, w = img.shape[:2]
                    center = (w // 2, h // 2)
                    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    img = cv2.warpAffine(img, matrix, (w, h),
                                          flags=cv2.INTER_CUBIC,
                                          borderMode=cv2.BORDER_REPLICATE)
        
        if config.get("threshold") and len(img.shape) == 2:
            if config["threshold_method"] == "adaptive":
                img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY,
                                            config["threshold_block_size"],
                                            config["threshold_c"])
            else:
                _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if config.get("remove_borders"):
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            coords = cv2.findNonZero(binary)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                margin = config["border_margin"]
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(gray.shape[1] - x, w + 2 * margin)
                h = min(gray.shape[0] - y, h + 2 * margin)
                img = img[y:y+h, x:x+w]
        
        if config.get("normalize"):
            if img.dtype != np.uint8:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # grayscale (после CLAHE)
        if config.get("grayscale") and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Sauvola
        if config.get("threshold"):
            if config["threshold_method"] == "sauvola":
                img = self._sauvola(img,
                                    config["threshold_block_size"],
                                    config["threshold_k"])
            else:
                img = cv2.adaptiveThreshold(img, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY,
                                            config["threshold_block_size"], 5)

        # morphology
        if config.get("morphology"):
            img = self._morphology_cleanup(img, config["morph_kernel"])

        # sharpen
        if config.get("sharpen"):
            img = self._sharpen(img)
        
        step_dir = os.path.join(self.output_dir, doc_id)
        os.makedirs(step_dir, exist_ok=True)
        
        cv2.imwrite(os.path.join(step_dir, "original.jpg"), original)
        cv2.imwrite(os.path.join(step_dir, "preprocessed.jpg"), img)
        
        return img
    
    def get_default_config(self) -> Dict[str, Any]:
        return self.default_config.copy()
    
    def update_default_config(self, updates: Dict[str, Any]):
        self.default_config.update(updates)