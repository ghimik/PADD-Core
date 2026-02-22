import torch
import cv2
import numpy as np
import json
import os
from typing import Dict, Tuple, Optional, Union
from datetime import datetime

from .cornflake import CornerHeatmapNet
from .utils import to_bl, from_bl, LABELS


class CornerBaneRefiner:
    """
    Уточнитель углов документа на основе CNN (Corner Bane).
    Принимает грубые углы от любого детектора и возвращает уточнённые.
    Сохраняет промежуточные результаты (патчи, хитмапы) в папку refiner_output.
    """
    
    def __init__(self, model_path: str, output_dir: str, device: Optional[str] = None):
        """
        Args:
            model_path: путь к .pth файлу модели Corner Bane
            output_dir: директория для сохранения результатов
            device: 'cuda', 'mps', 'cpu' или None (автоопределение)
        """
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.backends.cudnn.is_available() else "cpu"
        
        self.device = device
        self.output_dir = output_dir
        
        
        self.refiner_output_dir = os.path.join(output_dir, "refiner_output")
        os.makedirs(self.refiner_output_dir, exist_ok=True)
        
        
        self.model = CornerHeatmapNet(pretrained=False).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
    
    def _save_patch_with_heatmap(self, patch: np.ndarray, heatmap: np.ndarray, 
                                  label: str, coarse_point: Tuple[int, int], 
                                  refined_point: Tuple[int, int], timestamp: str):
        """
        Сохраняет патч, хитмапу и визуализацию для отдельного угла.
        
        Args:
            patch: исходный патч изображения
            heatmap: хитмапа уверенности
            label: метка угла
            coarse_point: грубая координата угла в патче
            refined_point: уточненная координата угла в патче
            timestamp: временная метка
        """
        
        patch_filename = f"patch_{label}_{timestamp}.jpg"
        patch_path = os.path.join(self.refiner_output_dir, patch_filename)
        cv2.imwrite(patch_path, patch)
        
        
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        heatmap_filename = f"heatmap_{label}_{timestamp}.jpg"
        heatmap_path = os.path.join(self.refiner_output_dir, heatmap_filename)
        cv2.imwrite(heatmap_path, heatmap_colored)
        
        
        vis = patch.copy()
        
        
        cv2.circle(vis, coarse_point, 5, (0, 0, 255), -1)
        
        
        cv2.circle(vis, refined_point, 5, (0, 255, 0), -1)
        
        
        cv2.putText(vis, f"{label} - coarse", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(vis, f"{label} - refined", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        vis_filename = f"vis_{label}_{timestamp}.jpg"
        vis_path = os.path.join(self.refiner_output_dir, vis_filename)
        cv2.imwrite(vis_path, vis)
        
        h, w = patch.shape[:2]
        composite = np.zeros((h, w * 3, 3), dtype=np.uint8)
        composite[:, :w] = patch
        composite[:, w:w*2] = heatmap_colored
        composite[:, w*2:] = vis
        
        
        cv2.line(composite, (w, 0), (w, h), (255, 255, 255), 2)
        cv2.line(composite, (w*2, 0), (w*2, h), (255, 255, 255), 2)
        
        
        cv2.putText(composite, "Patch", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(composite, "Heatmap", (w + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(composite, "Visualization", (w*2 + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        composite_filename = f"composite_{label}_{timestamp}.jpg"
        composite_path = os.path.join(self.refiner_output_dir, composite_filename)
        cv2.imwrite(composite_path, composite)
        
        return {
            "patch": patch_filename,
            "heatmap": heatmap_filename,
            "visualization": vis_filename,
            "composite": composite_filename
        }
    
    def _save_refinement_results(self, image: np.ndarray, coarse_corners: Dict[str, Tuple[int, int]],
                                  refined_corners: Dict[str, Tuple[int, int]], 
                                  bbox: Optional[Tuple[int, int, int, int]],
                                  patch_size_info: Union[str, float],
                                  corner_data: Dict[str, Dict]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        vis_full = image.copy()
        
        for label, (x, y) in coarse_corners.items():
            cv2.circle(vis_full, (x, y), 8, (0, 0, 255), 2)
            cv2.putText(vis_full, f"{label}_c", (x + 10, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        for label, (x, y) in refined_corners.items():
            cv2.circle(vis_full, (x, y), 8, (0, 255, 0), 2)
            cv2.putText(vis_full, f"{label}_r", (x + 10, y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(vis_full, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        vis_full_filename = f"refinement_full_{timestamp}.jpg"
        vis_full_path = os.path.join(self.refiner_output_dir, vis_full_filename)
        cv2.imwrite(vis_full_path, vis_full)
        
        
        refinement_data = {
            "timestamp": timestamp,
            "device": self.device,
            "patch_size": str(patch_size_info),
            "bbox": None if bbox is None else {
                "x1": int(bbox[0]), "y1": int(bbox[1]),
                "x2": int(bbox[2]), "y2": int(bbox[3]),
                "width": int(bbox[2] - bbox[0]),
                "height": int(bbox[3] - bbox[1])
            },
            "coarse_corners": {
                label: {"x": int(x), "y": int(y)} 
                for label, (x, y) in coarse_corners.items()
            },
            "refined_corners": {
                label: {"x": int(x), "y": int(y)} 
                for label, (x, y) in refined_corners.items()
            },
            "corrections": {
                label: {
                    "dx": int(refined_corners[label][0] - coarse_corners[label][0]),
                    "dy": int(refined_corners[label][1] - coarse_corners[label][1])
                }
                for label in LABELS
            },
            "full_visualization": vis_full_filename
        }
        
        
        def convert_to_python(obj):
            if isinstance(obj, dict):
                return {k: convert_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_python(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return convert_to_python(obj.tolist())
            else:
                return obj
        
        refinement_data["corner_files"] = convert_to_python(corner_data)
        
        json_filename = f"refinement_results_{timestamp}.json"
        json_path = os.path.join(self.refiner_output_dir, json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(refinement_data, f, indent=2, ensure_ascii=False)
        
        print(f"Результаты уточнения сохранены в: {self.refiner_output_dir}")
        print(f"  JSON: {json_filename}")
        print(f"  Визуализация: {vis_full_filename}")
    
    def get_heatmap(self, patch: np.ndarray, label: str) -> np.ndarray:
        """
        Получает хитмапу уверенности для патча с углом.
        
        Args:
            patch: BGR патч изображения (произвольного размера)
            label: метка угла ("TL", "TR", "BR", "BL")
            
        Returns:
            heatmap: numpy array (256, 256) с распределением уверенности
                    (уже ориентированная в исходную систему координат)
        """
        patch = cv2.resize(patch, (256, 256))
        
        patch_bl = to_bl(patch, label)
        
        
        tensor = torch.from_numpy(
            cv2.cvtColor(patch_bl, cv2.COLOR_BGR2RGB)
        ).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)
        
        
        with torch.no_grad():
            heatmap_bl = self.model(tensor)[0, 0].cpu().numpy()
        
        
        
        h, w = heatmap_bl.shape
        heatmap = np.zeros_like(heatmap_bl)
        
        for y in range(h):
            for x in range(w):
                
                orig_x, orig_y = from_bl(x, y, label, size=256)
                if 0 <= orig_x < w and 0 <= orig_y < h:
                    heatmap[orig_y, orig_x] = heatmap_bl[y, x]
        
        return heatmap
    
    def refine_single(self, patch: np.ndarray, label: str) -> Tuple[int, int]:
        """
        Уточняет один угол на патче.
        
        Args:
            patch: BGR патч изображения (произвольного размера)
            label: метка угла ("TL", "TR", "BR", "BL")
            
        Returns:
            (x, y) координаты угла в системе координат патча (0-255)
        """
        heatmap = self.get_heatmap(patch, label)
        
        
        iy, ix = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        
        return ix, iy
    
    def refine(
        self, 
        image: np.ndarray, 
        coarse_corners: Dict[str, Tuple[int, int]],
        bbox: Optional[Tuple[int, int, int, int]] = None,
        patch_size: Union[str, float] = 'auto',
        save_intermediate: bool = True
    ) -> Dict[str, Tuple[int, int]]:
        """
        Уточняет позиции углов документа.
        
        Args:
            image: BGR изображение
            coarse_corners: словарь грубых углов {"TL": (x,y), ...}
            bbox: bounding box документа (x1, y1, x2, y2) - опционально, 
                  нужен для auto patch_size
            patch_size: 
                - 'auto': автоматический размер патча на основе расстояния 
                          от угла до угла bbox
                - float: доля от меньшей стороны изображения (0.0-1.0)
            save_intermediate: сохранять ли промежуточные результаты
                
        Returns:
            refined_corners: словарь уточнённых углов
        """
        H, W = image.shape[:2]
        refined = {}
        
        
        if patch_size == 'auto' and bbox is None:
            raise ValueError("bbox required for auto patch_size")
        
        
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            bbox_corners = {
                "TL": (x1, y1),
                "TR": (x2, y1),
                "BR": (x2, y2),
                "BL": (x1, y2),
            }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        corner_data = {}
        
        for label in LABELS:
            x, y = coarse_corners[label]
            
            
            if patch_size == 'auto':
                
                dx = x - bbox_corners[label][0]
                dy = y - bbox_corners[label][1]
                patch_px = max(
                    int(np.sqrt(dx*dx + dy*dy) * 2),
                    int(min(H, W) * 0.1)
                )
            else:
                patch_px = int(min(H, W) * patch_size)
            
            half = min(patch_px // 2, x, W - x, y, H - y)
            
            
            x1_p = max(0, x - half)
            y1_p = max(0, y - half)
            x2_p = min(W, x + half)
            y2_p = min(H, y + half)
            
            patch = image[y1_p:y2_p, x1_p:x2_p]
            
            
            heatmap = self.get_heatmap(patch, label)
            patch_x, patch_y = np.unravel_index(np.argmax(heatmap), heatmap.shape)[::-1]
            
            
            px = int(patch_x / 256 * (x2_p - x1_p)) + x1_p
            py = int(patch_y / 256 * (y2_p - y1_p)) + y1_p
            
            refined[label] = (
                int(np.clip(px, 0, W - 1)),
                int(np.clip(py, 0, H - 1))
            )
            
            
            if save_intermediate:
                
                coarse_patch_x = int((x - x1_p) / (x2_p - x1_p) * 256)
                coarse_patch_y = int((y - y1_p) / (y2_p - y1_p) * 256)
                
                
                files = self._save_patch_with_heatmap(
                    cv2.resize(patch, (256, 256)),
                    heatmap,
                    label,
                    (coarse_patch_x, coarse_patch_y),
                    (patch_x, patch_y),
                    f"{timestamp}_{label}"
                )
                
                corner_data[label] = {
                    "files": files,
                    "coarse_image_coords": (x, y),
                    "refined_image_coords": refined[label],
                    "coarse_patch_coords": (coarse_patch_x, coarse_patch_y),
                    "refined_patch_coords": (patch_x, patch_y),
                    "patch_bbox": (x1_p, y1_p, x2_p, y2_p)
                }
        
        
        if save_intermediate:
            self._save_refinement_results(
                image, coarse_corners, refined, bbox, patch_size, corner_data
            )
        
        return refined
    
    def get_output_directory(self) -> str:
        """Возвращает путь к директории, где сохраняются результаты."""
        return self.refiner_output_dir