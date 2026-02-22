import numpy as np
import cv2
from typing import Tuple, Dict

LABELS = ["TL", "TR", "BR", "BL"]


def to_bl(patch: np.ndarray, label: str) -> np.ndarray:
    """Приводит патч к ориентации Bottom-Left (на которой обучена Corner Bane)"""
    if label == "BL":
        return patch
    if label == "BR":
        return cv2.flip(patch, 1)
    if label == "TL":
        return cv2.flip(patch, 0)
    if label == "TR":
        return cv2.flip(cv2.flip(patch, 1), 0)
    raise ValueError(f"Invalid label: {label}")


def from_bl(x: int, y: int, label: str, size: int = 256) -> Tuple[int, int]:
    """Переводит координаты из BL-ориентации обратно в исходную"""
    if label == "BL":
        return x, y
    if label == "BR":
        return size - 1 - x, y
    if label == "TL":
        return x, size - 1 - y
    if label == "TR":
        return size - 1 - x, size - 1 - y
    raise ValueError(f"Invalid label: {label}")


def order_points(pts: np.ndarray) -> np.ndarray:
    """Упорядочивает 4 точки в порядке TL, TR, BR, BL"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # TL
    rect[2] = pts[np.argmax(s)]      # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # TR
    rect[3] = pts[np.argmax(diff)]   # BL
    return rect


def reorder_corners(pts: Dict[str, Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
    """Переупорядочивает словарь углов в правильный порядок"""
    arr = np.array(list(pts.values()), dtype=np.float32)
    center = arr.mean(axis=0)
    angles = np.arctan2(arr[:, 1] - center[1], arr[:, 0] - center[0])
    arr = arr[np.argsort(angles)]

    s = arr.sum(axis=1)
    diff = np.diff(arr, axis=1).squeeze()

    return {
        "TL": tuple(arr[np.argmin(s)].astype(int)),
        "TR": tuple(arr[np.argmin(diff)].astype(int)),
        "BR": tuple(arr[np.argmax(s)].astype(int)),
        "BL": tuple(arr[np.argmax(diff)].astype(int)),
    }


def mask_to_quad(mask: np.ndarray) -> np.ndarray:
    """Преобразует бинарную маску в четырёхугольник"""
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    if len(approx) == 4:
        quad = approx.reshape(4, 2)
    else:
        rect = cv2.minAreaRect(cnt)
        quad = cv2.boxPoints(rect)

    return order_points(quad)
