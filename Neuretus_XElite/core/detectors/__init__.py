from .base import DocumentDetector
from .malboro import MalboroDetector
from .computantis import ComputantisDetector
from .corner_bane import CornerBaneRefiner
from .utils import LABELS

__all__ = [
    'DocumentDetector',
    'MalboroDetector', 
    'ComputantisDetector',
    'CornerBaneRefiner',
    'LABELS',
]