# PADD-Core

**Perspective-Aware Document Detector (core)**

Developed as part of a Bachelor's thesis on document digitization and perspective correction.

## Overview

PADD-Core is the central machine learning component of the PADD ecosystem. It contains all trained models and inference logic for document detection, corner refinement, and OCR processing. The library is designed to be used either directly or through the PADD-Backend API service.

## Features

- **Document Detection**: YOLO-based models (seg, pose) for document localization
- **Subpixel Corner Refinement**: CNN-based refiner achieving median error of 2.83px
- **Perspective Correction**: Homography transformation and A4 scaling
- **OCR Pipeline**: PaddleOCR-VL integration with automatic rotation detection
- **PDF Reconstruction**: Layout-aware PDF generation with table and image support

## Architecture

```
Neuretus_XElite/
├── api/               # Public interface
│   ├── elite.py       # Main NeuretusXElite class
│   └── processed_document.py
├── core/              # Internal modules
│   ├── detectors/     # Detection models (Malboro, Computantis, Corner Bane)
│   ├── geometry/      # Homography, scaling
│   ├── ocr/           # OCR and rotation detection
│   ├── pdfyer/        # PDF generation engine
│   └── preprocessing/ # Image preprocessing
└── models/            # Pretrained weights
    ├── sychok_bygarety.pt    # Main detector (YOLO)
    ├── computantis.pt        # Fallback detector
    ├── corner_bane.pth       # Corner refiner (CNN)
    └── fonts/                # Fonts for PDF generation
```

## Quick Start

```python
from Neuretus_XElite.api import NeuretusXElite

# Initialize
elite = NeuretusXElite(
    models_dir="./models",
    output_dir="./results"
)

# Full pipeline from image to PDF
doc = elite.process_full("document.jpg")
doc.save_pdf("output.pdf")

# Or step by step
image = cv2.imread("document.jpg")
corners, bbox = elite.find_corners_and_bbox(image)
refined = elite.refine_corners(image, corners, bbox)
warped = elite.warp_perspective(image, refined)
elite.do_ocr(warped)
```

# Components & Integration

## Core Components

### 1. **Detection Layer**
| Component | Type | Responsibility |
|-----------|------|----------------|
| **Sychok Bygarty** | YOLO-seg | Primary detector, finds document bounding box and rough corners |
| **Neuroretus Computantis** | YOLO-pose | Fallback detector, used when Malboro fails |
| **Corner Bane** | CNN | Subpixel corner refiner |

### 2. **Geometry Layer**
| Component | Responsibility |
|-----------|----------------|
| **HomographyCorrector** | Applies perspective transform based on four corners |
| **DocumentScaler** | Scales warped image to A4 format (portrait/landscape) |

### 3. **OCR Layer**
| Component | Responsibility |
|-----------|----------------|
| **RotationDetector** | PP-LCNet model, detects if image needs 0/90/180/270° rotation |
| **OCRProcessor** | PaddleOCR-VL wrapper, extracts text and layout structure |

### 4. **Output Layer**
| Component | Responsibility |
|-----------|----------------|
| **PDFEngine** | Reconstructs document from OCR JSON, handles tables and images |


## Integration Flow

1. **Image enters** → `RotationDetector` corrects orientation if needed
2. **Primary detection** → `Sychok Bygarty` attempts to find document
3. **Fallback** → If Malboro fails, `Neuroretus Computantis` provides pose estimation
4. **Refinement** → `Corner Bane` improves corner coordinates to subpixel accuracy
5. **Geometry** → `HomographyCorrector` warps perspective, `DocumentScaler` fits to A4
6. **OCR** → `OCRProcessor` extracts text and layout structure
7. **Output** → `PDFEngine` reconstructs document preserving tables, images, and formatting

## API Reference

### `NeuretusXElite` Class

Main entry point for all document processing operations.

#### Initialization

```python
def __init__(
    self,
    models_dir: str,           # Path to directory with model weights
    output_dir: str = "./results"  # Directory for output files
):
    """Initialize the document processing pipeline."""
```

#### Core Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `find_corners_and_bbox(image, doc_id)` | Detects document and returns corners + bounding box | `(corners: Dict[str, Tuple[int,int]], bbox: Tuple[int,int,int,int])` |
| `refine_corners(image, corners, bbox, doc_id)` | Improves corner coordinates using Corner Bane | `Dict[str, Tuple[int,int]]` |
| `warp_perspective(image, corners, doc_id)` | Applies homography transformation | `np.ndarray` (warped image) |
| `scale(image, doc_id)` | Scales to A4 format | `np.ndarray` |
| `do_ocr(image, doc_id)` | Performs OCR and generates PDF | `ProcessedDocument` |
| `process_full(image_path, doc_id)` | End-to-end pipeline: image → PDF | `ProcessedDocument` |

#### Utility Methods

| Method | Description |
|--------|-------------|
| `define_rotation_angle(image, doc_id)` | Returns detected rotation (0/90/180/270) |
| `rotate(image, angle, doc_id)` | Rotates image by specified angle |


### Example Usage

```python
from Neuretus_XElite.api import NeuretusXElite

# Initialize
elite = NeuretusXElite(
    models_dir="./models",
    output_dir="./results"
)

# Option 1: Full pipeline
doc = elite.process_full("invoice.jpg")

# Option 2: Step-by-step with manual correction
import cv2
img = cv2.imread("receipt.jpg")

# Detect
corners, bbox = elite.find_corners_and_bbox(img)

# Manual adjustment if needed
corners["TL"] = [120, 45]  # user correction

# Refine and warp
refined = elite.refine_corners(img, corners, bbox)
warped = elite.warp_perspective(img, refined)

# OCR
doc = elite.do_ocr(warped)
```

# Model Performance

| Model | Task | mAP50 | Precision | Recall | Median Error | ≤3px | ≤5px |
|-------|------|-------|-----------|--------|--------------|------|------|
| **SYCHOK_BYGARETY** | Detection (bbox) | 0.985 | 0.948 | 0.934 | - | - | - |
| **SYCHOK_BYGARETY** | Segmentation (mask) | 0.98 | - | - | - | - | - |
| **Computantis** | Pose estimation (corners as keypoints) | 0.982 | 0.989 | 0.959 | - | - | - |
| **Corner Bane** | Corner refinement | - | - | - | **2.83px** | **57.9%** | 78.6% |

## Requirements

- Python 3.10+
- PyTorch 2.0+
- OpenCV
- PaddleOCR
- See `requirements.txt` for full list
