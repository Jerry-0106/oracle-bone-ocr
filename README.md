# Oracle Bone Character OCR

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/YOLOv8-Ultralytics-0b9bcd.svg)](https://docs.ultralytics.com/)

A two-stage OCR system for detecting and recognizing ancient Chinese oracle bone characters (甲骨文).

## Features

- **Two-stage OCR pipeline** — YOLOv8n character detection + EfficientNet-B0 recognition
- **3.1M parameter detector** — lightweight, fast inference at 640×640
- **1,588 character classes** — trained on HUST-OBC dataset
- **TTA & multi-scale inference** — configurable test-time augmentation
- **Competition-ready output** — JSON format with bbox + recognized text
- **Full training pipeline** — XML conversion, YOLO dataset building, augmentation
- **Docker support** — ready for containerized deployment

## Pipeline

```
Input Image (甲骨拓片)
        │
        ▼
┌───────────────────┐
│  YOLOv8n Detector │  ← character-level bounding boxes
│    640×640        │
└────────┬──────────┘
         │  [bbox_1, bbox_2, ..., bbox_n]
         ▼
┌───────────────────┐
│ Character Cropping │  ← padding + shrink to improve IoU
└────────┬──────────┘
         │  [crop_1, crop_2, ..., crop_n]
         ▼
┌───────────────────┐
│  EfficientNet-B0   │  ← 1,588-class classification
│    224×224         │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Recognized Text   │  → {"image_id": [{"bbox": [...], "text": "骨"}]}
└───────────────────┘
```

## Dataset

This project uses the [HUST-OBC](https://github.com/HUST-OBC/HUST-OBC) dataset — oracle bone rubbing images with character-level bounding box annotations.

| Item | Detail |
|------|--------|
| Source | HUST-OBC (Huazhong Univ. of Science & Technology) |
| Classes | 1,588 oracle bone characters |
| Format | Images (.png/.jpg) + XML annotations |
| Download | See [HUST-OBC GitHub](https://github.com/HUST-OBC/HUST-OBC) |

**The dataset is not included in this repository.** Download it and place under `data/`:

```
data/
├── raw/            # original images + XML annotations
├── labels/         # converted YOLO-format labels
└── yolo/           # train/val split ready for training
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml
```

## Project Structure

```
├── src/                      # Core pipeline
│   ├── detector.py           # YOLOv8 detection wrapper
│   ├── recognizer.py         # EfficientNet recognition wrapper
│   ├── models.py             # Model architectures (EfficientNet, ResNet)
│   ├── dataset.py            # PyTorch Dataset classes
│   ├── utils.py              # Image I/O, bbox utilities
│   └── visualize.py          # Detection visualization
├── configs/                  # Inference & training configs
├── scripts/                  # Training / inference / evaluation
│   ├── train_detector_v5.py  # Detector training
│   ├── train_recognition.py  # Recognition training
│   ├── infer.py              # Single-image inference
│   ├── infer_competition.py  # Batch inference (competition format)
│   ├── build_dataset.py      # Build YOLO dataset from XML
│   ├── convert_xml.py        # XML → YOLO label conversion
│   └── evaluate.py           # End-to-end evaluation
├── checkpoints/              # Final model weights
│   ├── detector_v5.pt        # YOLOv8n (6 MB)
│   └── recognizer.pt         # EfficientNet-B0 (70 MB)
├── mappings/                 # Character class index ↔ ID ↔ Chinese
├── demo/                     # Sample images
├── Dockerfile                # Containerized deployment
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/oracle-bone-ocr.git
cd oracle-bone-ocr

pip install -r requirements.txt
```

For GPU training, ensure CUDA-compatible PyTorch is installed. CPU-only inference is fully supported.

## Quick Start

```bash
# Detect and recognize characters in a sample rubbing image
python scripts/infer.py --image demo/sample_rubbing_01.png

# Batch inference on a directory
python scripts/infer_competition.py --source demo/ --output results/
```

## Training

### 1. Prepare the dataset

```bash
# Convert XML annotations to YOLO format
python scripts/convert_xml.py --xml_dir data/raw/ --output data/labels/

# Build train/val split
python scripts/build_dataset.py --mode full
```

### 2. Train the detector

```bash
python scripts/train_detector_v5.py \
    --data data/yolo/data.yaml \
    --epochs 30 \
    --imgsz 640 \
    --device cuda
```

### 3. Train the recognizer

```bash
python scripts/train_recognition.py \
    --data_dir data/recognition/ \
    --epochs 30 \
    --batch_size 64 \
    --device cuda
```

## Inference

```bash
# Single image
python scripts/infer.py --image demo/sample_rubbing_01.png

# Full directory (competition format)
python scripts/infer_competition.py \
    --source data/test_images/ \
    --output submission/ \
    --format json \
    --save-vis
```

Output format:

```json
{
  "image_id": [
    {"bbox": [x, y, w, h], "text": "骨"},
    {"bbox": [x, y, w, h], "text": "文"}
  ]
}
```

## Model Zoo

| Model | Architecture | Params | Size | mAP@50 | Top-1 Acc |
|-------|-------------|--------|------|--------|-----------|
| `detector_v5.pt` | YOLOv8n | 3.1M | 6 MB | 0.477 | — |
| `recognizer.pt` | EfficientNet-B0 | 5.3M | 70 MB | — | competitive |

Pre-trained YOLO backbones auto-download from [Ultralytics](https://docs.ultralytics.com/) on first use.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Detection | Ultralytics YOLOv8 |
| Recognition | PyTorch + torchvision (EfficientNet-B0) |
| Image I/O | OpenCV, Pillow |
| Training HW | Apple M5 (MPS) / NVIDIA RTX 5070 Ti |

## Future Work

- [ ] Character-level recognition with sequence context (CRF / attention)
- [ ] Mixed-precision training for larger batch sizes
- [ ] ONNX export for faster CPU inference
- [ ] Support for more oracle bone rubbing formats
- [ ] Streaming inference for large collections

## License

MIT — see [LICENSE](LICENSE) for details. Dataset licensing follows HUST-OBC terms.
