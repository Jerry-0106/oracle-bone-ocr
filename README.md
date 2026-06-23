# Oracle Bone Character OCR

Two-stage OCR pipeline for detecting and recognizing ancient Chinese oracle bone characters (甲骨文). Competition-grade system with YOLO detection + Swin Transformer recognition, Dockerized for deployment.

## Final Results (v13)

| Metric | Online |
|---|---|
| **Final F1** | **0.495565** |
| Detection F1 | 0.908696 |
| Recognition Accuracy | 0.545358 |
| TP / FP / FN | 4106 / 4129 / 4230 |
| Precision / Recall | 0.4986 / 0.4926 |

## Pipeline

```
Input Image → YOLO11s Detector @1280px → Dynamic Crop Padding → Swin-Tiny Recognizer @224px → prediction.json
```

### Detector
- **Model**: YOLO11s trained at 1280×1280 (mAP50=0.84)
- **Config**: conf=0.20, iou=0.25, max_det=300, augment=True
- **Checkpoint**: `checkpoints/yolo11s_det_1280.pt`

### Recognizer
- **Model**: Swin-Tiny (`swin_tiny_patch4_window7_224`, 28.3M params)
- **Loss**: Cross-Entropy with label_smoothing=0.05
- **Classes**: 3,483 oracle bone characters
- **Preprocessing**: Resize(256) → CenterCrop(224) → ImageNet normalize
- **Checkpoint**: `checkpoints/swin_tiny_e30_v13_candidate.pt`

### Crop
- pad = max(8, int(0.08 × max(w, h)))
- Output bbox = original YOLO bbox (no shrink)

## Key Improvements Over Baseline

| Change | Online F1 Gain |
|---|---|
| ConvNeXt+ArcFace → Swin-Tiny+CE | +0.0868 |
| E20 → E25 extended training | +0.0072 |
| E25 → E30 continued training | +0.0007 |
| **Total (v9 → v13)** | **+0.0947** |

## Project Structure

```
src/                          # Core inference modules
  recognizer.py               # SwinRecognizer + ArcFaceRecognizer
scripts/
  infer.py                    # Full detection + recognition pipeline
configs/                      # Model configs (baseline)
mappings/
  idx_to_class.json           # 3483-class index → character mapping
reports/
  final_submission_summary.md # Final v13 submission details
  online_submissions.md       # v8→v13 submission history
  experiment_retrospective.md # What worked, what didn't
  swin_tiny_training_summary.md # Training curves E20→E35
Dockerfile                    # Docker build (Python 3.10, PyTorch 2.5.1, timm)
run.sh                        # Container entrypoint
requirements.txt              # Python dependencies
evaluate.py                   # Official competition evaluator
```

## Reproducing Inference

### Docker (Recommended)

```bash
# Build
docker build -t orc-ocr:v13 .

# Run — mounts /saisdata (input images) and /saisresult (output)
docker run --rm --gpus all \
  -v /path/to/images:/saisdata \
  -v /path/to/output:/saisresult \
  orc-ocr:v13
```

Output: `/saisresult/prediction.json` — `{"image_id": [{"bbox": [x,y,w,h], "text": "char"}, ...], ...}`

### Local

```bash
pip install -r requirements.txt
export INPUT_DIR=/path/to/images OUTPUT_DIR=/path/to/output
python scripts/infer.py
```

## Requirements

- Python 3.10+
- PyTorch 2.5.1+ with CUDA 12.4
- timm >= 1.0.0
- ultralytics >= 8.0.0
- opencv-python-headless, numpy, Pillow, tqdm

Full list in `requirements.txt`.

## Important Notes

- **Model weights are NOT included** in this repository. Download separately:
  - `checkpoints/yolo11s_det_1280.pt` — YOLO11s detector
  - `checkpoints/swin_tiny_e30_v13_candidate.pt` — Swin-Tiny E30 recognizer (116 MB)
- **Datasets are NOT included**. Training data is from the competition organizers.
- Docker images are hosted on Alibaba Cloud Container Registry (not GitHub).
- Checkpoints must be placed in `checkpoints/` before building Docker or running inference.

## Online Submission Tags

| Tag | Version | Online F1 |
|---|---|---|
| `v9_candidate_conf020_iou025` | ConvNeXt baseline | 0.400821 |
| `v11_swin_tiny_e20_candidate` | Swin-Tiny E20 | 0.487599 |
| `v12_swin_tiny_e25_timm_fix` | Swin-Tiny E25 | 0.494840 |
| `v13_final_online_best` | **Swin-Tiny E30 (final)** | **0.495565** |

## License

This project is for academic and competition use. Checkpoint weights and datasets may have separate licenses from their original sources.
