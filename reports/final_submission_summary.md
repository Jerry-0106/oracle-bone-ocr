# Final Submission Summary

## Best Online Result

| Item | Value |
|---|---|
| **Version** | v13_swin_tiny_e30 |
| **Online Final F1** | **0.495565** |
| Detection F1 | 0.908696 |
| Recognition Accuracy | 0.545358 |
| TP | 4106 |
| FP | 4129 |
| FN | 4230 |
| Precision | 0.498604 |
| Recall | 0.492562 |

## Local 931 Corrected Evaluation

| Item | Value |
|---|---|
| Corrected F1 | 0.7141 |
| Rec Acc | 0.8237 |
| TP | 5994 |
| FP | 2981 |
| FN | 1819 |
| Wrong text | 1283 |

## Model Configuration

### Detector
- Model: YOLO11s @ 1280px
- conf = 0.20
- iou = 0.25
- max_det = 300
- augment = True

### Recognizer
- Model: Swin-Tiny + CE (Cross-Entropy loss, label_smoothing=0.05)
- Backbone: `swin_tiny_patch4_window7_224` (28.3M params)
- Input: 224×224
- Transform: Resize(256) → CenterCrop(224) → Normalize(ImageNet stats)
- num_classes: 3483

### Crop
- pad = max(8, int(0.08 * max(w, h)))
- Output bbox: original YOLO bbox (no shrink)

## Checkpoint

```
checkpoints/swin_tiny_e30_v13_candidate.pt
```
- Size: 115 MB
- Epoch: 30
- orig_top1: 0.7961
- pipe_top1: 0.8673

## Docker Image

```
crpi-uqvecxl2rg3udb37.cn-hangzhou.personal.cr.aliyuncs.com/case_1/wjr_ai_orc:v13_swin_tiny_e30_candidate
```

Local tag: `orc-ocr:v13_swin_tiny_e30_candidate`

## Git

| Item | Value |
|---|---|
| Branch | `candidate/v13_swin_tiny_e30` |
| Commit | `d934e61` |
| Tag | `v13_swin_tiny_e30_candidate` |

## Key Components

| File | Purpose |
|---|---|
| `scripts/infer.py` | Full detection + recognition pipeline |
| `src/recognizer.py` | SwinRecognizer class |
| `run.sh` | Container entrypoint |
| `Dockerfile` | Build with timm, torch 2.5.1 |
| `mappings/idx_to_class.json` | 3483-class index → char mapping |
| `checkpoints/yolo11s_det_1280.pt` | YOLO11s detector |
| `checkpoints/swin_tiny_e30_v13_candidate.pt` | Swin-Tiny E30 recognizer |
