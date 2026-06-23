# Detection Training Report (Phase 2A)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | YOLO11s (Ultralytics) |
| Pretrained | yolo11s.pt (COCO) |
| Parameters | 9,428,179 |
| Input size | 1280×1280 |
| Dataset | Competition rebuilt (5,287 train / 931 val) |
| Total bboxes | 61,075 |
| Epochs | 90 (100 target, early convergence) |
| Batch size | 6 |
| Optimizer | AdamW (lr0=0.001, lrf=0.01) |
| LR schedule | Cosine with 3-epoch warmup |
| Augmentation | mosaic=0.5, mixup=0.1, copy_paste=0.1, degrees=2.0, no flip |
| Device | NVIDIA RTX 5070 Ti Laptop (11.9 GB) |
| Training time | ~90 epochs × ~5.6min ≈ 8.4 hours |

## Results Summary

### Best Epoch: 53

| Metric | Value |
|--------|-------|
| **mAP50** | **0.8409** |
| mAP50-95 | 0.5075 |
| Precision | 0.8378 |
| Recall | 0.8021 |

### Training Curve

| Epoch | mAP50 | mAP50-95 | Precision | Recall | box_loss |
|-------|-------|----------|-----------|--------|----------|
| 1 | 0.3742 | 0.1809 | 0.4341 | 0.4441 | 1.8309 |
| 10 | 0.7486 | 0.4362 | 0.7941 | 0.6961 | 1.5153 |
| 20 | 0.7787 | 0.4606 | 0.7983 | 0.7352 | 1.4510 |
| 30 | 0.7976 | 0.4733 | 0.8068 | 0.7585 | 1.3873 |
| 40 | 0.8209 | 0.4909 | 0.8255 | 0.7843 | 1.3437 |
| 50 | 0.8236 | 0.4969 | 0.8292 | 0.7955 | 1.2918 |
| **53** | **0.8409** | **0.5075** | 0.8378 | 0.8021 | 1.2780 |
| 60 | 0.8359 | 0.5056 | 0.8415 | 0.8058 | 1.2530 |
| 70 | 0.8385 | 0.5102 | 0.8424 | 0.8065 | 1.2085 |
| 80 | 0.8337 | 0.5044 | 0.8415 | 0.8123 | 1.1626 |
| 90 | 0.8348 | 0.5006 | 0.8503 | 0.8027 | 1.0588 |

### Convergence Analysis

- **Rapid initial learning**: mAP50 jumped from 0.374 → 0.749 in first 10 epochs
- **Steady improvement**: Epochs 10-40, mAP50 from 0.749 → 0.821
- **Peak at epoch 53**: mAP50=0.841
- **Plateau**: Epochs 50-70 oscillating around 0.835-0.841
- **Slight overfitting after 70**: val_box_loss flat while train_box_loss continues to drop
- **No early stopping triggered** — training ran to completion

### Overfitting Assessment

Train box_loss continued decreasing (1.83 → 1.06) while val box_loss plateaued (~1.44-1.47), indicating mild overfitting after epoch 50. The best model at epoch 53 captures peak validation performance before overfitting becomes significant. The model at epoch 90 still has comparable mAP50 (0.835 vs 0.841) — degradation is minimal, suggesting good generalization.

## Comparison with Previous Baseline

| Metric | Old (HUST-OBC YOLOv8n) | New (Competition YOLO11s) |
|--------|------------------------|---------------------------|
| Architecture | YOLOv8n (3.0M params) | YOLO11s (9.4M params) |
| Input size | 640×640 | 1280×1280 |
| Training data | HUST-OBC only | Competition data |
| mAP50 | 0.652 | **0.841** |
| mAP50-95 | 0.373 | **0.508** |
| Precision | 0.728 | **0.838** |
| Recall | 0.617 | **0.802** |

**29% mAP50 improvement** over the old detector. Both higher resolution (1280 vs 640) and the YOLO11s architecture contribute.

## Deliverables

| File | Size | Path |
|------|------|------|
| best.pt | 18.3 MB | `checkpoints/yolo11s_det_1280.pt` |
| last.pt | — | `runs/yolo11s_det_1280/weights/last.pt` |
| results.csv | — | `runs/yolo11s_det_1280/results.csv` |

## Conclusion

Detection training complete. YOLO11s at 1280px achieves mAP50=0.841 on competition validation set. Model converged well with no significant overfitting. Ready for inference deployment and Phase 2B error analysis.
