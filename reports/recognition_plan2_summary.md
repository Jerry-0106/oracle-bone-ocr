# Plan 2: Pipeline-Style Recognition Dataset Probe — Failed

## Date

2026-06-16

## Branch

`exp/plan_1b_fixed`

## Dataset

| Metric | Value |
|--------|-------|
| Source | 5287 detection training images × XML annotations |
| Crop method | XML bbox + v9 padding (max(8, 8%)) |
| Samples | 50,986 |
| Classes | 3,241 (242 of 3483 have zero samples) |
| Train/Val | 43,339 / 7,647 |

## Training

| Param | Value |
|-------|-------|
| Model | ConvNeXt-Tiny + ArcFace (3483 classes) |
| Hot-start | convnext_arcface_best.pt (epoch 53, top1=0.3218) |
| Augmentation | RandomAffine(3°, 3%, 3%) → Resize(256) → CenterCrop(224) |
| LR | backbone=5e-6, head=5e-5 (constant, micro-finetune) |
| Epochs | 5 |
| Zero-sample class handling | Gradient mask on ArcFace head rows |

## Training Results

| Epoch | pipeline_style val_top1 | vs E0 |
|-------|------------------------|-------|
| 0 (hot-start) | **0.5526** | baseline |
| 1 | 0.5471 | -0.0055 |
| 2 | 0.5411 | -0.0115 |
| 3 | 0.5354 | -0.0172 |
| 4 | 0.5309 | -0.0217 |
| 5 | 0.5278 | -0.0248 |

Model degraded steadily from hot-start. No epoch improved over E0.

## 931 Pipeline Evaluation

| Metric | v9 baseline | Plan 2 E0 | Δ |
|--------|------------|-----------|------|
| Det F1 | 0.8054 | 0.8054 | 0 |
| Rec Acc | 0.5380 | 0.5380 | 0 |
| Final F1 | 0.5323 | 0.5323 | 0 |

Plan 2 E0 = v9 baseline (same hot-start, verified).

## Failure Analysis

Pipeline-style crops didn't help the model improve. Despite training on crops closer to the inference distribution, the model still degraded from hot-start. Possible causes:

1. **The hot-start model has already converged on this data distribution.** The original training used XML crops from similar images — the 8% padding change may not be large enough to shift the distribution meaningfully.

2. **Training hyperparameters are wrong for this dataset.** The LR might still be too high/low, or the augmentation may not be appropriate even at the "light" setting.

3. **The pipeline-style dataset has 15% fewer samples and 242 fewer classes than the original.** This data volume reduction may offset any distribution alignment benefit.

## Verdict

**FAILED. Pipeline-style crops alone don't improve the model.** Do not use Plan 2 checkpoints.

## Assets

- Dataset: `datasets/recognition_pipeline_style/`
- Audit: `reports/pipeline_style_crop_audit/`
- Checkpoints: `checkpoints/plan_2_epoch_*.pt` (do not use)
- Training log: `reports/plan_2/training.log`
