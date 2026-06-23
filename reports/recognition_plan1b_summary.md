# Plan 1b: Fixed Split + Light Aug + Constant LR — Failed

## Date

2026-06-16

## Branch

`exp/plan_1b_fixed`

## Configuration

| Param | Value |
|-------|-------|
| Model | ConvNeXt-Tiny + ArcFace (s=30, m=0.5) |
| Hot-start | convnext_arcface_best.pt (epoch 53, top1=0.3218) |
| Train data | original_train_split.txt, filtered garbage (51062 samples) |
| Val data | original_val_split.txt, NO filtering (9010 samples) |
| Augmentation | RandomAffine(3°, 5%, 5%) → Resize(256) → CenterCrop(224) |
| LR | backbone=1e-5, head=1e-4 (constant) |
| Epochs | 5 |
| Batch size | 128 |
| AMP | ON |

## Training Results

| Epoch | val_top1 | val_top5 |
|-------|----------|----------|
| 0 (hot-start) | 0.3218 | 0.4496 |
| 1 | 0.3329 | 0.4578 |
| 2 | 0.3357 | 0.4624 |
| 3 | 0.3352 | 0.4637 |
| **4 (best)** | **0.3394** | 0.4623 |
| 5 | 0.3386 | 0.4667 |

val_top1 improved from 0.3218 → 0.3394 (Δ=+0.0176).

## 931 Pipeline Evaluation (Epoch 4 best)

| Metric | v9 baseline | Plan 1b E4 | Δ |
|--------|------------|-----------|------|
| Det F1 | 0.8054 | 0.8054 | 0 |
| Rec Acc | **0.5380** | **0.5183** | **-0.0197** |
| Final F1 | **0.5323** | **0.5179** | **-0.0144** |
| TP_text | 3915 | 3772 | -143 |
| Wrong_text | 3362 | 3505 | +143 |
| FP_total | 5060 | 5203 | +143 |

## Failure Analysis

**val_top1 improved but pipeline performance degraded.** This reveals a fundamental issue:

1. **original_val_split is NOT a reliable proxy for pipeline accuracy.** The crops used during recognition validation (from XML bboxes) differ from the crops used during inference (from YOLO bboxes with padding). A model optimized on XML crops can perform worse on YOLO crops.

2. **The train/inference crop distribution mismatch remains the core bottleneck.** Light affine augmentation doesn't bridge this gap — it makes the model better at the training distribution but not at the inference distribution.

3. **All future recognition experiments MUST use 931 pipeline corrected F1 as the primary decision metric.** val_top1 on any split is only a secondary reference.

## Verdict

**FAILED. Do not use Plan 1b checkpoints.** v9 baseline remains the stable version.

## Artifacts

- Checkpoints: `checkpoints/plan_1b_epoch_*.pt` (do not use)
- Training log: `reports/plan_1b/training.log`
- History: `reports/plan_1b/history.json`

## Lessons

1. Pipeline crop distribution != training crop distribution
2. Must evaluate on real pipeline, not just recognition val
3. Fixed split alone doesn't solve the core problem
4. Train/inference crop alignment is the next thing to address
