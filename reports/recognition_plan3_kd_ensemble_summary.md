# Plan 3: KD-Stabilized Fine-tune + Teacher-Student Ensemble

## Date

2026-06-17

## Plan 3: KD Training

| Param | Value |
|-------|-------|
| Teacher | convnext_arcface_best.pt (frozen) |
| Student | Same architecture, hot-start from teacher |
| Data | 50% original rec + 50% pipeline-style |
| KD | T=2.0, λ=0.5 |
| LR | backbone=1e-6, head=1e-5 (constant) |
| Epochs | 3 |

### Training Results

| Epoch | orig_val Top1 | pipe_val Top1 |
|-------|--------------|--------------|
| 0 | 0.3218 | 0.5526 |
| 1 | 0.3383 | 0.5529 |
| 2 | 0.3452 | 0.5516 |
| 3 | **0.3495** | 0.5517 |

KD prevented catastrophic forgetting — orig_val improved +0.0277.

### 931 Pipeline (Student Only)

| Metric | v9 | Plan 3 E3 | Δ |
|--------|-----|-----------|------|
| Rec Acc | 0.5380 | 0.5332 | -0.0048 |
| Final F1 | 0.5323 | 0.5288 | -0.0035 |

Student alone is worse than v9. Not replacing.

## Plan 3E: Teacher-Student Logits Ensemble

Zero-training ensemble: `logits = wt × teacher_logits + ws × student_logits`

### Broad Sweep

| wt | ws | Final F1 | Δ vs v9 |
|-----|-----|----------|----------|
| 1.0 | 0.0 | 0.5322 | baseline |
| 0.9 | 0.1 | 0.5348 | +0.0025 |
| 0.8 | 0.2 | 0.5362 | +0.0039 |
| 0.7 | 0.3 | 0.5366 | +0.0043 |
| 0.5 | 0.5 | 0.5355 | +0.0032 |

### Fine Sweep

| wt | ws | Final F1 | Rec Acc | TP | FP |
|-----|-----|----------|---------|-----|------|
| 0.85 | 0.15 | 0.5355 | 0.5424 | 3947 | 5028 |
| 0.80 | 0.20 | 0.5362 | 0.5434 | 3954 | 5021 |
| 0.75 | 0.25 | 0.5364 | 0.5436 | 3956 | 5019 |
| **0.70** | **0.30** | **0.5366** | **0.5439** | 3958 | 5017 |
| 0.65 | 0.35 | 0.5366 | 0.5439 | 3958 | 5017 |
| 0.60 | 0.40 | 0.5363 | 0.5435 | 3955 | 5020 |

### Verdict

- All ensemble configs exceed v9 baseline
- Best: 0.70/0.30, Final F1 = 0.5366 (+0.0043)
- Did NOT reach v10 candidate threshold (0.5370, missed by 0.0004)
- Did NOT reach strong candidate threshold (0.5400)
- **Status: Technical reserve. Not submitted. Not replacing v9.**

## Current Stable Version

v9_candidate_conf020_iou025 remains the active submission:
- Online Final F1 = 0.400821
- Local Corrected Final F1 = 0.5323
- Local Rec Acc = 0.5380

## Assets

- Student checkpoint: `checkpoints/plan_3_best.pt` (epoch 3, 3483 classes)
- Ensemble predictions: `reports/plan_3e2/prediction_t0.70_s0.30.json`
- Training log: `reports/plan_3_kd/training.log`
- Ensemble eval scripts: `_eval/eval_plan_3e.py`, `_eval/eval_plan_3e2.py`
