# Plan 1: Clean Data + Light Augmentation — Failed

## Date

2026-06-16

## Branch

`exp/rec_clean_aug_light`

## Hypothesis

Cleaning garbage labels (□, ZHFD-*, len>1) + blank/corrupted images, and replacing RandAugment with lightweight RandomAffine, would improve recognition accuracy.

## Data Cleaning

| Metric | Value |
|--------|-------|
| Original samples | 61007 |
| Removed garbage labels | 935 (□: 47, ZHFD-*: 887, len>1: 1) |
| Removed blank/corrupted | 20 (19 blank, 1 too small) |
| Clean samples | 60052 |
| Classes | 3483 (unchanged) |

## Training Config

| Param | Value |
|-------|-------|
| Model | ConvNeXt-Tiny + ArcFace (s=30, m=0.5) |
| Hot-start | convnext_arcface_best.pt (epoch 53, top1=0.3218) |
| Augmentation | RandomAffine(3°, 5% translate, 5% scale) |
| Epochs | 10 (probe) |
| LR | 1e-4 (backbone: 1e-5) |
| Batch size | 128 |
| AMP | ON |

## Training Results

| Epoch | train_top1 | val_top1 | val_loss |
|-------|-----------|----------|----------|
| 1 | 0.0886 | **0.5047** | 3.9953 |
| 2 | 0.1149 | 0.5002 | 4.0224 |
| 3 | 0.1434 | 0.4917 | 4.0461 |
| 5 | 0.1691 | 0.4837 | 4.0988 |
| 8 | 0.1845 | 0.4734 | 4.1283 |
| 10 | 0.1878 | 0.4745 | 4.1338 |

Best: Epoch 1, val_top1=0.5047. Degraded after E1.

## 931 Pipeline Evaluation (Epoch 1 best)

| Metric | Plan 1 | v9 baseline | Δ |
|--------|--------|-------------|---|
| Det F1 | 0.8054 | 0.8054 | 0 |
| Rec Acc | 0.5298 | 0.5380 | **-0.0082** |
| Final F1 | 0.6867 | 0.6901 | **-0.0034** |

## Root Causes of Failure

1. **Val set shift**: Cleaning changed total samples from 60072→60052, which changed the random_split output. The new val set had only 14.7% overlap with the original val set. The 50.47% val_top1 is NOT comparable to the baseline 32.18%.

2. **LR scheduler bug**: The warmup+cosine scheduler produced LR ~1e-5 (backbone) instead of the expected ~1e-4. Training was at 10x lower LR than intended, causing under-update.

3. **Model degradation**: val_top1 peaked at epoch 1 and steadily decreased, indicating the model was diverging from the hot-start rather than improving.

## Verdict

**FAILED. Do not use this checkpoint.** v9 baseline remains the stable version.

## Artifacts

- Checkpoint: `checkpoints/convnext_arcface_clean_best.pt` (do not use)
- Training log: `reports/rec_clean_training/training.log`
- Cleaning report: `reports/data_cleaning/cleaning_report.md`
- Clean labels: `datasets/recognition/labels_clean.csv`

## Lessons for Next Attempt

1. **Fix val split BEFORE cleaning**: Save the original split as fixed files, then clean only training data (keep val unchanged).
2. **Fix LR scheduler**: Use a constant LR or properly debugged warmup.
3. **Validate at epoch 0**: Run 931 pipeline eval BEFORE any training to confirm hot-start doesn't degrade.
