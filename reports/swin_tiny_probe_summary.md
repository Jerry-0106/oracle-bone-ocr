# Plan 6: Swin-Tiny Supervised Probe — Breakthrough

## Date

2026-06-18

## Model

Swin-Tiny (timm: `swin_tiny_patch4_window7_224`), ImageNet pretrained
Linear classifier head, CrossEntropyLoss(label_smoothing=0.05)
30.2M params, VRAM 4.7G

## E10 Results

| Epoch | orig_top1 | pipe_top1 | 931 F1 | Rec Acc | TP | FP | FN |
|-------|-----------|-----------|--------|---------|------|------|------|
| 10 | 0.6072 | 0.6353 | **0.5910** | 0.6221 | 4527 | 4448 | 1819 |

## vs v9 Baseline

| Metric | v9 (ConvNeXt+ArcFace) | Swin E10 | Δ |
|--------|----------------------|----------|------|
| Final F1 | 0.5323 | 0.5910 | +0.0587 |
| Rec Acc | 0.5380 | 0.6221 | +0.0841 |
| TP | 3915 | 4527 | +612 |
| FP | 5060 | 4448 | -612 |

## Training Curve

| Epoch | 931 F1 | Δ vs v9 |
|-------|--------|----------|
| 2 | 0.3821 | -0.1502 |
| 3 | 0.4281 | -0.1042 |
| 4 | 0.4625 | -0.0698 |
| 5 | 0.4942 | -0.0381 |
| 6 | 0.5217 | -0.0106 |
| 7 | 0.5425 | +0.0102 |
| 8 | 0.5611 | +0.0288 |
| 9 | 0.5776 | +0.0453 |
| 10 | 0.5910 | +0.0587 |

## Key Findings

1. Swin-Tiny + simple CE loss crushes ConvNeXt-Tiny + ArcFace
2. 10 epochs from scratch >>> 53 epochs of fine-tuning
3. Swin's hierarchical features are better suited for oracle bone characters
4. No ArcFace, no KD, no ensemble — pure supervised CE

## E10→E15 Extension

| Epoch | 931 F1 | Rec Acc | TP | FP | Δ vs v9 |
|-------|--------|---------|------|------|----------|
| 11 | 0.5917 | 0.6232 | 4535 | 4440 | +0.0594 |
| 12 | 0.5918 | 0.6233 | 4536 | 4439 | +0.0595 |
| 13 | 0.6090 | 0.6494 | 4726 | 4249 | +0.0767 |
| 14 | 0.6220 | 0.6695 | 4872 | 4103 | +0.0897 |
| **15** | **0.6341** | **0.6886** | **5011** | **3964** | **+0.1018** |

## E15 vs v9

| Metric | v9 | Swin E15 | Δ |
|--------|-----|----------|------|
| Final F1 | 0.5323 | 0.6341 | +0.1018 |
| Rec Acc | 0.5380 | 0.6886 | +0.1506 |
| TP | 3915 | 5011 | +1096 |
| FP | 5060 | 3964 | -1096 |

## Checkpoints

- E10: `checkpoints/swin_tiny_e10_v11_candidate.pt`
- E15: `checkpoints/swin_tiny_e15_v11_candidate.pt`
