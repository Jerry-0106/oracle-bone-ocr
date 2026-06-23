# Experiment Retrospective

## Effective Directions (+)

### Detection Tuning
- **conf=0.20 / iou=0.25** — Optimal detector params. Reduced FP while maintaining TP (+0.0098 local F1 vs conf=0.15).
- **Dynamic crop padding** — pad=max(8, 0.08*max(w,h)). Gave enough context for recognizer.
- **Remove bbox shrink** — Output original YOLO bbox. Preserved full character area.

### Recognition Architecture
- **Swin-Tiny + CE >> ConvNeXt-Tiny + ArcFace** — Largest single gain: +0.0868 online F1 (v9 → v11). Swin's hierarchical attention is better suited for oracle bone characters than ConvNeXt.
- **Cross-Entropy loss with label smoothing (0.05)** — Simple, stable, effective. No need for ArcFace angular margin for this task.

### Training Strategy
- **Mixed data: original + pipeline-style crops (50/50)** — Weighted sampling helped balance class distribution.
- **Continued training E20 → E25 → E30** — Consistent local improvement. Each epoch +0.005~0.008 F1.

### Infrastructure
- **timm dependency** — Swin models require timm package in Docker.
- **run.sh LF line endings** — CRLF causes silent failures on Linux containers.

## Ineffective or Insufficient Directions (-)

### Recognition Architectures (tested, not adopted)
| Experiment | Best Local F1 | vs Tiny E30 | Verdict |
|---|---|---|---|
| DINOv2 kNN retrieval | < 0.30 | Much worse | Not suitable for fine-grained OCR |
| DINOv2 supervised | < 0.40 | Much worse | Poor convergence on 3483 classes |
| ConvNeXt-Small hybrid | < 0.50 | Much worse | Various init strategies failed |
| ConvNeXt@384 | < 0.55 | Much worse | Resolution increase didn't help ConvNeXt |
| Swin-Small random head (Plan 7B) | 0.5192 @ E5 | -0.1949 | Slow convergence from scratch |
| Swin-Small True Hybrid (Plan 7C) | 0.7100 @ E10 | -0.0041 | Close but didn't surpass Tiny E30 |

### Training Techniques (tested, degraded)
| Technique | Effect |
|---|---|
| TTA (hflip, multi-pad) | -0.0121 ~ -0.0141 F1 |
| Mixup / CutMix | Not tested (excluded by design) |
| Knowledge Distillation | No improvement in Plan 3 |
| ArcFace loss (for Swin) | Not needed; CE works better |

### Post-Processing (tested, degraded)
| Technique | Effect |
|---|---|
| Confidence rejection | Reduced TP without significant FP reduction |
| Page candidate rerank | No gain on this dataset |
| Retrieval-based rerank | Degraded performance |
| Ensemble (multi-checkpoint) | Not tested (excluded for simplicity) |

### Training Extensions (diminishing returns)
| Extension | Local ΔF1 | Online ΔF1 |
|---|---|---|
| E20 → E25 | +0.0263 | +0.0072 |
| E25 → E30 | +0.0163 | +0.0007 |
| E30 → E35 | E31/E32 dropped | Not submitted |

## Core Conclusions

1. **Recognition is the bottleneck** — Detection F1 (0.9087) is near-solved; Recognition Acc (0.545) limits final F1.

2. **Swin-Tiny is the right architecture** — Beats ConvNeXt by a large margin. Good balance of capacity and training efficiency.

3. **Local→Online gap is large** — Local 931 corrected F1 (0.7141) vs online F1 (0.4956) = gap of 0.2185. The 931 val set is not fully representative of the online test distribution.

4. **E30 is the practical peak** — Further training (E31-E35) showed local drops. E25→E30 online gain was negligible (+0.0007), suggesting overfitting to local validation set.

5. **Swin-Small True Hybrid is close but not worth the risk** — E10 F1=0.7100 local vs Tiny E30=0.7141. Gap of -0.0041 is small but real. The extra 51.5M vs 28.3M params don't justify the marginal potential gain.

6. **Simple pipeline wins** — No TTA, no ensemble, no rerank. Clean detection + single recognizer + CE loss.
