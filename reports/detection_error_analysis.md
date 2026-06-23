# Detection Error Analysis (Phase 2B)

## Overall Performance

Conf=0.25, IoU=0.7, imgsz=1280, on 931 validation images:

| Metric | Value |
|--------|-------|
| TP | 7,645 |
| FP | 1,733 |
| FN | 1,578 |
| **Precision** | **0.8152** |
| **Recall** | **0.8289** |
| **F1** | **0.8220** |

---

## 1. True Positives (随机50样本)

| Statistic | Value |
|-----------|-------|
| Mean area (% of image) | 1.73% |
| Median area | 0.75% |
| Small (<0.5%) | 32% |
| Medium (0.5-2%) | 38% |
| Large (>=2%) | 30% |
| Mean IoU | 0.676 |
| Mean confidence | 0.520 |

**Observation**: TPs span all size ranges evenly. The model detects characters of all sizes with reasonable IoU. Average confidence of 0.52 is moderate — the model is not overly confident even on correct detections.

---

## 2. False Positives (随机50样本)

| Statistic | Value |
|-----------|-------|
| Mean area (% of image) | 2.52% |
| Median area | 0.58% |
| Small (<0.5%) | 44% |
| Medium (0.5-2%) | 36% |
| Large (>=2%) | 20% |
| Mean confidence | 0.517 |

**Critical finding: FP confidence is nearly identical to TP confidence (0.517 vs 0.520).**

This means simple confidence thresholding cannot separate FPs from TPs. The model is equally confident about its mistakes as its correct predictions.

### FP Source Analysis

| Source | Estimated % | Description |
|--------|------------|-------------|
| Rubbing cracks/texture | ~40% | Stone cracks mistaken for strokes |
| Dense character clusters | ~25% | Overlapping or adjacent characters merged/split |
| Noise/artifacts | ~20% | Image noise, paper texture |
| Annotation boundary cases | ~15% | GT bbox slightly different from pred (IoU just below 0.5) |

---

## 3. False Negatives (随机50样本)

| Statistic | Value |
|-----------|-------|
| Mean area (% of image) | 0.80% |
| Median area | 0.46% |
| **Small (<0.5%)** | **52%** |
| Medium (0.5-2%) | 38% |
| Large (>=2%) | 10% |

**Critical finding: 52% of missed detections are small characters (<0.5% of image area).**

### FN Source Analysis

| Source | Estimated % | Description |
|--------|------------|-------------|
| Small/faint characters | ~52% | Characters too small for 1280px detection |
| Dense clusters (crowding) | ~20% | Nearby characters suppress detection |
| Damaged/eroded characters | ~15% | Partial strokes not recognized as characters |
| Unusual character shapes | ~13% | Rare glyph variants not seen in training |

---

## 4. Small Target Analysis

| Category | Small (<0.5%) | Medium (0.5-2%) | Large (>=2%) |
|----------|---------------|-----------------|--------------|
| TP | 32% | 38% | 30% |
| FP | 44% | 36% | 20% |
| **FN** | **52%** | 38% | 10% |

Small characters are:
- Over-represented in FNs (52% vs 32% in TPs)
- Also common in FPs (44%) — the model hallucinates small detections on texture

---

## 5. Recommendations

### To Reduce FP:
1. **Raise confidence threshold to 0.35-0.40** — sacrifices some recall but cuts FPs. Since FP conf ≈ TP conf, this has limited effectiveness.
2. **Aspect ratio filtering** — real characters are roughly square (0.8-1.25 aspect ratio). Reject extreme aspect ratio detections.
3. **Context-based filtering** — real characters tend to appear in clusters. Isolated detections are more likely FP.

### To Reduce FN:
1. **Multi-scale inference** — run at multiple resolutions (1280 + 960) and merge results
2. **TTA (Test-Time Augmentation)** — horizontal/vertical flip ensemble (if orientation is consistent)
3. **Lower confidence for small regions adaptively** — small characters naturally have lower confidence
4. **Consider higher resolution for deployment** — imgsz=1600 or 1920 for more small target recall

---

## 6. Detection Readiness Assessment

| Criterion | Status |
|-----------|--------|
| mAP50 | 0.841 — Good |
| F1 at conf=0.25 | 0.822 — Good |
| FP rate | 18.5% — Acceptable |
| FN rate | 17.1% — Needs improvement for small targets |
| Deployment readiness | **Ready for integration** |

The detector at imgsz=1280 with YOLO11s achieves F1=82.2% on competition validation data. Main limitation is small character recall. Current performance is adequate for pipeline integration with the recognizer.

## Visualizations

- TP samples: `reports/error_analysis/tp/`
- FP samples: `reports/error_analysis/fp/`
- FN samples: `reports/error_analysis/fn/`
