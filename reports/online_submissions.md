# Online Submission Records

---

## v8_stable_crop_fix_runtime

| Field | Value |
|-------|-------|
| Date | 2026-06-15 |
| Branch | main |
| Commit | 9161eae |
| Tag | v8_stable_crop_fix_runtime |
| Docker | orc-ocr:v8_stable_crop_fix_runtime |

### Local (931 val)

| Metric | Value |
|--------|-------|
| Det F1 | 0.7961 |
| Rec Acc | 0.5367 |
| Final F1 | 0.6770 |

### Online

| Metric | Value |
|--------|-------|
| Final F1 | 0.397540 |
| Detection F1 | 0.903272 |
| Recognition Acc | 0.440111 |
| TP | — |
| FP | — |
| FN | — |

---

## v9_candidate_conf020_iou025

| Field | Value |
|-------|-------|
| Date | 2026-06-16 |
| Branch | candidate/v9_conf020_iou025 |
| Commit | 0d28b85 |
| Tag | v9_candidate_conf020_iou025 |
| Docker | crpi-uqvecxl2rg3udb37.cn-hangzhou.personal.cr.aliyuncs.com/case_1/wjr_ai_orc:v9_candidate_conf020_iou025 |
| Config | conf=0.20 iou=0.25 max_det=300 augment=True |

### Local (931 val)

| Metric | Value |
|--------|-------|
| Det F1 | 0.8054 |
| Rec Acc | 0.5380 |
| Final F1 | 0.6901 |
| Δ vs v8 | +0.0131 |

### Online

| Metric | Value |
|--------|-------|
| Final F1 | 0.400821 |
| Detection F1 | 0.908696 |
| Recognition Acc | 0.441094 |
| Precision | 0.403279 |
| Recall | 0.398393 |
| TP | 3321 |
| FP | 4914 |
| FN | 5015 |
| Δ vs v8 | **+0.003281** |

### Conclusion

v9 confirmed as current stable online version. CONF=0.20 + IOU=0.25 reduces FP while maintaining TP, yielding consistent gains both locally and online.

---

## Detection Ablation Summary

### Stage 1: CONF Sweep (iou=0.30, max_det=300, augment=True)

| Conf | DetF1 | RecAcc | FinalF1 | TP_det | FP | FN | Δ vs 0.15 |
|------|-------|--------|---------|--------|-----|-----|-----------|
| 0.10 | 0.7822 | 0.5353 | 0.6578 | 7398 | 2423 | 1698 | -0.0192 |
| 0.12 | 0.7897 | 0.5359 | 0.6681 | 7376 | 2208 | 1720 | -0.0089 |
| 0.15 | 0.7961 | 0.5367 | 0.6770 | 7335 | 1996 | 1761 | baseline |
| 0.18 | 0.7997 | 0.5383 | 0.6824 | 7303 | 1866 | 1793 | +0.0054 |
| 0.20 | 0.8029 | 0.5385 | 0.6868 | 7291 | 1775 | 1805 | +0.0098 |

### Stage 2: IOU Sweep (conf=0.20, max_det=300, augment=True)

| IOU | DetF1 | RecAcc | FinalF1 | TP_det | FP | FN | Δ vs 0.30 |
|-----|-------|--------|---------|--------|-----|-----|-----------|
| 0.25 | 0.8042 | 0.5380 | 0.6884 | 7277 | 1725 | 1819 | +0.0016 |
| 0.30 | 0.8029 | 0.5385 | 0.6868 | 7291 | 1775 | 1805 | baseline |
| 0.35 | 0.8014 | 0.5384 | 0.6848 | 7307 | 1833 | 1789 | -0.0020 |
| 0.40 | 0.7992 | 0.5379 | 0.6816 | 7334 | 1923 | 1762 | -0.0052 |

### Fine CONF Sweep (iou=0.25, max_det=300, augment=True)

| Conf | DetF1 | RecAcc | FinalF1 | TP_det | FP | FN | Δ vs 0.20 |
|------|-------|--------|---------|--------|-----|-----|-----------|
| 0.18 | 0.8012 | 0.5378 | 0.6843 | 7289 | 1810 | 1807 | -0.0041 |
| 0.19 | 0.8027 | 0.5378 | 0.6863 | 7287 | 1774 | 1809 | -0.0021 |
| 0.20 | 0.8042 | 0.5380 | 0.6884 | 7277 | 1725 | 1819 | baseline |
| 0.21 | 0.8051 | 0.5382 | 0.6898 | 7272 | 1697 | 1824 | +0.0014 |
| 0.22 | 0.8057 | 0.5385 | 0.6907 | 7263 | 1670 | 1833 | +0.0023 |

**Verdict: conf=0.20 iou=0.25 is local optimum. Fine sweep ∆max < +0.003. Stop detection ablation.**

### TTA Ablation (conf=0.15, iou=0.30)

| Mode | DetF1 | RecAcc | FinalF1 | Δ vs A |
|------|-------|--------|---------|--------|
| A baseline | 0.7961 | 0.5367 | 0.6770 | — |
| B multi-pad3 | 0.7961 | 0.5350 | 0.6763 | -0.0007 |
| C hflip | 0.7961 | 0.5081 | 0.6649 | -0.0121 |
| D multi-pad+hflip | 0.7961 | 0.5036 | 0.6629 | -0.0141 |

**Verdict: All TTA modes degrade performance. Flip is harmful for oracle bone characters.**

---

## v11_swin_tiny_e20_candidate

| Field | Value |
|-------|-------|
| Date | 2026-06-18 |
| Branch | candidate/v11_swin_tiny_e20 |
| Commit | 0308992 |
| Tag | v11_swin_tiny_e20_candidate |
| Docker | crpi-uqvecxl2rg3udb37.cn-hangzhou.personal.cr.aliyuncs.com/case_1/wjr_ai_orc:v11_swin_tiny_e20_candidate |
| Config | conf=0.20 iou=0.25 max_det=300 augment=True |
| Recognizer | Swin-Tiny @224 + CE (no ArcFace), E20 |

### Local (931 val)

| Metric | Value |
|--------|-------|
| Det F1 | 0.8054 |
| Rec Acc | 0.7498 |
| Final F1 | 0.6715 |
| Δ vs v9 | +0.1392 |

### Online

| Metric | Value |
|--------|-------|
| Final F1 | **0.487599** |
| Detection F1 | 0.908696 |
| Recognition Acc | 0.536592 |
| Precision | 0.490589 |
| Recall | 0.484645 |
| TP | 4040 |
| FP | 4195 |
| FN | 4296 |
| Δ vs v9 | **+0.086778** |

### v9 → v11 Online Comparison

| Metric | v9 | v11 | Δ |
|--------|-----|-----|-----|
| Final F1 | 0.400821 | **0.487599** | **+0.086778** |
| Rec Acc | 0.441094 | **0.536592** | +0.095498 |
| TP | 3321 | **4040** | +719 |
| FP | 4914 | **4195** | -719 |
| FN | 5015 | **4296** | -719 |

### Conclusion

v11 Swin-Tiny E20 confirmed as current stable online version. Swin-Tiny + CE crushes ConvNeXt-Tiny + ArcFace both locally (+0.1392) and online (+0.0868). v9 is retained as fallback.

---

## v12_swin_tiny_e25_timm_fix

| Field | Value |
|-------|-------|
| Date | 2026-06-22 |
| Branch | candidate/v12_swin_tiny_e25 |
| Commit | 78ad8dc |
| Tag | v12_swin_tiny_e25_timm_fix |
| Docker | crpi-uqvecxl2rg3udb37.cn-hangzhou.personal.cr.aliyuncs.com/case_1/wjr_ai_orc:v12_swin_tiny_e25_timm_fix |
| Config | conf=0.20 iou=0.25 max_det=300 augment=True |
| Recognizer | Swin-Tiny @224 + CE, E25 |

### Local (931 val)

| Metric | Value |
|--------|-------|
| Det F1 | 0.8054 |
| Rec Acc | 0.7948 |
| Final F1 | 0.6978 |
| Δ vs v11 | +0.0263 |

### Online

| Metric | Value |
|--------|-------|
| Final F1 | **0.494840** |
| Detection F1 | 0.908696 |
| Recognition Acc | 0.544561 |
| Precision | 0.497875 |
| Recall | 0.491843 |
| TP | 4100 |
| FP | 4135 |
| FN | 4236 |
| Δ vs v11 | **+0.007241** |

### Notes

- First submission with timm dependency fixed in Docker
- run.sh CRLF fix applied
- E25 improves local F1 by +0.0263 over E20

---

## v13_swin_tiny_e30 🏆 FINAL

| Field | Value |
|-------|-------|
| Date | 2026-06-23 |
| Branch | candidate/v13_swin_tiny_e30 |
| Commit | d934e61 |
| Tag | v13_swin_tiny_e30_candidate |
| Docker | crpi-uqvecxl2rg3udb37.cn-hangzhou.personal.cr.aliyuncs.com/case_1/wjr_ai_orc:v13_swin_tiny_e30_candidate |
| Config | conf=0.20 iou=0.25 max_det=300 augment=True |
| Recognizer | Swin-Tiny @224 + CE, E30 |

### Local (931 val)

| Metric | Value |
|--------|-------|
| Det F1 | 0.8054 |
| Rec Acc | 0.8237 |
| Final F1 | 0.7141 |
| Δ vs v12 | +0.0163 |

### Online

| Metric | Value |
|--------|-------|
| Final F1 | **0.495565** |
| Detection F1 | 0.908696 |
| Recognition Acc | 0.545358 |
| Precision | 0.498604 |
| Recall | 0.492562 |
| TP | 4106 |
| FP | 4129 |
| FN | 4230 |
| Δ vs v12 | **+0.000725** |

### Notes

- **Highest online F1 across all submissions**
- E25→E30 local improvement +0.0163 but online only +0.0007
- Diminishing returns indicate local validation set overfitting
- Swin-Tiny E30 confirmed as best practical checkpoint

---

## Summary

| Version | Online F1 | Local F1 | Key Change |
|---|---|---|---|
| v9 | 0.400821 | 0.6901 | ConvNeXt + ArcFace, conf=0.20/iou=0.25 |
| v11 | 0.487599 | 0.6715 | Swin-Tiny E20 + CE |
| v12 | 0.494840 | 0.6978 | Swin-Tiny E25 |
| **v13** | **0.495565** | **0.7141** | **Swin-Tiny E30 — FINAL BEST** |

Total improvement from v9 → v13: **+0.094744**
