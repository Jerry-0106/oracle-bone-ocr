
---

## E30 (v13 candidate) — 2026-06-22

| Metric | Value |
|---|---|
| **931 corrected Final F1** | **0.7141** |
| Rec Acc | 0.8237 |
| orig_val Top1 | 0.7961 |
| pipe_val Top1 | 0.8673 |
| TP | 5994 |
| FP | 2981 |
| FN | 1819 |
| Wrong text | 1283 |
| Precision | 0.6679 |
| Recall | 0.7672 |
| Checkpoint | `checkpoints/swin_tiny_e30_v13_candidate.pt` |

**Status**: 🏆 Current Best (ahead of E25 by +0.0163)

**Plan**: Tomorrow pack as v13_swin_tiny_e30_candidate Docker image if no higher epoch surpasses it.

---

## E30→E35 Extension Results — 2026-06-22

| Epoch | F1 | Rec Acc | TP | FP | FN | Wrong | P | R | Δ vs E30 | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| **E30** | **0.7141** | 0.8237 | 5994 | 2981 | 1819 | 1283 | 0.6679 | 0.7672 | — | 🏆 BEST |
| E31 | 0.7057 | 0.8087 | 5885 | 3090 | 1819 | 1392 | 0.6557 | 0.7639 | -0.0084 | drop #1 |
| E32 | 0.7105 | 0.8174 | 5948 | 3027 | 1819 | 1329 | 0.6627 | 0.7658 | -0.0036 | drop #2 → STOP |

**Stop reason**: 2 consecutive F1 drops (E31, E32 both below E30).

**Conclusion**: E30 is the peak for Swin-Tiny @224 + CE. Further epochs yield diminishing returns.
