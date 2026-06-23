# Recognition Mapping Report (Phase 2C)

## Data Sources

| Source | Unique Characters | Total Samples | Description |
|--------|------------------|---------------|-------------|
| HUST-OBC (ID_to_chinese.json) | 1,781 | ~49,647 images (1,016 class dirs) | Deciphered oracle bone characters |
| Competition (labels.csv, cleaned) | 3,483 | 60,072 valid samples | Characters from competition training set |

## Class Overlap Analysis

| Category | Count | % of Competition |
|----------|-------|-----------------|
| **Overlap** (in both) | **939** | 27.0% of 3,483 |
| **Competition-only** (new) | **2,544** | 73.0% of 3,483 |
| **HUST-OBC-only** | 842 | — |

### Sample Distribution

| Category | Samples | % of Competition Total |
|----------|---------|----------------------|
| Overlap characters | 45,185 | **75.2%** |
| Competition-only characters | 14,887 | 24.8% |
| Total | 60,072 | 100% |

**Key insight**: While only 27% of character classes overlap, they account for 75.2% of all training samples. The 2,544 new classes are mostly rare (long-tail).

## Label Quality

| Type | Count | Action |
|------|-------|--------|
| Valid CJK characters | 60,072 | Use directly |
| ZHFD-* metadata strings | 888 | Filter out |
| "□" placeholder | 47 | Filter out |
| **Total valid** | **60,072** | — |
| **Total garbage** | **935** | Remove from dataset |

## HUST-OBC Training Data

- **1,016 class directories** under `backup_unused/datasets/recognition_data/deciphered/`
- Directory names use composite IDs (e.g., `0011_0012_0013`) for merged classes
- **No pre-made train/val split** — split must be created at training time
- The 1,016 directories map to 1,781 unique Chinese characters (via `ID_to_chinese.json`)

## Implications for Recognition Training

### Transfer Learning Strategy

1. **939 shared classes**: Can directly transfer weights from HUST-OBC pretrained `recognizer.pt`
2. **2,544 new classes**: Need new classifier head; weights initialized randomly
3. **842 HUST-OBC-only classes**: Irrelevant for competition; can be pruned or kept as auxiliary

### Recommended Approach

**Two-stage training**:

```
Stage 1 (pretrain): HUST-OBC 1,588 classes → Recognizer backbone
Stage 2 (finetune): Replace head with 3,483 classes
                    → Freeze backbone (epochs 1-5)
                    → Unfreeze all (epochs 6-50)
                    → Class-balanced sampling for tail classes
```

### Class Count Reconciliation

The original `dataset_stats.json` reported 4,101 classes, which included 618 garbage labels (ZHFD-* metadata strings and "□"). After cleaning, the true class count is **3,483**.
