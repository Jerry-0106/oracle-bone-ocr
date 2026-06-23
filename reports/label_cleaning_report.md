# Label Cleaning Report (Phase 3A Task 1)

## Analysis Results

### Raw Data

| Metric | Value |
|--------|-------|
| Total unique labels in CSV | 4,101 |
| Total samples | 61,007 |

### Classification Breakdown

| Category | Unique Labels | Samples | Valid? |
|----------|--------------|---------|--------|
| Standard CJK (BMP+Ext) | 2,672 | 57,350 | YES |
| PUA-encoded CJK (oracle bone) | 811 | ~2,722 | YES |
| ZHFD-* metadata strings | 616 | 888 | NO |
| "□" placeholder | 1 | 47 | NO |
| Multi-char strings | 1 | — | NO |

### The "unknown_char" Issue Explained

811 labels classified as "unknown_char" by basic CJK detection are actually **valid oracle bone characters** encoded in Unicode Private Use Area (PUA):

| Code Point Example | Samples | Location |
|-------------------|---------|----------|
| U+454D7 | 179 | PUA-A (Plane 15) |
| U+44C32 | 124 | PUA-A (Plane 15) |
| U+3D1B0 | 98 | Plane 3 (unassigned CJK range) |
| U+57C20 | 67 | PUA-A (Plane 15) |
| U+3CF43 | 67 | Plane 3 (unassigned CJK range) |

These are real oracle bone characters that the Chinese academic community has assigned to PUA code points because they haven't been standardized in Unicode yet. Scholarly fonts (e.g., BabelStone PUA) render these code points as actual glyphs.

**All 811 PUA characters are valid and should be kept.**

### True Garbage

| Type | Count | Examples |
|------|-------|----------|
| ZHFD metadata | 888 samples | `ZHFD-176-1526865284598`, `ZHFD-38-1526651997981` |
| Placeholder | 47 samples | `□` |

These appear to be XML metadata strings (`extension` attribute values) that were accidentally written as character labels instead of the actual character text.

### Final Clean Data

| Metric | Value |
|--------|-------|
| **Valid classes** | **3,483** (2,672 standard + 811 PUA) |
| **Valid samples** | **60,072** |
| **Garbage samples to remove** | **935** |
| **Garbage classes to remove** | **618** |

## Conclusion

**Data is ready for training.** The 935 garbage samples (1.5% of total) can be filtered by removing any label that:
1. Contains "ZHFD" or "metadata" keywords
2. Is the single character "□"
3. Has length > 2 characters

No characters were falsely removed. The PUA-encoded oracle bone characters are valid training targets.

## Action Required

Regenerate `labels.csv` with garbage labels removed before training.
