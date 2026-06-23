# HUST-OBC Dataset Analysis (Phase 3A Task 2)

## Data Overview

| Metric | Value |
|--------|-------|
| Data location | `backup_unused/datasets/recognition_data/deciphered/` |
| Class directories | 1,016 |
| Unique Chinese characters | 1,145 (after ID→Chinese mapping) |
| Total images | 49,647 |

## Class Distribution

| Metric | Value |
|--------|-------|
| Avg per class | 43.4 |
| Max per class | 318 |
| Min per class | 1 |
| Median per class | 39 |
| Classes with 1 sample | 32 (2.8%) |
| Classes with <5 samples | 133 (11.6%) |
| Classes with <10 samples | 221 (19.3%) |
| Imbalance ratio | 318:1 |

## Comparison with Competition Data

| Metric | HUST-OBC | Competition |
|--------|----------|-------------|
| Classes | 1,145 | 3,483 |
| Total samples | 49,647 | 60,072 |
| Avg/class | 43.4 | 17.2 |
| Singletons | 32 (2.8%) | ~2,000 (57%) |
| Imbalance ratio | 318:1 | 1,136:1 |
| Median/class | 39 | 2 |

**HUST-OBC is far more balanced** — much better for stable pretraining.

## Top 20 Characters

| Rank | Char | Samples |
|------|------|---------|
| 1 | 風 | 318 |
| 2 | 鳳 | 318 |
| 3 | 鳯 | 318 |
| 4 | 子 | 307 |
| 5 | 月 | 298 |
| 6 | 夕 | 298 |
| 7 | 福 | 246 |
| 8 | 衣 | 227 |
| 9 | 卒 | 227 |
| 10 | 燎 | 212 |

## Overlap with Competition

| Category | Classes | % |
|----------|---------|---|
| Shared (HUST ∩ Comp) | 939 | — |
| HUST-only | 206 | 18% |
| Competition-only | 2,544 | — |

939 shared classes make HUST-OBC excellent for transfer learning.

## Data Structure

- Flat directory: `deciphered/{class_id}/image.png`
- Class IDs are numeric with composite names (e.g., `0011_0012_0013`)
- No pre-existing train/val split
- `ID_to_chinese.json` maps class IDs to Chinese characters

## Training Readiness

**Ready.** Data is well-structured for PyTorch `ImageFolder` loading. Recommended split: 85/15 stratified by class.
