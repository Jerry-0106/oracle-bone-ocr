# Plan 5: Train-Page Retrieval Reranking — Marginal

## Date

2026-06-17

## Method

For each validation image, retrieve topK most similar training pages using grayscale embedding cosine similarity + pHash. Use retrieved pages' XML character sets as candidate set for recognition reranking.

## Gallery

| Metric | Value |
|--------|-------|
| Training pages | 5,094 |
| Avg chars/page | 9.2 |

## Retrieval Quality

| TopK | GT Coverage | Avg Candidate Size |
|------|------------|-------------------|
| 1 | 9.2% | 7.5 |
| 3 | 20.0% | 20.9 |
| 5 | 26.6% | 33.0 |
| GT page candidate | 100% | — |

## Best Reranking Results (cached topK data)

| Config | F1 | Δ vs R0 |
|--------|-----|---------|
| R0 baseline | 0.5469 | — |
| T5_prior_a1.0 | 0.5512 | +0.0043 |
| T5_prior_a0.5 | 0.5510 | +0.0041 |
| T3_prior_a0.5 | 0.5503 | +0.0034 |
| T1_soft | 0.4590 | -0.0879 |

## Verdict

**Marginal improvement. Do not submit.**

- Best Δ = +0.0043, below +0.005 candidate threshold
- Retrieval coverage too low (9-27%) for reliable reranking
- Soft rerank severely degrades performance
- Prior-based rerank shows only marginal gains
- Image similarity ≠ character set similarity for oracle bone rubbings

## Status

**Technical reserve. Not submitted. Not replacing v9.**
