# HUST-OBC Pretraining Report (Phase 3A Task 5)

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | ConvNeXt-Tiny |
| Parameters | 28.6M |
| Pretrained | None (network blocked, trained from scratch) |
| Input size | 224×224 |
| Classes | 1,016 (HUST-OBC directory count) |
| Epochs | 40 |
| Batch size | 128 |
| Optimizer | AdamW (lr=0.002, wd=0.05) |
| LR schedule | 5-epoch warmup + cosine |
| Augmentation | RandAugment(n=2,m=9) + Mixup(0.2) + CutMix(0.2) |
| Regularization | LabelSmoothing=0.1 |
| Mixed precision | AMP enabled |
| GPU | RTX 5070 Ti Laptop (11.9 GB) |
| VRAM used | 0.5 GB |
| Training data | 42,200 train / 7,447 val (85/15 split) |

## Training Curve

| Epoch | Train Loss | Val Loss | Top-1 | Top-5 | LR |
|-------|-----------|----------|-------|-------|-----|
| 1 | 6.715 | 6.591 | 0.008 | 0.028 | 0.00056 |
| 5 | 5.242 | 4.419 | 0.247 | 0.487 | 0.00200 |
| 10 | 3.287 | 2.765 | 0.603 | 0.812 | 0.00190 |
| 15 | 2.902 | 2.339 | 0.699 | 0.879 | 0.00162 |
| 20 | 2.572 | 2.182 | 0.742 | 0.903 | 0.00122 |
| 25 | 2.385 | 2.120 | 0.761 | 0.911 | 0.00078 |
| 30 | 2.245 | 1.995 | 0.780 | 0.918 | 0.00038 |
| 35 | 2.209 | 1.991 | 0.792 | 0.926 | 0.00010 |
| **40** | **2.132** | **1.967** | **0.793** | **0.926** | 0.00000 |

## Best Results

| Metric | Value |
|--------|-------|
| **Best Top-1** | **79.31%** (epoch 40) |
| **Best Top-5** | **92.64%** (epoch 35) |
| Best epoch | 40 (still improving) |
| Final val loss | 1.967 |

## Overfitting Assessment

**No overfitting observed.** Both train and val loss decreased steadily. Train loss (2.13) is close to val loss (1.97). The gap is small and consistent throughout training. The model was still improving at epoch 40.

## Convergence Status

**Reached reasonable convergence.** Top-1 improvement slowed from +0.7%/epoch (E10-20) to +0.13%/epoch (E35-40). Further training (60-80 epochs) would likely yield +1-3% additional accuracy. However, 40 epochs provides a solid feature extractor for downstream finetuning.

## Suitability for Competition Finetuning

**Yes — suitable as initialization.** The 79.3% top-1 accuracy on HUST-OBC demonstrates that the backbone has learned meaningful oracle bone character features. The backbone can be frozen and a new 3,483-class head trained for competition data.

### Specific strengths:
1. ConvNeXt-Tiny has strong feature extraction capacity (28.6M params)
2. No ImageNet bias — features are purely from oracle bone domain
3. Good class coverage: 939 of 1,016 HUST classes overlap with competition
4. Lightweight: only 0.5 GB VRAM during training

### Limitations:
1. Trained from scratch (ImageNet weights unavailable due to network)
2. Only 1,016 classes vs. competition's 3,483
3. 40 epochs may not be fully converged

## Deliverable

| File | Size | Path |
|------|------|------|
| Pretrained checkpoint | 109 MB | `checkpoints/convnext_hust_pretrain.pt` |

## Next Step

Ready for Phase 3B: Competition finetuning with head expansion 1,016 → 3,483.
