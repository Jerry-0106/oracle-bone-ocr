# Recognition Training Plan (Phase 2D)

## Data Summary

| Property | Value |
|----------|-------|
| Competition classes | 3,483 (after removing 935 garbage labels) |
| Competition samples | 60,072 |
| HUST-OBC classes | 1,781 (1,016 class directories) |
| HUST-OBC overlap with Competition | 939 classes (75.2% of competition samples) |
| HUST-OBC pretrained model | `recognizer.pt` (EfficientNet-B0, ~84% acc on HUST-OBC) |

## Scheme A: ConvNeXt-Tiny

| Property | Value |
|----------|-------|
| Parameters | 28.6M |
| Input size | 224×224 |
| Batch size (est.) | 128 (at 224px, much smaller than detection input) |
| GPU memory (est.) | ~4-6 GB at batch=128 |
| Training time (est.) | ~3-4 hours for 50 epochs |
| Pretrained | ImageNet-1K (strong general features) |
| Pros | Modern architecture, better feature extraction, handles multi-scale well |
| Cons | Larger model, no HUST-OBC pretrained weights available |
| Strategy | Train from ImageNet pretrained; Stage 1: all data → Stage 2: freeze backbone, retrain head with class-balanced sampling |

## Scheme B: EfficientNet-B2

| Property | Value |
|----------|-------|
| Parameters | 9.1M |
| Input size | 260×260 (native) or 224×224 |
| Batch size (est.) | 128-256 |
| GPU memory (est.) | ~3-5 GB at batch=128 |
| Training time (est.) | ~2-3 hours for 50 epochs |
| Pretrained | ImageNet-1K |
| HUST-OBC pretrained | `recognizer.pt` is EfficientNet-B0, can transfer backbone partially |
| Pros | Smaller, faster, EfficientNet family proven on this task |
| Cons | B2 vs B0 weight mismatch for HUST-OBC transfer, smaller capacity than ConvNeXt |

## Comparison

| Metric | ConvNeXt-Tiny | EfficientNet-B2 | Winner |
|--------|--------------|-----------------|--------|
| Parameters | 28.6M | 9.1M | Eff-B2 (lighter) |
| Training speed | Slower | Faster | Eff-B2 |
| GPU memory | ~6 GB | ~4 GB | Eff-B2 |
| HUST-OBC transfer | From scratch | Partial (B0→B2) | ConvNeXt (less confusing) |
| Capacity for 3,483 classes | Better | Adequate | ConvNeXt |
| Expected accuracy | ~88-92% | ~85-88% | ConvNeXt |

## Recommendation: ConvNeXt-Tiny

**Rationale**:
1. **3,483 classes** is a large classification space. ConvNeXt's higher capacity (28.6M vs 9.1M) provides headroom.
2. HUST-OBC `recognizer.pt` is EfficientNet-B0 — weights cannot transfer cleanly to B2 (different architecture dimensions). ConvNeXt avoids this false hope.
3. The long-tail distribution (49% singletons) benefits from stronger feature extraction. ConvNeXt's modern design handles fine-grained features better.
4. At 224×224 input with batch=128, ConvNeXt-Tiny fits comfortably in 11.9 GB VRAM.
5. Training can use the HUST-OBC data as auxiliary training data (not as weight initialization) by adding shared classes as extra samples.

## Training Strategy

### Stage 1: Full Training with Class-Balanced Sampling (50 epochs)
- Model: ConvNeXt-Tiny (ImageNet pretrained)
- Data: All 60,072 competition samples + HUST-OBC samples for 939 shared classes
- Sampling: WeightedRandomSampler with class weights ∝ 1/sqrt(count)
- Loss: Focal Loss (γ=2) to focus on hard/rare classes
- Optimizer: AdamW (lr=0.001, weight_decay=0.0001)
- Schedule: 5-epoch warmup + cosine annealing
- Augmentation: RandAugment(n=2, m=9), RandomResizedCrop, ColorJitter

### Stage 2: Head Refinement (20 epochs)
- Freeze backbone
- Retrain classifier head with heavy tail oversampling
- Lower LR (0.0001)
- Label smoothing = 0.05

### Data Cleaning Required Before Training
1. Remove 935 garbage labels (ZHFD-* metadata strings and "□")
2. Regenerate `labels.csv` with only valid CJK characters
3. Create train/val split (85/15, stratified by class where possible)

## Expected Outcomes

| Metric | Target |
|--------|--------|
| Top-1 accuracy | 88-92% |
| Top-5 accuracy | 95-98% |
| Per-class accuracy (head, >100 samples) | >95% |
| Per-class accuracy (tail, 1 sample) | 30-50% (acceptable with Focal Loss) |

Note: Tail class accuracy will remain low due to the extreme long-tail. The goal is to maximize head class accuracy while achieving reasonable tail performance through class-balanced sampling and Focal Loss.
