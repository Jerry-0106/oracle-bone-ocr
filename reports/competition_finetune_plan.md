# Competition Finetune Plan (Phase 3A Task 2)

## Head Expansion: 1,016 → 3,483

### Current State

| Layer | HUST-OBC | Competition |
|-------|----------|-------------|
| Classes | 1,016 | 3,483 |
| Shared classes | — | 939 |
| New classes | — | 2,544 |

### Expansion Strategy

```
ConvNeXt-Tiny Backbone (frozen initially)
    ↓ feature_dim=768
    ↓
[Original Head: Linear(768, 1016)]  ← HUST pretrained weights
    ↓
[New Head: Linear(768, 3483)]       ← expanded, partially initialized
```

### Implementation

```python
# Load HUST pretrained backbone
ckpt = torch.load('checkpoints/convnext_hust_pretrain.pt')
model.load_state_dict(ckpt['model_state_dict'], strict=False)

# Save old head weights for shared classes
old_head_weight = model.classifier[2].weight.data  # [1016, 768]
old_head_bias = model.classifier[2].bias.data      # [1016]

# Build mapping: HUST class_idx → Competition class_idx
# (using Chinese character as bridge via ID_to_chinese.json)

# Create new head
new_head = nn.Linear(768, 3483)
nn.init.xavier_uniform_(new_head.weight)
nn.init.zeros_(new_head.bias)

# Copy shared class weights
for hust_idx, comp_idx in shared_mapping:
    new_head.weight[comp_idx] = old_head_weight[hust_idx]
    new_head.bias[comp_idx] = old_head_bias[hust_idx]

model.classifier[2] = new_head
```

## Shared Category Initialization (939 classes)

939 classes appear in both HUST-OBC and Competition. Strategy:

1. Build character-to-character mapping via Chinese character
2. For each shared class: copy HUST pretrained weight → Competition head position
3. These 939 classes start with strong initialization (~40 epochs of HUST training)
4. Account for 75.2% of competition training samples

## Competition-Only Initialization (2,544 classes)

2,544 classes only in Competition. Strategy:

1. **Xavier uniform initialization** for weights
2. **Zero initialization** for biases
3. These classes learn from scratch during finetuning
4. Account for 24.8% of competition training samples
5. Will benefit from frozen backbone's feature extraction

## Focal Loss Configuration

```python
criterion = FocalLoss(
    alpha=None,     # Auto-compute from class frequencies
    gamma=2.0,      # Standard focal gamma
    reduction='mean'
)
```

**Why Focal Loss**:
- 82% of classes have <10 samples
- Standard cross-entropy over-trains on head classes
- Gamma=2.0 down-weights easy (head class) examples
- Focus learning on hard (tail class) examples

## Class-Balanced Sampler Configuration

```python
# Compute sample weights
class_counts = Counter(labels)
weights = 1.0 / torch.tensor([class_counts[c] for c in range(num_classes)])
sample_weights = weights[labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset),
    replacement=True
)

# Effective frequency smoothing
weights = 1.0 / sqrt(class_counts)  # Square-root smoothing
```

### Sampling Strategy

| Method | Head Classes (>100) | Tail Classes (1-5) | Recommended |
|--------|--------------------|--------------------|-------------|
| Uniform | Oversampled | Severely undersampled | No |
| Inverse frequency | Rarely seen | Oversampled (unstable) | No |
| **Sqrt(inverse freq)** | Moderately sampled | Moderately oversampled | **Yes** |
| Effective num samples | ~30-50% reduction | ~3-5x oversampling | — |

**Recommendation**: `WeightedRandomSampler` with `weight = 1/sqrt(count)`. This balances head/tail without extreme oversampling of singletons.

## Two-Stage Training Plan

### Stage 1: Full Model (20 epochs)
- Freeze backbone for first 5 epochs
- Train only new classifier head
- Unfreeze backbone after epoch 5
- Lower LR for backbone (0.1x head LR)
- Class-balanced sampling
- Focal Loss

### Stage 2: Head Refinement (10 epochs)
- Freeze backbone
- Retrain classifier head only
- Higher LR (0.0005)
- Label smoothing = 0.05
- Standard sampling (no class balance — fine-tune on natural distribution)

## Expected Outcomes

| Metric | Target |
|--------|--------|
| Top-1 (head classes >100) | >90% |
| Top-1 (tail classes 1-5) | >25% |
| Top-5 overall | >95% |
| Top-1 overall | >80% |
