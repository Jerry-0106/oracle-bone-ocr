# 分辨率升级全面分析报告

> 生成日期: 2026-06-12
> 基线: ConvNeXt-Tiny + ArcFace @ 224×224, Top1=32.18%
> 前提: ArcFace s/m/Mixup/CutMix/Epoch/LR 调参路线已确认失败

---

## Task A: 图像读取 & 尺寸配置全景

### A1. 图像读取位置

| 环节 | 文件 | 行号 | 代码 |
|------|------|------|------|
| 训练数据读取 | `src/dataset.py` | 88 | `Image.open(img_path).convert('RGB')` |
| ArcFace 训练数据 | `_training/train_arcface.py` | 187 | `Image.open(self.img_dir / fn).convert('RGB')` |
| 推理 — 检测输入 | `scripts/infer.py` | 135 | `cv2.imread(str(img_path))` |
| 推理 — 识别输入 | `scripts/infer.py` | 218 | `cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)` → `Image.fromarray(crop_rgb)` |
| Recognizer.predict() | `src/recognizer.py` | 98 / 271 | `Image.open(image).convert('RGB')` |
| Recognizer.predict_batch() | `src/recognizer.py` | 299 | `[self.transform(img) for img in images]` |
| Detector | `src/detector.py` | 38 | `self.model(image_path, ...)` — YOLO 内部读取 |

### A2. 数据增强配置

| 环节 | 文件 | 行号 | 配置 |
|------|------|------|------|
| 训练增强 (ArcFace) | `_training/train_arcface.py` | 405-410 | `RandomResizedCrop(224, scale=(0.8,1.0))` + `RandAugment(n=2, m=9)` + Normalize |
| 训练增强 (Round1) | `_training/train_round1.py` | 258-263 | 同上 + 训练循环内 `Mixup(α=0.2)` / `CutMix(α=0.2)` |
| 训练增强 (Round1B) | `_training/train_round1b.py` | 175-179 | 同上 + Mixup/CutMix |
| 训练增强 (通用) | `src/dataset.py` | 104-133 | `RandomResizedCrop` / `RandAugment` / `ColorJitter` / `RandomRotation` |
| 通用 Val 增强 | `src/dataset.py` | 134-144 | `Resize(256)` → `CenterCrop(input_size)` → Normalize |
| YOLO 检测增强 | `_training/train_yolo11s_detector.py` | 88-98 | mosaic=0.5, mixup=0.1, copy_paste=0.1, degrees=2°, scale=0.5, no flip |
| 检测推理增强 | `scripts/infer.py` | 145-147 | `augment=True` (YOLO built-in TTA) |

### A3. Train Image Size — 全部硬编码 224

| 文件 | 行号 | 变量/值 | 说明 |
|------|------|---------|------|
| `_training/train_arcface.py` | **326** | `IMG_SIZE = 224` | ArcFace 训练，硬编码 |
| `_training/train_round1.py` | **197** | `IMG_SIZE = 224` | Round1，硬编码 |
| `_training/train_round1b.py` | **138** | `IMG_SIZE, ... = 512, 224, 128` | Round1B，硬编码 |
| `_training/train_competition_finetune.py` | **73** | `SZ=224` | Competition finetune，硬编码 |
| `_training/train_convnext_hust.py` | **20** | `SZ=224` | HUST pretrain，硬编码 |
| `scripts/train_recognition.py` | config | `input_size: 224` | 通用训练脚本，config 驱动 |

### A4. Val Image Size — 与训练一致

| 文件 | 行号 | 变换 | 实际尺寸 |
|------|------|------|----------|
| `_training/train_arcface.py` | 411-416 | `Resize(256)` → `CenterCrop(IMG_SIZE)` | **224×224** |
| `_training/train_round1.py` | 264-269 | 同上 | **224×224** |
| `_training/train_round1b.py` | 182-184 | 同上 | **224×224** |
| `_training/train_competition_finetune.py` | 79-82 | 同上 | **224×224** |
| `src/dataset.py` | 136-138 | `Resize(256)` → `CenterCrop(config["input_size"])` | config 驱动 |

### A5. Infer Image Size — 关键硬编码点

| 文件 | 行号 | 值 | 说明 |
|------|------|-----|------|
| `src/recognizer.py` ArcFaceRecognizer | **189** | `self.input_size = 224` | **硬编码！** 注释: `# ArcFace training uses 224` |
| `src/recognizer.py` ArcFaceRecognizer transform | **200-205** | `Resize(256)` → `CenterCrop(self.input_size)` | 与 val 一致 |
| `src/recognizer.py` Recognizer (legacy) | **31** | `self.config.get("input_size", 224)` | 从 checkpoint config 读取，默认 224 |
| `src/recognizer.py` Recognizer transform | **46-50** | `Resize(256)` → `CenterCrop(self.input_size)` | |
| `scripts/infer.py` 检测 | **44** | `IMGSZ = 1280` (env) | YOLO 检测分辨率 |
| `scripts/infer.py` crop 送入识别 | **218** | 原始 bbox + 3px padding | 任意尺寸，未 resize |

**完整推理尺寸链**:

```
原始图片 → YOLO11s @ 1280px → bbox crop (任意尺寸)
→ PIL Image → Resize(256) → CenterCrop(224) → ArcFace 推理
```

---

## Task B: 224 → 320 → 384 迁移方案

### B1. 当前状态确认

| 维度 | 当前尺寸 | 实现方式 |
|------|---------|---------|
| Train | **224×224** | 5 个训练脚本全部硬编码 `IMG_SIZE=224` |
| Val | **224×224** | `Resize(256) → CenterCrop(224)` |
| Infer | **224×224** | `ArcFaceRecognizer.input_size = 224` 硬编码 |

### B2. 改为 320×320 需要的代码修改

| # | 文件 | 行号 | 修改前 | 修改后 |
|---|------|------|--------|--------|
| 1 | `_training/train_arcface.py` | 326 | `IMG_SIZE = 224` | `IMG_SIZE = 320` |
| 2 | `_training/train_arcface.py` | 412 | `transforms.Resize(256)` | `transforms.Resize(365)` |
| 3 | `_training/train_arcface.py` | 405 | `RandomResizedCrop(IMG_SIZE, ...)` | 自动适配 `IMG_SIZE` (无需改) |
| 4 | `src/recognizer.py` ArcFaceRecognizer | 189 | `self.input_size = 224` | `self.input_size = 320` |
| 5 | `src/recognizer.py` ArcFaceRecognizer | 201 | `transforms.Resize(256)` | `transforms.Resize(365)` |
| 6 | `src/arcface_model.py` | 151 | `convnext_tiny(weights=None)` | **不变** — ConvNeXt 自适应输入尺寸 |
| 7 | `_training/train_round1.py` | 197 | `IMG_SIZE = 224` | `IMG_SIZE = 320` |
| 8 | `_training/train_round1b.py` | 138 | `IMG_SIZE, ... = 512, 224, 128` | `IMG_SIZE, ... = 512, 320, 128` |

> Resize 值计算: `320 × 256 / 224 ≈ 365`

### B3. 改为 384×384 需要的代码修改

| # | 文件 | 行号 | 修改前 | 修改后 |
|---|------|------|--------|--------|
| 1 | `_training/train_arcface.py` | 326 | `IMG_SIZE = 224` | `IMG_SIZE = 384` |
| 2 | `_training/train_arcface.py` | 412 | `transforms.Resize(256)` | `transforms.Resize(438)` |
| 3 | `_training/train_arcface.py` | 405 | `RandomResizedCrop(IMG_SIZE, ...)` | 自动适配 (无需改) |
| 4 | `src/recognizer.py` ArcFaceRecognizer | 189 | `self.input_size = 224` | `self.input_size = 384` |
| 5 | `src/recognizer.py` ArcFaceRecognizer | 201 | `transforms.Resize(256)` | `transforms.Resize(438)` |
| 6 | `src/arcface_model.py` | 151 | `convnext_tiny(weights=None)` | **不变** |
| 7 | `_training/train_round1.py` | 197 | `IMG_SIZE = 224` | `IMG_SIZE = 384` |
| 8 | `_training/train_round1b.py` | 138 | `IMG_SIZE, ... = 512, 224, 128` | `IMG_SIZE, ... = 512, 384, 128` |

> Resize 值计算: `384 × 256 / 224 ≈ 438`

### B4. GPU 显存影响估算

| 配置 | 像素/batch | FLOPs | 预估 VRAM (batch=128) | 建议 batch |
|------|-----------|-------|----------------------|-----------|
| 224×224 | 6.4M | 4.5G | ~8 GB | 128 (当前) |
| 320×320 | 13.1M | 9.2G | ~10 GB | 64–72 |
| 384×384 | 18.9M | 13.2G | >11.9 GB (OOM) | **32–48** |

> 硬件: RTX 5070 Ti Laptop 11.9GB VRAM
> 384 需要降 batch: 建议 48 (可能 32 更安全)

### B5. 推荐: 保存 input_size 到 Checkpoint

当前 checkpoint **不保存** `input_size`，导致推理时必须硬编码。建议在 `train_arcface.py` 保存 checkpoint 时加入:

```python
'input_size': IMG_SIZE,  # 新增
```

推理时 `src/recognizer.py:189` 改为:

```python
self.input_size = ckpt_meta.get('input_size', 224)
```

---

## Task C: 利用现有 Checkpoint 评估高分辨率推理

### C1. 技术可行性

**ConvNeXt-Tiny 架构分析**:

- 最终使用 `AdaptiveAvgPool2d` → 固定输出 768 维
- backbone **天然支持任意输入尺寸**
- 但 BN 统计量针对 224×224 特征图优化

| 方案 | 可行性 | 风险 |
|------|--------|------|
| **224 训练 → 384 验证** | ✅ 技术上可行 | BN 统计偏移，精度可能低于预期 |
| **224 训练 → 384 推理** | ✅ 技术上可行 | 同上，预测质量不可预期 |

### C2. 最小代码修改 (224-train checkpoint → 384-infer)

**仅需 2 处修改**:

1. `src/recognizer.py:189`:

```python
# 修改前
self.input_size = 224  # ArcFace training uses 224

# 修改后 (硬编码覆盖)
self.input_size = 384
```

2. `src/recognizer.py:201`:

```python
# 修改前
transforms.Resize(256),

# 修改后
transforms.Resize(438),
```

### C3. 风险评估

| 风险 | 概率 | 影响 |
|------|------|------|
| BN 统计不匹配 → 精度下降 | **中等** | ConvNeXt BN 层较多，但 adaptive pooling 后的 BN 不受影响 |
| 特征图尺寸变化 → backbone 输出分布偏移 | **低** | AdaptiveAvgPool 规范化后影响有限 |
| 推理结果不可预期 | **低** | 可以 A/B 对比 224 和 384 推理结果 |

**建议**: 直接用现有 224 checkpoint 做一次 384-infer 对比 (10 张图即可)，观测 Top1 是否变化。这只需要改 2 行代码，零训练成本。

---

## Task D: 分辨率提升 Top1 预估

### D1. 文献依据

| 基准 | 提升 | 来源 |
|------|------|------|
| ConvNeXt-T 224→384 (ImageNet) | +1.2–1.8pp | ConvNeXt paper (Liu et al., 2022) |
| DeiT/ViT 224→384 | +1.5–2.5pp | Touvron et al., 2021 |
| 细粒度分类 (CUB-200) 224→448 | +3–5pp | 常见细粒度文献 |
| OCR/文字识别 增大输入 | +2–5pp | 取决于原始字符像素密度 |

### D2. 甲骨文场景分析

**当前像素密度**:

```
YOLO bbox crop (任意尺寸) → Resize(256) → CenterCrop(224)
  ├── 小字 (bbox <30px):  极端上采样，有效信息极少
  ├── 中字 (bbox 30-100px): 上采样到 224，丢失部分笔画细节
  └── 大字 (bbox >100px):  可能 OK
```

**像素量对比**:

| 尺寸 | 总像素 | 相对 224 | 笔画可表达性 |
|------|--------|---------|-------------|
| 224×224 | 49,152 | 1.0× | 笔画宽度 2-5px，细节不足 |
| 320×320 | 102,400 | 2.1× | 笔画宽度 3-7px，中等改善 |
| 384×384 | 147,456 | 3.0× | 笔画宽度 4-9px，显著改善 |
| 512×512 | 262,144 | 5.3× | 笔画宽度 5-12px，但感受野可能饱和 |

### D3. 预估

| 方案 | 保守估计 | 乐观估计 | 核心依据 |
|------|---------|---------|---------|
| **224 → 320** | **+1.5pp** (33.7%) | **+3.0pp** (35.2%) | 2.1× 像素，中等细节恢复 |
| **224 → 384** | **+3.0pp** (35.2%) | **+5.0pp** (37.2%) | 3.0× 像素，显著细节恢复 |
| **224 → 512** | **+4.0pp** (36.2%) | **+7.0pp** (39.2%) | 5.3× 像素，但 ConvNeXt-T 感受野可能饱和 |

### D4. 上限分析

**为什么不会更高**: ConvNeXt-Tiny 容量上限 (~28M params)。即使无限分辨率，有限容量的 backbone 无法完美区分 3,483 个类。当前 intra-class cosine dist 和 inter-class gap 问题说明:

> **backbone 特征区分度才是根本瓶颈，分辨率只是其中一个因素。**

384 主要解决的是"输入信息损失"问题，但不能解决"模型容量不足"问题。

---

## Task E: 比赛 6 小时容器限制分析

### E1. 当前推理耗时估算

基于 `scripts/infer.py` 的 pipeline 分析:

```
单张图片流程:
  1. cv2.imread (I/O)
  2. YOLO11s @ 1280px detection + TTA (augment=True)
  3. bbox post-processing (clamp, filter, shrink)
  4. 逐 crop: BGR→RGB→PIL Image
  5. ArcFace batch recognition (batch=32)
  6. 输出 assembly
```

**每张图耗时估算 (GPU)**:

| 阶段 | 耗时 | 说明 |
|------|------|------|
| YOLO @ 1280px + TTA | ~80–150ms | 取决于图片尺寸和 bbox 数量 |
| 后处理 | ~5–10ms | 纯 CPU |
| 识别 (50 crops, batch=32) | ~30–60ms | 2 个 batch |
| I/O + 其他 | ~10–20ms | |
| **合计 per image** | **~150–250ms** | |

**总耗时预估**:

| 测试集规模 | 当前耗时 |
|-----------|---------|
| 2,000 张 | **5–8 分钟** |
| 5,000 张 | **13–21 分钟** |
| 10,000 张 | **25–42 分钟** |
| 20,000 张 | **50–83 分钟** |

### E2. 384 推理的额外成本

ConvNeXt-Tiny FLOPs 比例:
- 224: 4.5G FLOPs
- 384: 13.2G FLOPs (**~2.9×**)

识别阶段 @ 384:
- 原来 30–60ms → **90–175ms**
- 总 per image @ 384: **~200–350ms**

| 测试集规模 | 384 总耗时 |
|-----------|-----------|
| 2,000 张 | **7–12 分钟** |
| 5,000 张 | **17–29 分钟** |
| 10,000 张 | **33–58 分钟** |
| 20,000 张 | **67–117 分钟** |

### E3. TTA 额外成本

识别 TTA = 水平翻转 + 原图 = 2× forward:

| 方案 | 识别耗时 (50 crops) | 总 per image |
|------|-------------------|-------------|
| 224 + TTA | ~60–120ms | ~200–310ms |
| 384 + TTA | ~180–350ms | ~300–530ms |

### E4. 多尺度推理额外成本

Multi-scale = [224, 320, 384]:

| 方案 | 识别耗时 | 总 per image |
|------|---------|-------------|
| 224 × 3 scales | ~90–180ms | ~230–370ms |
| 384 × 3 scales | ~270–525ms | ~390–715ms |

### E5. 所有组合方案 6 小时安全性

| 方案 | 单张耗时 | 5,000 张 | 10,000 张 | 20,000 张 | 6h 内安全? |
|------|----------|---------|----------|----------|-----------|
| **当前: 224 + TTA** | ~200ms | ~17min | ~33min | ~67min | ✅ 安全 |
| **384 only** | ~280ms | ~23min | ~47min | ~93min | ✅ 安全 |
| **384 + TTA** | ~420ms | ~35min | ~70min | ~140min | ✅ 安全 |
| **384 + TTA + Multi-scale(3)** | ~1.2s | ~100min | ~200min | ~400min | ✅ 安全 |
| **384 + TTA + Multi-scale(5)** | ~2.0s | ~167min | ~333min | ~667min | ⚠️ 临界 (20k) |
| **检测 Multi-scale + 384 + TTA** | ~2.5s | ~208min | ~417min | ~833min | ⚠️ 临界 (15k+) |

> 测试集规模预估: 比赛通常 2,000–8,000 张

### E6. 结论

> **即使最激进方案 (检测多尺度 + 384 + 识别 TTA)，在 5,000–10,000 张测试集上 6 小时内仍然安全。仅在超过 15,000 张图片时需要谨慎。**

---

## 【优先级 1】是否应该立即做 384 实验？

### 🟢 是。强烈建议立即执行。

**五大论据**:

**1. 技术可行性 = 100%**

- ConvNeXt-Tiny 的 AdaptiveAvgPool2d 天然支持任意分辨率
- 仅需修改 5–8 处代码 (详见 Task B)
- 不涉及架构变更，checkpoint 格式完全兼容
- 可以 hot-start 从现有 `convnext_arcface_best.pt` 继续训练

**2. 预期收益明确**

- 保守 +3.0pp Top1 (32.18% → ~35.2%)
- 乐观 +5.0pp Top1 (32.18% → ~37.2%)
- 甲骨文字符的笔画细节是**明确的像素级信息损失**，高分辨率直接恢复这些信息

**3. 成本极低**

- 代码修改: ~10 行
- 训练时间: batch=48, 384px, 60 epochs → 估计 3–4 小时 (RTX 5070 Ti)
- 推理时间在 6 小时内安全
- 可以先用 224 checkpoint 做一次 384-infer 探测 (2 行代码，零训练)

**4. 时机完美**

- ✅ ArcFace s/m/Mixup/CutMix 调参路线已确认失败
- ✅ 60 epochs 已收敛，不需要更长的训练
- ✅ 分辨率是当前唯一未探索的"低成本高回报"路线
- ✅ 已有成熟的 checkpoint 可以 hot-start

**5. 风险可控**

- 384 不 work → 回退成本为 0，保留 224 checkpoint
- BN 统计偏移 → 可以 warmup 几个 epoch 重新校准
- 显存不足 → 降 batch 即可，训练时间仍在可控范围

### 实验设计

```
Phase 1 — 快速探测 (零训练成本, 5 分钟):
  ├── 修改 src/recognizer.py input_size=384
  ├── 用现有 224 checkpoint 推理 100 张 val 图
  ├── 对比 224 vs 384 推理 Top1
  └── 如果 Top1 有提升 → 确认分辨率瓶颈，进入 Phase 2

Phase 2 — 384 训练 (3–4 小时):
  ├── 修改 train_arcface.py: IMG_SIZE=384, batch=48
  ├── Hot-start 从 convnext_arcface_best.pt
  ├── 10 epoch 快速探测
  │   ├── Top1 明显提升 → 继续到 60 epoch
  │   └── Top1 无变化 → 终止，诊断原因
  └── 保存 input_size 到 checkpoint

Phase 3 — 如果 384 有效 (1–2 天):
  ├── 384 + Strong ArcFace (s=48, m=0.5) 再试
  ├── 384 + Mixup/CutMix 再试
  └── 384 + ConvNeXt-Small
```

---

## 【优先级 2】如果 384 有效，再做什么？

### 路线图

```
384 有效 (Top1 ≥ 35%) →
    │
    ├── 【立即】Phase 1: 384 + ArcFace 调参再验证
    │    ├── s=48, m=0.5 (之前无效可能是特征质量不够)
    │    ├── s=64, m=0.3 (更温和的 margin)
    │    └── 10 epoch 快速探测即可
    │
    ├── 【短期】Phase 2: 512×512 探测
    │    ├── 5.3× 像素 vs 224
    │    ├── 风险: ConvNeXt-Tiny 感受野可能饱和
    │    ├── batch 需要降到 20–24
    │    └── 5 epoch 足够判断趋势
    │
    ├── 【中期】Phase 3: Backbone 升级
    │    ├── ConvNeXt-Small + 384: 50M params (+2–3pp 预估)
    │    ├── ConvNeXt-Base + 384: 89M params (+3–5pp 预估)
    │    ├── 注意显存: Small@384 batch~24, Base@384 batch~12
    │    └── 需要确认 GPU 显存是否够用
    │
    ├── 【中期】Phase 4: DINOv2-L 微调
    │    └── 见优先级 3 分析
    │
    └── 【长期】Phase 5: Ensemble
         ├── ConvNeXt-S@384 + DINOv2-L@224
         ├── 预计额外 +2–4pp over best single model
         └── 需要验证推理时间在 6 小时内
```

### 如果 384 无效 (Top1 无明显提升)

```
384 无效 →
    │
    ├── 【诊断】特征空间分析
    │    ├── intra/inter gap 是否改善?
    │    ├── 如果 gap 没变 → 瓶颈在 backbone 容量，不是分辨率
    │    └── 如果 gap 改善但 Top1 没变 → 问题在其他地方 (数据/loss)
    │
    ├── 【方向 A】升级 Backbone
    │    ├── ConvNeXt-Small/Base (更大的特征容量)
    │    └── DINOv2-L (见优先级 3)
    │
    ├── 【方向 B】数据侧改进
    │    ├── 更多训练数据 (competition training set)
    │    ├── 合成数据 / copy-paste 增强
    │    ├── 针对 tail classes 的数据增强
    │    └── 更好的 class-balanced sampling
    │
    └── 【方向 C】Loss 改进
         ├── Focal Loss (针对 hard examples)
         ├── Sub-center ArcFace (针对噪声)
         └── CurricularFace (adaptive margin)
```

---

## 【优先级 3】DINOv2 是否值得投入？

### 🟡 中等优先级。384 实验完成后再决定。

### 现有数据

| 指标 | DINOv2-L k-NN (zero-shot) | ArcFace ConvNeXt-T (trained) | Δ |
|------|--------------------------|------------------------------|-----|
| **Top1** | **9.81%** | **32.18%** | **-22.37pp** |
| **Top5** | 20.79% | 44.96% | -24.17pp |
| Head (>50 train) | 12.76% | — | — |
| Medium (10-50) | 3.88% | — | — |
| Tail (≤5) | 2.42% | — | — |
| Feature dim | 1024 | 512 | 2× |
| Params | 307M | 28M | 11× |
| Feature extraction (60k imgs) | ~1100s (18 min) | — | — |

> DINOv2 数据来自 `reports/dinov2_knn_eval_results.json`

### 分析

**为什么 DINOv2 k-NN 只有 9.81%? 这不代表 DINOv2 不好:**

1. **k-NN 是 zero-shot，没有针对任务微调** — 这恰恰说明 DINOv2 的通用特征需要 task-specific fine-tuning
2. **k-NN 不考虑类间关系** — ArcFace 的 angular margin 专门优化类间可分性
3. **k-NN 对类别不平衡极度敏感** — 长尾分布下，k-NN 天然偏向高频类
4. **类比**: ImageNet 上 DINOv2 k-NN ~84%, fine-tuned ~87%. 对于甲骨文这种特殊域，fine-tune 的提升幅度更大

**DINOv2 的真正潜力**:

| 配置 | 预估 Top1 | 依据 |
|------|----------|------|
| DINOv2-L k-NN (当前) | 9.81% | 实测 |
| DINOv2-L + Linear Probe | ~25–35% | 线性分类器，1 epoch |
| DINOv2-L + ArcFace head + 微调 10 ep | ~35–45% | head tuning |
| DINOv2-L + ArcFace + full finetune 30 ep | **~45–55%** | 全模型微调 |

**DINOv2 的成本与风险**:

| 项目 | 详情 |
|------|------|
| 模型文件 | ~1.2GB (vs ConvNeXt-T ~110MB) |
| Docker 镜像 | 需增加 `transformers` 依赖 (~500MB) |
| 推理速度 | 约 3–5× ConvNeXt-T 的 FLOPs |
| 训练显存 | batch=8–16 (384 可能需要 gradient checkpointing) |
| 训练时间 | 30 epochs full finetune: 6–12 小时 |

### 决策树

```
384 ConvNeXt-Tiny 实验结果:

  Top1 ≥ 38%:
    → DINOv2 优先级降低
    → 先深化 ConvNeXt 路线 (Small/Base + 更高分辨率)
    → DINOv2 作为备选方案

  Top1 35–38%:
    → DINOv2 值得做对比实验
    → 先 Linear Probe (1 小时) 探测 DINOv2 特征质量
    → 如果 Linear Probe >30% → DINOv2 微调大概率有效

  Top1 < 35%:
    → ConvNeXt-Tiny 容量已到上限
    → DINOv2 成为高优先级
    → 立即启动 DINOv2-L + ArcFace head 微调

  DINOv2 Linear Probe >30%:
    → 全量投入 DINOv2 微调
    → 30 epochs full finetune
    → 目标 Top1 >45%
```

### 当前建议

> **不要同时做 384 和 DINOv2。先做 384 实验 (1-2 天出结果)，然后根据结果决定 DINOv2 的优先级。**

具体路线:

1. **本周**: 384 ConvNeXt-Tiny 训练
2. **下周**: 根据 384 结果决定
   - 384 效果好 → ConvNeXt-Small + 384
   - 384 效果一般 → 启动 DINOv2 微调
3. **DINOv2 启动条件**: 384 结果 < 35% 或 需要 >40% 才能竞赛有竞争力

---

## 总结

| 优先级 | 行动 | 预期收益 | 代码修改 | 时间成本 | 风险 |
|--------|------|---------|---------|---------|------|
| **P1** | **384×384 训练** | **+3–5pp Top1** | ~10 行 | 1–2 天 | 低 |
| P1a | 224 ckpt → 384 推理探测 | 验证分辨率瓶颈 | 2 行 | 5 分钟 | 无 |
| P2a | 384 + Strong ArcFace 再试 | +0–2pp | 0 行 | 几小时 | 低 |
| P2b | 512×512 探测 | +1–3pp | 3 行 | 1 天 | 中 (显存) |
| P2c | ConvNeXt-Small + 384 | +2–3pp | 5 行 | 1–2 天 | 低 |
| P3 | DINOv2-L 微调 + ArcFace | +10–20pp | ~50 行 | 2–3 天 | 中 (时间) |
| P4 | Model Ensemble | +2–4pp | ~20 行 | 1 天 | 低 |

**一句话**: 先跑 384，1-2 天见分晓，然后决定下一步。

---

## 附录: Checkpoint 跨分辨率兼容性验证

### 验证问题

> 将 IMG_SIZE 从 224 改为 384 后，现有的 `convnext_arcface_best.pt` (224 训练) 能否直接 hot-start 继续训练？

### 验证结论: ✅ 完全兼容，无需任何模型架构修改

### 逐层验证

#### 1. 模型构建与 input_size 无关

**`build_arcface_model()` 参数** (`src/arcface_model.py:135`):

```python
def build_arcface_model(num_classes=3483, embedding_dim=512, s=30.0, m=0.5):
    backbone = models.convnext_tiny(weights=None)
    backbone.classifier[2] = nn.Identity()
    model = ArcFaceModel(backbone=backbone, backbone_dim=768, ...)
    return model
```

**没有任何 `input_size` 参数。** 架构完全与分辨率解耦。

#### 2. ConvNeXt-Tiny Backbone: LayerNorm, 非 BatchNorm

这是关键。ConvNeXt 使用 **LayerNorm** 而非 BatchNorm：

```
torchvision ConvNeXt 实现:
  Permute([0, 2, 3, 1])       # (B, C, H, W) → (B, H, W, C)
  → nn.LayerNorm(C)            # 跨通道归一化，每个空间位置独立
  → Permute([0, 3, 1, 2])      # (B, H, W, C) → (B, C, H, W)
```

**LayerNorm vs BatchNorm 的关键区别**:

| 属性 | BatchNorm | LayerNorm (ConvNeXt) |
|------|-----------|---------------------|
| 归一化维度 | 跨 batch, 跨空间 | 跨通道 (C) |
| running_mean/var | **有** (分辨率相关) | **无** |
| 推理时行为 | 使用 running stats | 实时计算 (与训练一致) |
| 对分辨率敏感性 | **敏感** (特征图尺寸变化 → 统计偏移) | **不敏感** |

**结论**: ConvNeXt 的 LayerNorm 层在 224 和 384 下行为完全一致，不存在 BN 统计不匹配问题。

#### 3. Backbone 卷积层: 空间不变

ConvNeXt-Tiny 的所有卷积层:
- Stem: `Conv2d(3, 96, kernel=4, stride=4)` — 核 (96, 3, 4, 4) 与空间无关
- Depthwise: `Conv2d(C, C, kernel=7, groups=C)` — 核 (C, 1, 7, 7) 与空间无关
- Pointwise: `Conv2d(C, 4C, kernel=1)` — 核 (4C, C, 1, 1) 与空间无关
- Downsample: `Conv2d(C, 2C, kernel=2, stride=2)` — 与空间无关

**所有权重矩阵的形状与输入分辨率无关。**

#### 4. AdaptiveAvgPool2d: 分辨率归一化

```python
nn.AdaptiveAvgPool2d((1, 1))  # 最终层 → 总是输出 (B, 768, 1, 1)
```

这是 ConvNeXt 分类头之前的标准操作。无论输入是 224×224 还是 384×384，输出始终是 (B, 768)。

#### 5. Embedding Head: 固定维度

```python
nn.BatchNorm1d(768)     # 输入永远 768 维 (来自 AdaptiveAvgPool)
nn.Linear(768, 512)     # 固定维度矩阵乘法
nn.BatchNorm1d(512)     # 固定维度
```

这里的 BN1d 归一化的是 768/512 维特征向量（跨 batch），与空间分辨率无关。

#### 6. ArcFace Head: 固定维度

```python
ArcMarginProduct(512, 3483)  # weight: (3483, 512)
```

与分辨率无关。

#### 7. Checkpoint 格式验证

**保存** (`train_arcface.py:656-669`):

```python
torch.save({
    'model_state_dict': model.state_dict(),  # 所有权重
    'num_classes': NUM_CLASSES,
    'embedding_dim': EMBEDDING_DIM,
    'arc_s': ARC_S,
    'arc_m': ARC_M,
    ...
}, 'convnext_arcface_best.pt')
```

**不存在 `input_size` 字段** — 因为 checkpoint 本身与分辨率无关。

**加载** (`src/recognizer.py:169-184`):

```python
ckpt_meta = torch.load(model_path, ...)
num_classes = ckpt_meta.get('num_classes', 3483)
embedding_dim = ckpt_meta.get('embedding_dim', 512)
arc_s = ckpt_meta.get('arc_s', 30.0)
arc_m = ckpt_meta.get('arc_m', 0.5)

model = build_arcface_model(num_classes=num_classes, embedding_dim=embedding_dim,
                             s=arc_s, m=arc_m)
model, meta = load_arcface_checkpoint(model, model_path, map_loc)
# → model.load_state_dict(ckpt['model_state_dict'])
```

**`load_state_dict` 只要求 key 名和 shape 匹配。** 更改 IMG_SIZE 不会改变任何层的 shape，因此加载必然成功。

### 实际需要修改的内容

分辨率切换时，**唯一需要改的是预处理 transform**:

| 位置 | 224 配置 | 384 配置 |
|------|---------|---------|
| `transform.Resize` | `256` | `438` (≈384×256/224) |
| `transform.CenterCrop` | `224` | `384` |
| 训练 `IMG_SIZE` 变量 | `224` | `384` |
| 推理 `self.input_size` | `224` | `384` |

**模型架构: 零修改。**
**Checkpoint 格式: 零修改。**
**权重加载: 必然成功。**

### 跨分辨率 Hot-Start 方案

```
Step 1: 修改 train_arcface.py
  IMG_SIZE = 384
  val_tfm: Resize(438), CenterCrop(384)

Step 2: 加载现有 224 checkpoint (无需任何转换)
  model = build_arcface_model(...)    # 架构与之前完全一样
  ckpt = torch.load('convnext_arcface_best.pt')
  model.load_state_dict(ckpt['model_state_dict'])  # ✅ 必然成功

Step 3: 继续训练
  # optimizer 和 scheduler 可以从头开始 (warmup + cosine)
  # 或者也加载 optimizer state 继续 fine-tune

Step 4: 保存新的 384 checkpoint (可选加入 input_size)
  torch.save({..., 'input_size': 384}, 'convnext_arcface_384_best.pt')
```

### 唯一的理论风险

LayerNorm 在 224×224 训练时学习到的特征表示，在 384×384 下首次 forward 时可能产生略微不同的激活值分布（因为空间尺寸变化导致池化前的特征图模式不同）。但:

1. **这不影响权重加载** — weight shape 完全匹配
2. **训练 1-2 epoch 内会自然适应** — 梯度会调整权重以适应新的空间分布
3. **这本质上等同于 fine-tune 的标准操作** — 从 ImageNet 224 预训练权重 fine-tune 到目标域是常规做法

### 最终确认

| 检查项 | 状态 |
|--------|------|
| 模型架构依赖 input_size? | ❌ 不依赖 |
| Conv 层权重 shape 依赖分辨率? | ❌ 不依赖 |
| LayerNorm 有 running stats? | ❌ 没有 (LN 实时计算) |
| AdaptiveAvgPool 输出固定? | ✅ 固定 768 维 |
| Embedding head 依赖分辨率? | ❌ 不依赖 |
| ArcFace head 依赖分辨率? | ❌ 不依赖 |
| load_state_dict 会失败? | ❌ 不会 (key 名和 shape 全匹配) |
| 唯一需要修改的 | Transform 的 Resize/CenterCrop 值 |
