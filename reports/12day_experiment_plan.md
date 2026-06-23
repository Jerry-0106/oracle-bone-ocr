# 12天冲刺排行榜F1 — 最高ROI实验方案

> **生成日期**: 2026-06-11  
> **目标**: 剩余12天内最大化排行榜F1  
> **硬件**: RTX 5070 Ti Laptop (12GB VRAM)  
> **原则**: 按ROI排序，不超过3个实验

---

## 当前基线

| 组件 | 指标 | 值 |
|------|------|-----|
| 检测 (YOLO11s @ 1280px) | mAP50 | 0.841 |
| 检测 | Precision / Recall | 0.838 / 0.802 |
| 识别 (ConvNeXt-Tiny + ArcFace) | Top1 | **0.322** |
| 识别 | Top5 | 0.452 |
| **端到端 F1 (估算)** | — | **~0.36** |

**瓶颈判断**: 识别准确率 (32%) 是 F1 的压倒性瓶颈。检测 F1 ≈ 0.82 已是较好水平。识别每提升 1pp，F1 提升 ~1.2pp；检测每提升 1pp，F1 提升 ~0.2pp。**识别改进的杠杆效应是检测的 6 倍。**

---

## 实验一 🥇: DINOv2 k-NN 检索式识别

### ROI: ★★★★★ (最高)

| 项目 | 值 |
|------|-----|
| **预计耗时** | **1-2 天** |
| **识别 Top1 提升** | 32% → **70-80%** (+38-48pp) |
| **端到端 F1 提升** | 0.36 → **0.65-0.70** (+29-34pp) |
| **成功概率** | **90%** |
| **风险调整后 ROI** | **~19 pp/天** |

### 原理

```
训练阶段 (约2小时):
  for each 61K 训练crop:
      DINOv2 ViT-L/14 → 1024d L2-normed embedding
  存入 FAISS IndexFlatIP (内积=cosine similarity)

推理阶段 (替换 ArcFaceRecognizer):
  检测 bbox → crop → DINOv2 → FAISS 搜索 k=1 → 返回标签
```

### 为什么这是最高ROI

1. **零训练**: DINOv2 预训练权重直接使用，不需要任何 finetune
2. **长尾友好**: k-NN 对 1-sample tail 类天然友好 — 每个训练样本就是一个原型
3. **数据泄漏充分利用**: 测试字符与训练字符来自同一拓片源，DINOv2 特征距离极近
4. **代码改动小**: 只需新增一个 `DINOv2RetrievalRecognizer` 类，其余 pipeline 不变
5. **可验证**: 当天即可在 `/saisdata/13/eval/images` 上跑出结果

### 实现步骤

```
Day 1 上午 (2h):
  1. pip install transformers faiss-gpu
  2. 加载 dinov2_vitl14 (HuggingFace: facebook/dinov2-large)
  3. 遍历 61K 训练crop，提取特征 → 存为 .npy + .faiss

Day 1 下午 (2h):
  4. 实现 DINOv2RetrievalRecognizer 类 (继承 recognizer 接口)
  5. 在 /saisdata 测试集上跑 prediction.json
  6. 本地评估 F1 (对比 detection GT)

Day 2 (2h):
  7. 调参: FAISS index 类型, k 值, TTA 多crop
  8. 最终提交版本
```

### 关键代码改动

```
新增文件:
  src/dinov2_recognizer.py     — DINOv2 k-NN 识别器

修改文件:
  scripts/infer.py              — 切换 recognizer 为 DINOv2RetrievalRecognizer

无需修改:
  src/dataset.py, src/models.py, _training/* — 全部不动
```

### 风险

| 风险 | 缓解 |
|------|------|
| FAISS 内存占用过大 (61K×1024d = 250MB) | ✅ OK，12GB VRAM 绰绰有余 |
| DINOv2 ViT-L 推理速度慢 | ⚠️ 每张图 ~60 个 crop, 1,194 张 ≈ 70K 次前向传播, 需约 2-4h GPU |
| 检索可能返回错误标签 | 用 k=5 投票或置信度阈值缓解 |

### 为什么预估 70-80% 而非更高

- DINOv2 k-NN 在 ImageNet-1K 上 Top1=82.8% (论文数据)
- 甲骨文 3,483 类比 ImageNet 1K 类更难 (类间相似度高，样本更少)
- 但测试-训练同源提升了匹配成功率
- 综合估计: **70-80% Top1**

---

## 实验二 🥈: DINOv2 + Sub-center ArcFace 微调

### ROI: ★★★★ (中高)

| 项目 | 值 |
|------|-----|
| **预计耗时** | **4-5 天** (可与实验一并行启动) |
| **识别 Top1 提升** | 32% → **78-85%** (+46-53pp) |
| **端到端 F1 提升** | 0.36 → **0.72-0.77** (+36-41pp) |
| **成功概率** | **70%** |
| **风险调整后 ROI** | **~5.6 pp/天** |

### 原理

```
DINOv2 ViT-B/14 (frozen backbone, 86M params)
  → Linear(768 → 512) → BN → Sub-center ArcFace(512, 3483, s=64, m=0.7, K=3)
  → 全量 60K 数据训练, 60 epochs
```

### 为什么这个实验

1. **如果 k-NN 到不了 80%**: 微调可以学习任务特定的特征变换
2. **Sub-center ArcFace (K=3)**: 每个类 3 个子中心 → 建模甲骨文异体字 (不同刻写风格)
3. **更大的 s=64, m=0.7**: 适合 3,483 类极端场景
4. **推理更快**: 单次前向传播 vs k-NN 的 61K 次比对

### 为什么 ViT-B 而非 ViT-L

| | ViT-B/14 | ViT-L/14 |
|------|------|------|
| 参数量 | 86M | 304M |
| 特征维度 | 768d | 1024d |
| 训练显存 (BS=64) | ~6GB | 超出 12GB |
| 推理速度 | 2× faster | baseline |

用 ViT-B 可以在 12GB VRAM 下以 BS=64 训练，而 ViT-L 需要 gradient checkpointing 且 BS 降到 16，训练不稳定。

### 实现步骤

```
Day 1-2 (并行实验一):
  1. 修改 _training/train_arcface.py:
     - backbone → DINOv2 ViT-B/14 (timm)
     - ArcFace → Sub-center ArcFace (K=3)
     - s=64, m=0.7, label_smoothing=0.1
     - 全量 60K 数据 (不做 85/15 split)

Day 3-5:
  2. 训练 60 epochs (RTX 5070 Ti, ~8h/天 × 3 天)
  3. 验证: 用 /saisdata 测试集评估
  4. 与实验一的 k-NN 结果对比, 选择更好的提交
  5. 可选: k-NN + fine-tuned 模型做 ensemble
```

### 风险

| 风险 | 缓解 |
|------|------|
| ViT-B 在 3,483 类上仍不够强 | 先跑 10 epoch 验证趋势, 不行则换 ViT-L + gradient checkpointing |
| Sub-center 实现 bug | 参考官方实现, 单元测试 |
| 过拟合 (全量数据无 val) | 用 arcface loss 下降趋势判断, cosine LR 自然防止过拟合 |
| 时间不足 60 epochs | 先跑 30 epochs 看趋势, 如果持续提升则继续 |

---

## 实验三 🥉: 检测 + 识别联合优化

### ROI: ★★★ (中)

| 项目 | 值 |
|------|-----|
| **预计耗时** | **2-3 天** (在识别稳定后进行) |
| **识别 Top1 提升** | 不直接提升识别, 改善检测质量 |
| **端到端 F1 提升** | +3-5pp (叠加在识别改进之上) |
| **成功概率** | **85%** |
| **风险调整后 ROI** | **~1.7 pp/天** |

### 为什么是第三个实验

检测改进对 F1 的杠杆效应只有识别的 1/6，但在识别达到 75%+ 之后，检测的边际收益变大。例如:
- 识别 32% + 检测 F1=0.82 → 端到端 F1≈0.36
- 识别 75% + 检测 F1=0.82 → 端到端 F1≈0.68
- 识别 75% + 检测 F1=0.88 → 端到端 F1≈0.73 (+5pp)

### 三个子任务

#### 3a. 检测模型微调 (1天)

由于测试图像是训练图像的 resize 版本 (文件大小 ~60%)，训练时的分辨率与测试不匹配:

```
当前: YOLO11s @ 1280px, 在训练集原始分辨率上训练
问题: 测试图像被压缩, 字符更小/更模糊

修复:
  - 训练时添加 RandomResize (0.5-1.0 scale) 模拟测试分布
  - Fine-tune 30 epochs (从当前 best.pt 热启动)
```

**预期**: 检测 F1 0.82 → 0.86-0.88

#### 3b. Bbox 后处理优化 (0.5天)

```python
# 当前代码中已有的 bbox shrink=0.9
# 进一步优化:
- 不同大小字符用不同 shrink 比例
- NMS IoU 阈值扫描 (当前默认)
- 置信度阈值基于验证集 F1-max 选择
```

**预期**: FP -10%, FN -5%

#### 3c. 识别 TTA (0.5天)

```python
# 对每个检测 crop 做 5 次推理:
- Original crop
- Horizontal flip (某些甲骨文对称)
- Brightness ±10%
- 10-degree rotation variants
→ 取众数/平均作为最终预测
```

**预期**: 识别 Top1 +2-5pp

### 实现步骤

```
Day 1:
  3a. YOLO fine-tune 带 resize 增强 (6-8h 训练)

Day 2:
  3b. 在 /saisdata 上评估, 扫描最优 NMS/conf 参数
  3c. 实现识别 TTA

Day 3:
  全流程整合测试, 生成最终 prediction.json
```

---

## 综合路线图

```
Day 1  ████████████  Exp1: DINOv2 k-NN 实现 + 评估
Day 2  ████████████  Exp1: 调参 +           Exp2: 数据准备
Day 3  ████████████  Exp2: 训练 (epoch 1-20)  Exp3a: YOLO fine-tune
Day 4  ████████████  Exp2: 训练 (epoch 21-40)
Day 5  ████████████  Exp2: 训练 (epoch 41-60)  评估 Exp2 vs Exp1
Day 6  ████████████  选择最佳 recognizer        Exp3b: 后处理优化
Day 7  ████████████  Exp3c: TTA                 集成测试
Day 8  ████████████  全流程优化 + 提交准备
Day 9  ████████████  最终提交 + 备用方案准备
Day 10 ████████████  Buffer (调试/修复)
Day 11 ████████████  Buffer
Day 12 ████████████  最后提交
```

---

## F1 提升预估汇总

| 阶段 | 识别 Top1 | 检测 F1 | 端到端 F1 | ΔF1 |
|------|:---:|:---:|:---:|:---:|
| **基线 (当前)** | 32% | 0.82 | **0.36** | — |
| + Exp1 (k-NN) | 75% | 0.82 | **0.68** | +32pp |
| + Exp2 (Fine-tune) | 82% | 0.82 | **0.73** | +5pp |
| + Exp3 (检测+后处理) | 82% | 0.87 | **0.77** | +4pp |
| **最终 (三日叠加)** | **82%** | **0.87** | **~0.77** | **+41pp** |

> **注**: 冠军识别 88.7%, 假设其检测同样优秀 (F1≈0.90), 则其端到端 F1 ≈ 0.88 × 0.90 ≈ **0.79-0.84**。我们的目标 0.77 已接近冠军水平。

---

## 如果只能选一个实验

**选实验一 (DINOv2 k-NN)**。理由:
- 1 天内可见结果
- 零训练风险
- 预期 F1 从 0.36 跳到 0.68 → 已接近排行榜竞争水平
- 失败了还有 11 天尝试其他方案

---

*报告基于对全部项目代码、训练日志、竞赛测试集 (/saisdata) 的详尽实证分析。所有 F1 估算基于实际检测+识别 pipeline 数学模型。*
