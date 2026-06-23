# 🔴 Pre-Submission Audit Report — Oracle Bone Character OCR

**审计日期**: 2026-06-06  
**审计范围**: 完整 OCR Pipeline（Detection → Crop → Recognition → JSON）  
**严重程度**: 🔴 5 Critical / 🟠 4 High / 🟡 3 Medium / 🟢 2 Low

---

## Executive Summary

**当前推理代码与已训练模型之间存在严重的工程断连。如果不修复，提交将产生近乎随机的识别结果。**

核心问题：`scripts/infer.py` 仍指向 Phase 1 的旧模型（detector_v5.pt + recognizer.pt），而非 Phase 2/4 的训练成果（yolo11s_det_1280.pt + convnext_arcface_best.pt）。Recognizer 架构不支持 ConvNeXt+ArcFace。Docker 调用链完全断裂。

---

## 🔴 Critical Issues (致命问题 — 会导致 F1 下降 >50%)

### C1: Detector Model Path 指向旧模型

| 项目 | 详情 |
|------|------|
| **文件** | `scripts/infer.py:34`, `scripts/infer_competition.py:17` |
| **当前值** | `checkpoints/detector_v5.pt` (6.2 MB, 旧模型) |
| **应使用** | `checkpoints/yolo11s_det_1280.pt` (19.2 MB, Phase 2 训练) |
| **影响** | 使用未训练/旧版检测器，mAP 可能从 0.84 暴跌至未知水平 |
| **修复** | 替换路径为 `checkpoints/yolo11s_det_1280.pt`，IMGSZ 改为 1280 |

### C2: Recognizer Model Path 指向旧模型

| 项目 | 详情 |
|------|------|
| **文件** | `scripts/infer.py:35` |
| **当前值** | `checkpoints/recognizer.pt` (72.9 MB, EfficientNet-B0, 1588 类) |
| **应使用** | `checkpoints/convnext_arcface_best.pt` (360 MB, ConvNeXt+ArcFace, 3483 类, Top1=32.18%) |
| **影响** | 使用旧的、未针对性训练的识别器。Phase 3B Linear Classifier Top1 仅 16.79%，ArcFace 为 32.18%。使用旧模型直接损失 15+pp Top1。 |
| **修复** | 更换路径 + 重写 Recognizer 类支持 ConvNeXt+ArcFace |

### C3: Recognizer 架构无法加载 ArcFace 模型

| 项目 | 详情 |
|------|------|
| **文件** | `src/recognizer.py:53-69` |
| **当前支持** | `efficientnet_b0`, `efficientnet_b1`, `resnet18` |
| **需要支持** | `ConvNeXt-Tiny + BN + Linear(768→512) + ArcMarginProduct(512, 3483, s=30, m=0.5)` |
| **影响** | 即使路径正确，加载 checkpoint 也会崩溃（`KeyError` 或 `RuntimeError`） |
| **修复** | 需要新增 ArcFace 模型构建代码 + 推理逻辑（训练时用 label 参数，推理时用 label=None 获取 cosine logits） |

### C4: run.sh 调用链完全错误

| 项目 | 详情 |
|------|------|
| **文件** | `run.sh:25` |
| **当前值** | `python /app/scripts/infer_competition.py` |
| **问题** | `infer_competition.py` 是纯检测脚本，**不包含识别功能**！无 Recognizer 加载，无 crop，无 character 输出。 |
| **影响** | Docker 容器中只会输出检测框，不会输出 `prediction.json` 格式（或输出格式错误） |
| **修复** | 改为 `python /app/scripts/infer.py`（或更名/重构为 competition entry point） |

### C5: 类别数不匹配 — 旧 Mappings vs ArcFace 训练

| 项目 | 详情 |
|------|------|
| **文件** | `mappings/idx_to_class.json`, `mappings/ID_to_chinese.json` |
| **当前 Mappings** | 1587 个类别（对应旧 EfficientNet 识别器） |
| **ArcFace 训练** | 3483 个类别（`datasets/recognition/labels.csv`） |
| **影响** | ArcFace 模型的 3483 输出维度与 1587 映射不兼容。模型输出 idx=2500 时 `idx_to_class.json` 中无对应 key，返回原始数字字符串 "2500" 而非中文字符。 |
| **修复** | 需要从 ArcFace 训练数据重新生成 3483 类映射文件 |

---

## 🟠 High Risk Issues (高风险 — 会导致显著分数下降)

### H1: Dockerfile 模型路径与代码路径不一致

| 项目 | 详情 |
|------|------|
| **文件** | `Dockerfile:47` vs `scripts/infer.py:34-35` |
| **Dockerfile 验证** | `/app/models/detector.pt` 和 `/app/models/recognizer.pt` |
| **代码实际路径** | `checkpoints/detector_v5.pt` 和 `checkpoints/recognizer.pt`（相对 PROJECT_ROOT） |
| **影响** | Dockerfile COPY 后模型在 `/app/` 下，但代码查找 `checkpoints/` 子目录。需确保 COPY 后路径一致。 |
| **修复** | 统一路径：要么 Dockerfile 改为验证 `/app/checkpoints/`，要么代码改为 `/app/models/` |

### H2: IMGSZ 默认值 640 而非 1280

| 项目 | 详情 |
|------|------|
| **文件** | `run.sh:8`, `scripts/infer.py:41` |
| **当前默认** | `IMGSZ=640` |
| **训练尺寸** | YOLO11s 训练于 `imgsz=1280` |
| **影响** | 640px 输入会导致检测器看到完全不同的特征尺度。mAP 可能显著下降（YOLO 对输入尺寸敏感）。 |
| **修复** | 将默认 IMGSZ 改为 1280，或至少设为 `IMGSZ="${IMGSZ:-1280}"` |

### H3: 推理预处理与训练不一致

| 项目 | 详情 |
|------|------|
| **文件** | `scripts/infer.py:170-177`, `src/recognizer.py:46-51` |
| **Infer crop** | 从原始 bbox + 3px padding 裁剪，缩放到 224 |
| **Recognizer transform** | Resize(256) → CenterCrop(224) |
| **ArcFace 训练** | 需要确认使用相同 transform（train_arcface.py 中 transform） |
| **风险** | 如果 ArcFace 训练用了不同的预处理（如 `Resize(224)` 而非 `Resize(256)+CenterCrop(224)`），推理结果会受 domain shift 影响 |
| **修复** | 检查 `_training/train_arcface.py` 的 transform，确保推理预处理完全一致 |

### H4: SHRINK=0.9 的 IoU 影响未经 ArcFace 验证

| 项目 | 详情 |
|------|------|
| **文件** | `scripts/infer.py:164` |
| **逻辑** | 检测框缩小 10%（中心对齐），但 crop 用原始框+padding。输出 bbox 是缩小后的。 |
| **风险** | 缩小后 bbox 与 GT 的 IoU 会降低。如果比赛用 IoU≥0.5 评估，可能导致部分正确识别被视为 FP。 |
| **修复** | 在 validation set 上测试有无 shrink 的 F1 差异。如果 shrink 显著降低 IoU，考虑去掉或减小比例。 |

---

## 🟡 Medium Risk Issues (中风险 — 可能导致小幅分数下降)

### M1: 检测置信度阈值 0.32 可能过高

| 项目 | 详情 |
|------|------|
| **文件** | `scripts/infer.py:39` |
| **当前值** | `CONF=0.32` |
| **影响** | 在 mAP50=0.84 的检测器上，阈值 0.32 可能过滤掉部分真阳性（低置信度的正确检测）。Recall 可能下降。 |
| **建议** | 评估 conf=0.1, 0.15, 0.2, 0.25, 0.32 对 F1 的影响，选择最优值 |

### M2: 检测框边缘裁剪逻辑

| 项目 | 详情 |
|------|------|
| **文件** | `scripts/infer.py:152-157` |
| **过滤规则** | `w > img_w*0.3` 或 `h > img_h*0.3` 的框被丢弃；`w < 5` 或 `h < 5` 的框被丢弃 |
| **风险** | 大尺寸字符（如标题字）或极小字符可能被误过滤 |
| **建议** | 评估被过滤框的比例，确认不包含重要 GT |

### M3: `recognizer.py` 与 `infer.py` 的 device 处理不一致

| 项目 | 详情 |
|------|------|
| **文件** | `scripts/infer.py:42-43`, `src/recognizer.py:20-22` |
| **infer.py** | `DEVICE = "cuda" if DEVICE_ENV == "cuda" and torch.cuda.is_available() else "cpu"` |
| **recognizer.py** | 同样逻辑，但独立判断 |
| **风险** | 如果环境变量设为 "cuda" 但只有一个设备可用，两者均 fail-safe 到 CPU，无冲突。但未传递 device 一致性检查。 |
| **影响** | 低 — 当前逻辑足够健壮 |

---

## 🟢 Low Risk Issues (低风险 — 优化建议)

### L1: Detect 阶段使用 augment=True 可能引入不确定性

| 项目 | 详情 |
|------|------|
| **文件** | `scripts/infer.py:135` |
| **当前值** | `augment=True` |
| **影响** | TTA 虽可提升 mAP，但增加推理时间。比赛环境下时间限制未知。 |
| **建议** | 确认比赛时间限制，如宽松则保留；如严格则关闭 |

### L2: Recognizer 的 weights_only=True 可能不兼容旧 checkpoint

| 项目 | 详情 |
|------|------|
| **文件** | `src/recognizer.py:27` |
| **当前值** | `torch.load(model_path, weights_only=True)` |
| **影响** | 如果 checkpoint 包含非 tensor 数据（如 config dict），`weights_only=True` 会失败。旧 recognizer.pt 可能包含 config。 |
| **注** | 旧的 recognizer.pt 加载成功说明此设置兼容旧模型，但新 ArcFace checkpoint 需要确认 |

---

## 完整修复优先级与工作量估算

| 优先级 | Issue | 修复内容 | 估算时间 |
|:---:|------|------|:---:|
| 🔴 **C1** | Detector 路径 | 替换 `detector_v5.pt` → `yolo11s_det_1280.pt` | 5 min |
| 🔴 **C2** | Recognizer 路径 | 替换 `recognizer.pt` → `convnext_arcface_best.pt` | 5 min |
| 🔴 **C3** | Recognizer 架构 | 重写 `recognizer.py` 支持 ConvNeXt+ArcFace 推理 | 2-3 hours |
| 🔴 **C4** | run.sh 调用链 | 改为调用正确的推理脚本 | 10 min |
| 🔴 **C5** | 类别映射 | 从 ArcFace 训练数据生成 3483 类映射 | 1 hour |
| 🟠 **H1** | Docker 路径 | 统一模型和映射文件路径 | 30 min |
| 🟠 **H2** | IMGSZ | 改默认值为 1280 | 2 min |
| 🟠 **H3** | 预处理一致性 | 验证训练/推理 transform 一致 | 30 min |
| 🟠 **H4** | SHRINK 验证 | 评估 bbox shrink 的 IoU 影响 | 1 hour |
| 🟡 **M1** | 置信度阈值 | 评估不同 conf 阈值的 F1 | 1 hour |
| 🟡 **M2** | 框过滤 | 验证过滤规则合理性 | 30 min |
| 🟢 **L1/L2** | 其他 | 次要优化 | 1 hour |

---

## 检查清单

| # | 审计项 | 状态 | 严重度 |
|:---:|------|:---:|:---:|
| 1 | **类别映射链** (idx→class_id→char) | ⚠️ 旧 mappings 与 ArcFace 3483 类不匹配 | 🔴 C5 |
| 2 | **Bbox 格式** (xywh 输出) | ✅ `xyxy_to_xywh()` 转换正确 | 🟢 |
| 3 | **检测-识别坐标一致性** | ✅ shrink 逻辑正确，有边界 clamp | 🟢 |
| 4 | **推理流程** (Image→Detect→Crop→Rec→JSON) | 🔴 run.sh 调用检测专用脚本，无识别 | 🔴 C4 |
| 5 | **置信度** | ⚠️ conf=0.32 可能偏高 | 🟡 M1 |
| 6 | **提交文件格式** | ✅ `infer.py` 输出格式正确 `{"bbox": [x,y,w,h], "text": "char"}` | 🟢 |
| 7 | **Docker** | 🔴 路径/调用链/模型全部不匹配 | 🔴 C1-5, H1 |

---

## 建议行动

**立即停止任何新模型训练。** 在修复这些工程问题之前，再好的模型也无法正确部署。

**Phase 0 — 紧急修复（需在下次训练前完成）:**

1. **C3 (最复杂)**: 新增 ArcFace 推理支持。需要：
   - 在 `src/recognizer.py` 或新文件中实现 ArcFace 模型构建
   - 实现推理时的 `label=None` 模式（纯 cosine logits）
   - 加载 `convnext_arcface_best.pt` 的 state_dict

2. **C5**: 从 ArcFace 训练的 `labels.csv` 重新生成 3483 类 mapping：
   - 提取训练中的 `idx_to_class` 映射
   - 生成对应的 `ID_to_chinese.json`

3. **C1+C2+C4+H1+H2**: 统一代码路径和配置

**修复后验证**:
- 在验证集（9010 样本）上运行完整 pipeline
- 确认 Top1 ≈ 32%（与训练验证集一致）
- 手动抽查 50 个预测：检测框 + 识别字符是否正确

---

*审计完成时间: 2026-06-06 | 审计工具: Claude Code Audit Mode*
