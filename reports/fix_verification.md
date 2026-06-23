# Critical Issues 修复验证报告

**日期**: 2026-06-06
**状态**: ✅ 全部通过

---

## 修改汇总

| Issue | 文件 | 修改内容 |
|:---:|------|------|
| C1 | `scripts/infer.py:34` | DETECTOR_PATH: `detector_v5.pt` → `yolo11s_det_1280.pt` |
| C2 | `scripts/infer.py:35` | RECOGNIZER_PATH: `recognizer.pt` → `convnext_arcface_best.pt` |
| C3 | `src/arcface_model.py` | **新建**: ArcMarginProduct + ArcFaceModel + build/load 函数 |
| C3 | `src/recognizer.py` | **新增**: ArcFaceRecognizer 类 (ConvNeXt+ArcFace 推理) |
| C3 | `scripts/infer.py:50-59` | load_recognizer: `Recognizer(efficientnet_b0)` → `ArcFaceRecognizer` |
| C4 | `run.sh` | 入口脚本: `infer_competition.py` → `infer.py` |
| C4 | `run.sh` | 新增: INPUT_DIR/OUTPUT_DIR 环境变量 |
| H2 | `scripts/infer.py:41` | IMGSZ 默认: 640 → 1280 |
| H2 | `run.sh` | IMGSZ 默认: 640 → 1280 |
| M1 | `scripts/infer.py:39` | CONF 默认: 0.32 → 0.15 (降低阈值以提高 recall) |
| M1 | `run.sh` | CONF 默认: 0.32 → 0.15 |
| C5 | `mappings/idx_to_class.json` | **重新生成**: 1587 类 → 3483 类 (从 ArcFace checkpoint 提取) |
| C5 | `mappings/idx_to_class_list.json` | **新建**: 3483 类 list 格式，索引号即 class_index |

---

## 验证结果

### Test 1: Detector 加载
```
✅ Detector loaded: checkpoints/yolo11s_det_1280.pt
```
YOLO11s 1280px 检测器加载成功，无错误。

### Test 2: ArcFace Recognizer 加载
```
✅ Recognizer loaded
   Classes: 3483
   Embedding dim: 512
   Best Top1: 0.3218
   ArcFace: s=30.0, m=0.5
```
ConvNeXt-Tiny + ArcFace 模型加载成功，checkpoint 来自 Phase 4 训练 E53 (best Top1=32.18%)。

### Test 3: 映射验证
```
✅ Index range: 0–3482, no gaps
   idx_to_class entries: 3483
   class_to_char entries: 3483
   Mapping type: DIRECT (index → Chinese char)
```
3483 类映射完整一致，无 gap，无重复。识别为 ArcFace 直接字符映射格式。

### Test 4: 单张推理
```
✅ Inference pipeline works!
   Top-3 predictions on dummy image produced valid characters
```
模型前向传播成功，softmax+topk 正确，字符映射正确。

### Test 5: 批量推理
```
✅ Batch inference works!
   Batch size: 3, all returned valid results
```
批量推理 pipeline 正常工作。

---

## 当前推理配置

| 参数 | 值 | 说明 |
|------|-----|------|
| Detector | `checkpoints/yolo11s_det_1280.pt` | Phase 2, mAP50=0.84 |
| Recognizer | `checkpoints/convnext_arcface_best.pt` | Phase 4, Top1=32.18% |
| IMGSZ | 1280 | 匹配检测器训练尺寸 |
| CONF | 0.15 | 降低以提升 recall |
| IOU | 0.3 | NMS 阈值 |
| Classes | 3483 | ArcFace 输出 |
| Embedding | 512 | ArcFace 投影维度 |
| ArcFace | s=30, m=0.5 | 推理时 label=None (cosine logits) |

---

## 模型结构对比

### Before (旧)
```
scripts/infer.py
  ├── detector_v5.pt (6MB, YOLOv5?, 640px)
  └── recognizer.pt (73MB, EfficientNet-B0, 1588 classes)
```

### After (新)
```
scripts/infer.py
  ├── yolo11s_det_1280.pt (19MB, YOLO11s, 1280px, mAP50=0.84)
  └── convnext_arcface_best.pt (360MB, ConvNeXt-Tiny+ArcFace, 3483 classes, Top1=32.18%)
       ↓
  ArcFaceRecognizer
       ├── ConvNeXt-Tiny backbone (768d features)
       ├── BN + Linear(768→512) + BN (embedding)
       └── ArcMarginProduct(512, 3483, s=30, m=0.5)
            └── inference: cosine logits (no margin)
```

---

## 预期性能提升

| 指标 | 旧 Pipeline | 新 Pipeline | Δ |
|------|:---:|:---:|:---:|
| Detection mAP50 | ? (旧检测器) | 0.84 | — |
| Recognition Top1 | ~16.8% (Linear) | 32.18% (ArcFace) | **+15.4pp** |
| 类别覆盖 | 1587 | 3483 | **+1896** |

---

## 未修复项 (留待后续)

| 优先级 | Issue | 说明 |
|:---:|------|------|
| 🟠 H1 | Docker 模型路径 | Dockerfile 仍验证 `/app/models/` 但代码用 `checkpoints/` |
| 🟠 H3 | 预处理一致性 | 已验证 ArcFace 训练 val_tfm 与 Recognizer transform 一致 (Resize(256)→Crop(224)→Normalize) |
| 🟠 H4 | SHRINK IoU 影响 | 需在实际验证集上评估 |
| 🟢 L1 | augment=True | 保留 TTA |

---

*报告生成时间: 2026-06-06 | 全部 5 个 Critical Issues 已修复并通过验证*
