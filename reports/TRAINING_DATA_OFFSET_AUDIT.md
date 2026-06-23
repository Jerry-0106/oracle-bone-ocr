# 训练数据偏移审计报告

**日期**: 2026-06-15
**审计范围**: Detection XML bboxes + Recognition crop 居中质量
**样本量**: 200 张检测原图 (1923 个 bbox) + 300 张识别 crop
**方法**: 前景像素 (dark ink) 中心 vs bbox/图像 中心偏移计算

---

## 1. Detection XML Bbox 审计

### 1.1 样本统计

| 指标 | 值 |
|------|-----|
| 审计原图数 | 200 |
| 总 bbox 数 | 1,923 |
| 每图平均 bbox | 9.6 |
| XML 未找到 | 0/200 (100% 匹配) |

### 1.2 Bbox 偏移统计

| 指标 | 值 | 判断 |
|------|-----|------|
| dx (归一化) | mean=-0.0006, std=0.0454, median=-0.0027 | **无系统性偏移** |
| dy (归一化) | mean=-0.0018, std=0.0535, median=-0.0017 | **无系统性偏移** |
| dx (像素) | mean=-0.1px, std=10.9px, median=-0.5px | 偏移极小 |
| dy (像素) | mean=0.1px, std=14.9px, median=-0.4px | 偏移极小 |
| 偏移 > 10% bbox 尺寸 | 190/1923 (9.9%) | 轻微 |
| 偏移 > 15% bbox 尺寸 | 77/1923 (4.0%) | 少量异常 |

### 1.3 Bbox 尺寸

| 指标 | 值 |
|------|-----|
| 宽度 | mean=235px, median=212px, min=2, max=1330 |
| 高度 | mean=277px, median=229px, min=1, max=2124 |
| 宽高比 | mean=0.96 (接近正方形) |
| 注意 | **存在 2×1px 的退化 bbox** (极少数异常标注) |

### 1.4 字符贴边分析

| 指标 | 比例 |
|------|------|
| 前景距 bbox 边缘 ≤5px | **98.2%** |
| 其中：贴顶部 | 95.8% |
| 贴底部 | 96.2% |
| 贴左侧 | 95.5% |
| 贴右侧 | 96.5% |

### 1.5 Bbox 松紧度

| 指标 | 比例 |
|------|------|
| 前景占比 (fg_ratio) | mean=0.616, median=0.705 |
| 过紧 (fg > 85% bbox 面积) | **22.7%** |
| 过松 (fg < 15% bbox 面积) | 6.1% |

### 1.6 最严重样本分析

Top 20 最偏移样本中：
- **8/20 是 fg_ratio < 0.15 的过松 bbox** (包含大块背景，前景稀少)
- **2/20 是退化 bbox** (2×1px, fg=100%)
- **1/20 是空 bbox** (fg_ratio=0.0)
- **9/20 是正常的但偏移较大的 bbox**

真正的"偏移"问题只占约 4%——大部分"最差"样本其实是标注质量问题，不是偏移问题。

---

## 2. Recognition Crop 居中审计

### 2.1 样本统计

| 指标 | 值 |
|------|-----|
| 审计 crop 数 | 300 |
| 空 crop (无前景) | 5 (1.7%) |

### 2.2 居中偏移统计

| 指标 | 值 | 判断 |
|------|-----|------|
| dx (归一化) | mean=-0.0051, std=0.0429, median=-0.0038 | **居中良好** |
| dy (归一化) | mean=-0.0018, std=0.0377, median=-0.0023 | **居中良好** |
| 偏移 > 10% | 25/300 (8.3%) | 轻微 |
| 偏移 > 15% | 7/300 (2.3%) | 少量异常 |
| 前景距边缘 ≤5px | **95.0%** | 大部分字符贴边 |

### 2.3 Crop 尺寸

| 指标 | 值 |
|------|-----|
| 宽度 | mean=254px, median=226px |
| 高度 | mean=300px, median=251px |
| 宽度 < 224px 的 crop | 148/300 (49.3%) |
| 高度 < 224px 的 crop | 129/300 (43.0%) |
| 极端宽高比 (>3:1) | 1/300 (0.3%) |

### 2.4 Resize(256)+CenterCrop(224) 影响

| 结果 | 数量 | 比例 |
|------|------|------|
| 安全 (字符完全/大部分在 crop 内) | 287 | 95.7% |
| 部分截断 (50%+ 字符丢失) | 13 | 4.3% |
| 完全截断 (字符完全在 crop 外) | 0 | 0.0% |

---

## 3. 判断与发现

### 3.1 是否存在系统性 bbox 偏移？

**否。没有发现系统性偏移。** 

- dx 和 dy 的均值都接近 0（分别为 -0.0006 和 -0.0018）
- 中位数接近 0
- 标准差仅 ~5% 的 bbox 尺寸

XML 标注质量整体较高。98.2% 的 bbox 前景贴边是 **标注风格导致的，不是 bug**——甲骨文标注者习惯画紧凑包围框，让 bbox 刚好包住字符。这本身正确，但意味着 crop 出来的字符天生贴边，留给检测偏移的容错空间为 0。

### 3.2 Recognition crop 是否字符不居中？

**基本居中。8.3% 的 crop 偏移 > 10%，2.3% 偏移 > 15%。**

大部分 crop 的字符位于图像中心附近。95% 的 crop 前景贴边与 detection bbox 的情况一致——标注风格导致的紧凑裁剪，不是居中问题。

### 3.3 当前 Resize(256)+CenterCrop(224) 是否会切掉字符？

**对于训练数据 (recognition crops)：4.3% 的 crop 会受到显著影响。**

但对于推理时 YOLO 检测出的 crop（可能更不准确），实际受影响比例会更高。原因是：
- YOLO 检测框可能比 XML 标注框偏移 5-10px
- 推理时 0.9x shrink + 3px padding 进一步压缩了可用空间
- 窄长字符（如 `丨` 类竖笔）在 Resize+CenterCrop 下特别容易截断

### 3.4 是否需要改成 letterbox (保持比例 + 补白)？

**推荐改为 letterbox，理由如下：**

1. **当前 CenterCrop 在极端宽高比字符上会截断**：虽然只有 4.3% 训练 crop 受影响，但推理时影响更大
2. **保持比例避免字符变形**：Resize 对齐短边后再 CenterCrop 会改变字符的宽高比
3. **letterbox 补白模拟训练数据的 padding**：训练 crop 来自 XML bbox，天然有少量背景 padding。推理时 letterbox 可以提供类似的上下文
4. **实现简单**：`Resize such that longest side=224, pad short side to 224 with mean color`

---

## 4. 推荐的修正策略

### 策略 1：推理时增大 crop padding ⭐⭐⭐⭐⭐

**当前**: CROP_PADDING=3
**建议**: CROP_PADDING=**10** (对于 <50px 的小字符) 或 **max(8, 0.1 * bbox_width)**

**理由**: 98.2% 的 XML bbox 前景贴边，3px padding 对于 YOLO 的检测框偏移几乎没有任何容错。

### 策略 2：替换 CenterCrop 为 letterbox resize ⭐⭐⭐⭐

**当前**:
```python
transforms.Resize(256)         # resizes shorter side to 256
transforms.CenterCrop(224)     # crops center 224x224
```

**建议**:
```python
transforms.Resize(224, max_size=224)  # longest side = 224, keep aspect
# + pad to 224x224 with mean pixel value
```

**理由**: 避免窄长字符被截断，保持字符原始比例。

### 策略 3：训练时也使用 letterbox ⭐⭐⭐

将训练 pipeline 的 RandomResizedCrop 改为 "先 letterbox resize，再随机裁剪 224×224 区域"。这样训练和推理使用相同的前处理。

### 策略 4：移除推理时的 0.9x bbox shrink ⭐⭐⭐

**当前**: infer.py 对 bbox 做 0.9x 中心收缩
**建议**: 移除 shrink，或在**仅输出 prediction.json 时对 bbox shrink**，crop 使用原始 bbox + padding

**理由**: Recognizer 看到的区域 ≠ 评估时用的 bbox，造成系统性的不一致。

---

## 5. 最严重样本路径

最严重的 20 个 bbox 偏移样本保存在：
- `reports/bbox_audit_samples/worst_20_samples.json`

最严重的 20 个 crop 居中样本保存在：
- `reports/recognition_crop_audit/worst_20_crops.json`

这些大部分是空 bbox、退化 bbox 或包含大块背景的过松 bbox，而非真正的"偏移"问题。

---

## 6. 结论

| 问题 | 严重度 | 说明 |
|------|--------|------|
| 系统性 bbox 偏移 | 🟢 无 | dx/dy 均值为 0，无系统性偏移 |
| 字符贴边 | 🟡 标注风格 | 98.2% 贴边是紧凑标注风格，不是 bug，但导致 crop padding 容错极低 |
| 字符居中 | 🟢 良好 | 8.3% 偏移 >10%，整体居中 |
| CenterCrop 截断 | 🟡 轻微 | 4.3% 训练 crop 受影响，推理时可能更高 |
| 退化 bbox | 🟡 少量 | 存在 2×1px 极少量标注异常 |

**最重要的发现：98.2% 的 XML bbox 前景贴边不是 bug，而是"紧凑标注"风格。这本身正确，但意味着 YOLO 检测框的任何微小偏移（>3px）都会导致字符被切边。当前 infer.py 的 CROP_PADDING=3 太小。**

**最优先修复：将 CROP_PADDING 从 3 改为 8-10，并使用 letterbox resize 替代 Resize+CenterCrop。**
