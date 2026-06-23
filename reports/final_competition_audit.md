# 🔴 复赛提交前终审报告 (Final Competition Audit)

**审计时间**: 2026-06-06 19:20  
**审计目标**: OCR 竞赛复赛 Docker 提交  
**审计结果**: ✅ **PASS — 可以提交**

---

## A. Docker 合规性检查

| # | 检查项 | 结果 | 详情 |
|:---:|------|:---:|------|
| A1 | Dockerfile 构建 | ✅ PASS | `docker build` 成功，镜像 ID `51379c9f2be2` |
| A2 | 依赖安装 | ✅ PASS | torch 2.12, torchvision 0.27, ultralytics 8.4.60, opencv 4.13 |
| A3 | 运行时 pip install | ✅ PASS | 代码中无任何 `pip install`、`subprocess`、`os.system` 调用 |
| A4 | 联网下载 | ✅ PASS | 无 `urllib`、`requests`、`wget`、`curl` 调用；torch.hub 未使用 |
| A5 | 外部 API 调用 | ✅ PASS | 无任何网络请求 |
| A6 | 硬编码路径 | ✅ PASS | PROJECT_ROOT 由 `Path(__file__).resolve().parent.parent` 动态解析 |
| A7 | 容器内路径 | ✅ PASS | `/saisdata` (输入), `/saisresult` (输出), `/app/` (代码) 均正确定义 |

### 容器内路径拓扑

```
/app/
├── run.sh                    ← ENTRYPOINT
├── scripts/infer.py          ← 推理入口
├── src/
│   ├── arcface_model.py      ← ArcFace 模型定义
│   ├── recognizer.py          ← ArcFaceRecognizer
│   └── ...
├── checkpoints/
│   ├── yolo11s_det_1280.pt   ← YOLO11s (19MB)
│   └── convnext_arcface_best.pt ← ArcFace (344MB, E53)
└── mappings/
    ├── idx_to_class.json     ← 3483 类映射
    └── ID_to_chinese.json    ← 1781 字符映射
```

**A 章结论**: ✅ **PASS**

---

## B. 推理入口检查

| # | 检查项 | 结果 | 详情 |
|:---:|------|:---:|------|
| B1 | run.sh 唯一入口 | ✅ PASS | `ENTRYPOINT ["./run.sh"]` + `set -e` |
| B2 | 调用正确脚本 | ✅ PASS | `python /app/scripts/infer.py` (路径: 检测+识别完整 pipeline) |
| B3 | 参数传递完整 | ✅ PASS | CONF=0.15, IOU=0.3, IMGSZ=1280, DEVICE=cpu (可覆盖) |
| B4 | 输入目录 | ✅ PASS | `INPUT_DIR="${INPUT_DIR:-/saisdata}"` → fallback /saisdata, /input |
| B5 | 输出目录 | ✅ PASS | `OUTPUT_DIR="${OUTPUT_DIR:-/saisresult}"` → prediction.json |
| B6 | prediction.json 生成 | ✅ PASS | `OUTPUT_JSON = OUTPUT_DIR / "prediction.json"` |

### 调用链

```
ENTRYPOINT ["./run.sh"]
  ├── export CONF=0.15 IOU=0.3 IMGSZ=1280 DEVICE=cpu
  ├── export INPUT_DIR=/saisdata OUTPUT_DIR=/saisresult
  └── python /app/scripts/infer.py
        ├── YOLO 检测 (1280px, augment=True)
        ├── Crop + 0.9x shrink + 3px padding
        └── ArcFaceRecognizer (3483 classes, cosine logits)
              └── /saisresult/prediction.json
```

**B 章结论**: ✅ **PASS**

---

## C. 模型加载检查

| # | 检查项 | 结果 | 详情 |
|:---:|------|:---:|------|
| C1 | YOLO 路径 | ✅ PASS | `/app/checkpoints/yolo11s_det_1280.pt` (19MB) |
| C2 | ArcFace 路径 | ✅ PASS | `/app/checkpoints/convnext_arcface_best.pt` (344MB) |
| C3 | Mappings 路径 | ✅ PASS | `/app/mappings/idx_to_class.json` (63KB, 3483 entries) |
| C4 | 权重文件存在 | ✅ PASS | `test -f` 验证通过 (Dockerfile L50-67) |
| C5 | Docker 中可访问 | ✅ PASS | `COPY . /app` 复制全部文件 |

### 容器内加载验证

```
Building ArcFace model...
Total params: 30.0M               ← ConvNeXt-Tiny + ArcFace
Loading checkpoint...
Loaded: epoch=53, best_top1=0.3218 ← Phase 4 最佳
num_classes=3483, embedding_dim=512
ArcFace model load: PASS

torch=2.12.0+cu130                ← CUDA 13 build
convnext_tiny params: 28.6M       ← Backbone alone
ultralytics OK                    ← YOLO loadable
```

**C 章结论**: ✅ **PASS**

---

## D. 离线推理模拟

| # | 检查项 | 结果 | 详情 |
|:---:|------|:---:|------|
| D1 | 离线启动 | ✅ PASS | Docker build 阶段安装全部依赖，无需运行时下载 |
| D2 | 模型加载 | ✅ PASS | 全部本地 checkpoint，无 torch.hub 下载 |
| D3 | 离线推理 | ✅ PASS | Docker 容器测试: 7 img, 12 chars, 4s CPU |

### 离线安全证据

- `models.convnext_tiny(weights=None)` — 不下载预训练权重
- 无 `torch.hub.load()` 调用
- 无 `requests`, `urllib` 导入
- YOLO 从本地 `.pt` 文件加载
- Ultralytics settings 写入 `/tmp/` (本地)

**D 章结论**: ✅ **PASS**

---

## E. 输出格式检查

| # | 检查项 | 结果 | 详情 |
|:---:|------|:---:|------|
| E1 | 字段名 | ✅ PASS | `{"image_id": [{"bbox": [...], "text": "..."}, ...]}` |
| E2 | 数据类型 | ✅ PASS | image_id: str, bbox: list[int], text: str |
| E3 | bbox 格式 | ✅ PASS | `[x, y, w, h]` (整数，非 x1y1x2y2) |
| E4 | 坐标顺序 | ✅ PASS | x, y, w, h (x/y 为左上角) |
| E5 | 类别字段 | ✅ PASS | `text` 字段，UTF-8 中文字符 |
| E6 | 编码 | ✅ PASS | `ensure_ascii=False`, UTF-8 |
| E7 | 空图片 | ✅ PASS | 无检测时返回 `[]` 而非省略 |

### prediction.json 样例

```json
{
  "image_001": [
    {"bbox": [1299, 226, 167, 198], "text": "牛"}
  ],
  "image_002": []
}
```

**E 章结论**: ✅ **PASS**

---

## F. 风险扫描

### 🔴 Critical (致命 — 0 项)

*无*

### 🟠 High (高风险 — 0 项)

*无*

### 🟡 Medium (中风险 — 2 项)

| # | 风险 | 详情 | 缓解措施 |
|:---:|------|------|------|
| **M1** | PyTorch 2.12 CUDA 13 与 V100 驱动兼容性 | V100 (CC 7.0) 需 NVIDIA 驱动 ≥ 545。如果竞赛平台驱动版本过低，`torch.cuda.is_available()` 返回 False，推理退回 CPU。 | Dockerfile 中已包含 fallback (`DEVICE=cpu`) |
| **M2** | DEVICE 默认值=cpu | run.sh 默认 DEVICE=cpu。竞赛平台需显式设置 `DEVICE=cuda` 才能使用 V100。如不设置，4 小时 CPU 推理可能不够处理大量图片。 | 平台文档通常会说明 DEVICE 环境变量 |

### 🟢 Low (低风险 — 3 项)

| # | 风险 | 详情 |
|:---:|------|------|
| **L1** | 镜像大小 11.5GB | 接近某些平台限制。PyTorch 2.12 自带大量 CUDA 库。 |
| **L2** | Ultralytics settings 写入 | 首次运行写入 `/tmp/Ultralytics/settings.json`，无兼容性问题。 |
| **L3** | `augment=True` | TTA 翻倍推理时间。如果 4 小时不够，可设 `--no-augment`。 |

---

## 7. 最终结论

```
╔══════════════════════════════════════════════╗
║                                              ║
║   复赛提交终审: ✅ PASS                       ║
║                                              ║
║   风险等级: 🟢 LOW                            ║
║   可以直接提交复赛评测                         ║
║                                              ║
╚══════════════════════════════════════════════╝
```

### 提交前建议

| # | 建议 | 优先级 |
|:---:|------|:---:|
| 1 | **本地 Docker 实测** — 已在本机完成 7 图 12 字符端到端测试 | ✅ 已完成 |
| 2 | **确认 DEVICE=cuda** — 与竞赛主办方确认环境变量设置 | 🟡 |
| 3 | **确认 NVIDIA 驱动版本** — 确保 ≥ 545 以兼容 CUDA 13 | 🟡 |
| 4 | **考虑减小镜像** — 使用 PyTorch 2.5.1 CPU wheel 可缩小 3-4GB | 🟢 |

### 提交命令

```bash
# 导出 tar (提交格式)
docker save ocr-competition:latest | gzip > ocr-competition.tar.gz

# 平台运行
docker run --rm --gpus all \
  -v /saisdata:/saisdata:ro \
  -v /saisresult:/saisresult \
  -e DEVICE=cuda \
  ocr-competition:latest
```

### 剩余风险

| 风险 | 概率 | 影响 | 缓解 |
|------|:---:|:---:|------|
| GPU 驱动不兼容 → CPU fallback | 低 | 推理变慢 | PyTorch 2.5.1 CUDA 12 可能更兼容 |
| DEVICE 未设 cuda → CPU | 低 | 推理变慢 | 平台文档通常覆盖此场景 |
| 镜像太大被拒 | 极低 | 提交失败 | 可优化至 ~5GB |

---

## 附录: 完整检查清单

| 章节 | 检查项数 | PASS | FAIL |
|------|:---:|:---:|:---:|
| A. Docker 合规性 | 7 | 7 | 0 |
| B. 推理入口 | 6 | 6 | 0 |
| C. 模型加载 | 5 | 5 | 0 |
| D. 离线推理 | 3 | 3 | 0 |
| E. 输出格式 | 7 | 7 | 0 |
| **总计** | **28** | **28** | **0** |

---

*终审完成时间: 2026-06-06 19:20  
审计状态: ✅ 28/28 PASS — CLEAR FOR SUBMISSION*
