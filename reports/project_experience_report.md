# 甲骨文 OCR 项目经历报告

> 第四届世界科学智能大赛 · 古文字识别赛道

---

## 1. 项目背景

本项目参加的是"第四届世界科学智能大赛"的古文字识别赛道。任务是对甲骨文拓片图像进行**文字检测与识别**：给定一张高分辨率拓片图像，需要先检测出每个文字的位置（bounding box），再识别出对应的汉字，最终输出 JSON 格式的预测结果参与平台评测。

评测方式为**线上自动评分**，平台使用 Final F1 作为主要指标，综合考虑 Detection F1 和 Recognition Accuracy：

- **Final F1** = 2 × Precision × Recall / (Precision + Recall)
- 其中 Precision/Recall 基于检测框匹配（IoU ≥ 0.5）后的字符级正确与否计算
- 一个字符位置正确且文字正确才算 TP；位置正确但文字错误计为 FP 和 FN 各一次

系统采用经典的**两阶段 OCR pipeline**：检测（Detection）→ 裁剪（Crop）→ 识别（Recognition）。

---

## 2. 项目目标

核心目标：**在检测性能已较高的前提下，最大化识别准确率，从而提升最终 F1。**

初始 baseline（v9）的检测 F1 已达 0.9087 左右，但识别 Accuracy 仅约 0.44，远未饱和。由于 Final F1 的上限受识别准确率直接约束，后续所有优化重点均放在识别模型上。

---

## 3. 技术栈

| 类别 | 技术 |
|---|---|
| 语言 | Python 3.10+ |
| 深度学习框架 | PyTorch 2.5.1 |
| 检测模型 | Ultralytics YOLO11s |
| 识别模型 | Swin Transformer (timm), ConvNeXt |
| 度量学习 | ArcFace (实验阶段) |
| 图像处理 | OpenCV, Pillow, torchvision |
| 容器化 | Docker, Docker Desktop (WSL2) |
| 镜像仓库 | 阿里云容器镜像服务 (ACR) |
| 版本管理 | Git, GitHub |
| 训练硬件 | NVIDIA RTX 5070 Ti Laptop GPU (12GB VRAM) |
| 评测框架 | 比赛官方 evaluate.py |

---

## 4. 系统整体架构

```
┌──────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│ 拓片原图  │ →  │ YOLO11s 检测  │ →  │ bbox clamp +   │ →  │ 动态 padding  │
│ (高分辨率) │    │ @1280px      │    │ 过滤过大/过小框 │    │ crop 字符区域   │
└──────────┘    └──────────────┘    └───────────────┘    └──────┬───────┘
                                                                 │
                                                    ┌────────────▼───────────┐
                                                    │ Swin-Tiny 识别          │
                                                    │ Resize(256)→CenterCrop │
                                                    │ (224)→Normalize→分类   │
                                                    └────────────┬───────────┘
                                                                 │
                                          ┌──────────────────────▼────────────┐
                                          │ 组装 prediction.json               │
                                          │ {"image_id": [{"bbox": [x,y,w,h],  │
                                          │   "text": "char"}, ...]}           │
                                          └──────────────────────┬────────────┘
                                                                 │
                                          ┌──────────────────────▼────────────┐
                                          │ Docker 容器 / 线上平台评测          │
                                          │ /saisdata → run.sh → /saisresult   │
                                          └───────────────────────────────────┘
```

**容器运行流程**：比赛平台挂载 `/saisdata`（输入图片）和 `/saisresult`（输出目录），执行 `run.sh` 启动 `scripts/infer.py`，自动完成检测→裁剪→识别→输出全流程。

---

## 5. 数据与预处理

### 5.1 数据来源

- **检测数据**：比赛提供的拓片图像 + XML 标注文件，每个字符标注包含位置和多边形坐标
- **识别数据**：两种来源
  1. **原始 recognition crop**：基于原始标注裁剪的字符图，每个字符有标准位置和 label
  2. **pipeline-style crop**：用 YOLO detector 在实际图像上检测后裁剪的字符图，分布更接近线上推理

### 5.2 两种 crop 分布的差异

原始 recognition crop 和 YOLO 实际检测 crop 在字符位置精度、padding、图像质量上存在差异。如果只在原始 crop 上训练识别模型，上线时面对 YOLO 输出的 crop 会产生分布偏移，导致识别准确率下降。

### 5.3 训练与评测策略

- **训练**：混合 50% 原始 recognition crop + 50% pipeline-style crop，使用 WeightedRandomSampler 平衡类别
- **评测**：构建 931 张图片的 corrected pipeline evaluation，使用真实 YOLO detector 跑检测 + 识别 + GT 解析，计算 corrected F1。这比只看 recognition validation set 的 Top1 更能反映真实上线表现
- 后期所有决策**只看 931 corrected F1 和 Rec Acc**，不再以 orig_val Top1 为准

---

## 6. 检测模型优化

### 6.1 模型选择

使用 **YOLO11s @ 1280px** 进行字符检测。YOLO11s 是 YOLO 系列的中小型模型，在速度和精度间取得良好平衡。1280px 的高分辨率输入对甲骨文小字检测至关重要。

### 6.2 参数调优

对 conf（置信度阈值）和 iou（NMS 阈值）进行了系统性搜索：

| 实验 | 结果 |
|---|---|
| conf sweep (0.10~0.22) | conf=0.20 时 local Final F1 最高 |
| iou sweep (0.25~0.40) | iou=0.25 时 TP/FP 平衡最优 |
| max_det=300 | 足够覆盖单张图片的最大字符数 |
| augment=True | YOLO 推理时 test-time augmentation |

最终选择 **conf=0.20, iou=0.25, max_det=300, augment=True**。

### 6.3 Crop 后处理策略

- **移除 bbox shrink**：不缩小检测框，保留完整字符区域
- **动态 padding**：`pad = max(8, int(0.08 * max(w, h)))`，根据字符大小自适应留白
- **边界 clamp**：严格限制 crop 区域不超出图像边界
- **异常过滤**：过滤过小（<5px）、过大（>30% 图像面积）的检测框

### 6.4 检测效果

检测 F1 稳定在 **0.908696**（线上），多次提交该值不变（detector 未再做改动）。检测已不是系统的主要瓶颈，后续优化全部聚焦于识别。

---

## 7. 识别模型演进

### 7.1 ConvNeXt-Tiny + ArcFace Baseline

项目最初使用 **ConvNeXt-Tiny + ArcFace loss** 作为识别模型。ConvNeXt 是现代化的 CNN backbone，ArcFace 是常用于人脸识别的 angular margin loss。

**问题**：
- 本地 orig_val Top1 仅约 32.18%
- 线上 Recognition Acc 约 0.441
- 3483 个甲骨文字符中有大量字形极其相似的类别，CNN 的局部感受野在处理细粒度差异时能力不足

### 7.2 失败或收益有限的实验（Ablation Study）

在找到最终方案前，系统性地验证了以下方向，均未获得显著正收益：

| 方向 | 实验内容 | 结论 |
|---|---|---|
| ArcFace 参数调整 | 调节 margin(m)、scale(s)、embedding dim | 收益微小，本质瓶颈不在 loss |
| ConvNeXt-Small | 扩大 CNN backbone | 参数量增加但本地 F1 < 0.50 |
| ConvNeXt@384 | 提高输入分辨率 | 对 ConvNeXt 体系帮助有限 |
| DINOv2 kNN | 自监督特征 + kNN 检索 | 不适合细粒度 OCR，F1 < 0.30 |
| DINOv2 supervised | DINOv2 作为 backbone 微调 | 3483 类收敛困难，F1 < 0.40 |
| TTA | hflip、multi-pad 测试时增强 | 水平翻转对甲骨文有害（字形不对称），F1 下降 0.01~0.014 |
| Confidence rejection | 低置信度拒绝 | TP 下降幅度大于 FP 下降 |
| Page candidate rerank | 页面级候选重排 | 该数据集无收益 |
| Retrieval rerank | 基于特征检索的重排 | 性能退化 |
| Knowledge Distillation | 大模型蒸馏小模型 | Plan 3 中未观察到改善 |
| Swin-Small random head | ImageNet backbone + 新随机 head | E5 F1=0.5192，收敛过慢 |
| Swin-Small True Hybrid | Tiny E30 权重迁移初始化 | E10 F1=0.7100，离 Tiny E30 仍差 0.004 |

这些实验虽然未直接带来最终提升，但**帮助定位了系统瓶颈**：CNN backbone 的表示能力不足以处理 3483 类细粒度甲骨文分类，需要更强的主干网络。

### 7.3 Swin-Tiny + CE 成为关键突破

**关键决策**：将识别 backbone 从 ConvNeXt 切换为 Swin Transformer。

Swin Transformer 的分层注意力机制天然适合处理字符的局部结构和全局形态。与 ArcFace 不同，标准 Cross-Entropy loss + label smoothing（0.05）在实际训练中更稳定、易于调参。

**训练设置**：
- 模型：`swin_tiny_patch4_window7_224`（28.3M 参数）
- 输入：224×224，预处理 Resize(256) + CenterCrop(224)
- 优化器：AdamW，backbone lr=1e-5，head lr=1e-4，weight_decay=0.05
- 数据：50% original + 50% pipeline-style crop，WeightedRandomSampler
- 从 E20 开始记录，继续训练至 E25→E30

**线上结果提升**：

| 版本 | 识别模型 | Online F1 | Rec Acc | vs 上一版 |
|---|---|---|---|---|
| v9 (ConvNeXt baseline) | ConvNeXt-Tiny + ArcFace | 0.400821 | 0.441094 | — |
| v11 | Swin-Tiny E20 | 0.487599 | 0.536592 | **+0.0868** |
| v12 | Swin-Tiny E25 | 0.494840 | 0.544561 | +0.0072 |
| v13 | Swin-Tiny E30 | **0.495565** | **0.545358** | +0.0007 |

**总计**：从 v9 到 v13，Final F1 提升 **+0.0947**，Recognition Acc 提升 **+0.1043**。

---

## 8. 关键实验结果

### 8.1 线上提交记录

| Version | Detector | Recognizer | Online F1 | Detection F1 | Recognition Acc |
|---|---|---|---|---|---|
| v9 | YOLO11s | ConvNeXt-Tiny + ArcFace | 0.400821 | 0.908696 | 0.441094 |
| v11 | YOLO11s | Swin-Tiny E20 | 0.487599 | 0.908696 | 0.536592 |
| v12 | YOLO11s | Swin-Tiny E25 | 0.494840 | 0.908696 | 0.544561 |
| **v13** | YOLO11s | **Swin-Tiny E30** | **0.495565** | 0.908696 | **0.545358** |

### 8.2 Swin-Tiny 本地训练曲线

| Epoch | Local 931 Corrected F1 | Rec Acc | 说明 |
|---|---|---|---|
| E20 | 0.6714 | — | v11 提交版 |
| E25 | 0.6978 | 0.7948 | v12 提交版 |
| **E30** | **0.7141** | **0.8237** | **v13 提交版 (最佳)** |
| E31 | 0.7057 | 0.8087 | ↓ 回落 |
| E32 | 0.7105 | 0.8174 | ↓ 继续回落 → 停止 |

### 8.3 本地 vs 线上 Gap

- 本地 931 corrected F1：**0.7141**
- 线上 Final F1：**0.495565**
- Gap：**0.2185**（约 22 个百分点）

这个差距主要源于：
1. 本地 931 验证集与线上测试集的**分布不同**（不同拓片来源、不同标注质量）
2. 线上使用真实 detector 输出（有 FP），而本地 931 用 GT 标注匹配
3. E30→E35 阶段本地改善但线上几乎不变（+0.0007），表明**后期开始过拟合本地验证集**

---

## 9. 工程化与 Docker 提交

比赛要求以 Docker 镜像形式提交。主要工程工作：

1. **Docker 构建**：基于 `python:3.10-slim`，安装 PyTorch 2.5.1 (CUDA 12.4)、ultralytics、timm 等依赖
2. **文件挂载**：`/saisdata` 输入 → `/saisresult` 输出
3. **run.sh CRLF 修复**：Windows 下编辑的 shell 脚本换行符为 CRLF，Linux 容器中无法执行。使用 `sed` 转换为 LF
4. **timm 依赖修复**：SwinTransformer 需要 timm 库，初始 Dockerfile 遗漏该依赖，线上报 `ModuleNotFoundError`。补充 `pip install timm` 后修复
5. **镜像管理**：镜像约 43GB（含 PyTorch + 两个模型权重），推送到阿里云容器镜像服务 ACR
6. **版本管理**：每次提交使用独立 tag（v12_lf_fix → v12_timm_fix → v13_candidate），确保可回退

---

## 10. 项目难点

1. **字符类别多、样本长尾**：3483 个甲骨文字符类别，部分罕见字仅有几十个样本，长尾分布严重

2. **字形极度相似**：许多甲骨文字符之间仅有微小差异（一个笔画的弯曲方向不同），需要极强的细粒度识别能力

3. **检测 crop 与训练 crop 分布不一致**：YOLO 检测框的位置精度、padding 方式与原始标注的 crop 有差异，直接在原始数据上训练会导致上线退化

4. **本地验证与线上分布存在 gap**：本地 931 corrected F1 (0.7141) 与线上 Final F1 (0.4956) 差距达 0.22，本地验证集不能完全代表线上测试分布

5. **Docker 环境差异**：本地 Windows + WSL2 训练环境与线上 Linux Docker 环境不完全一致，需反复验证依赖完整性

6. **大量实验的版本管理**：多轮实验（ConvNeXt、DINOv2、Swin-Small、True Hybrid 等）需严格记录 checkpoint、参数、结果，避免覆盖有效版本

7. **识别是核心瓶颈**：检测 F1 (0.9087) 已接近天花板，任何提升都必须来自识别模型，而识别模型的改进远比检测参数调优复杂

---

## 11. 我的贡献

我独立完成了该项目的全套技术方案设计与实现：

- 搭建了完整的**检测 → 裁剪 → 识别两阶段 OCR pipeline**，从数据解析、模型训练、推理脚本到最终输出 prediction.json
- 完成 **YOLO11s 检测模型训练**，系统性搜索 conf/iou/max_det 等参数，确定最优推理配置
- 设计**动态 padding crop 策略**（pad = max(8, 0.08 × max(w,h))），平衡字符上下文与背景噪声
- 对 **ConvNeXt + ArcFace、DINOv2、Swin-Tiny、Swin-Small** 等多种识别 backbone 进行多轮实验，逐步定位识别瓶颈
- **发现并推动关键突破**：将识别模型从 ConvNeXt+ArcFace 切换为 Swin-Tiny+CE，线上 F1 提升 **+0.0868**，是项目最大单次提升
- 完成 **Docker 容器化**：编写 Dockerfile、run.sh，修复 CRLF 换行和 timm 依赖缺失等线上问题，推送至阿里云容器镜像仓库
- 维护 **GitHub 仓库**和实验记录，完成项目归档和版本管理

---

## 12. 项目成果

### 线上最佳成绩

| 指标 | 值 |
|---|---|
| Final F1 | **0.495565** |
| Detection F1 | 0.908696 |
| Recognition Acc | 0.545358 |
| TP / FP / FN | 4106 / 4129 / 4230 |
| Precision / Recall | 0.4986 / 0.4926 |

### 相比初始 Baseline 的提升

| 指标 | v9 (Baseline) | v13 (Final) | 提升 |
|---|---|---|---|
| Final F1 | 0.400821 | 0.495565 | **+0.0947** |
| Recognition Acc | 0.441094 | 0.545358 | **+0.1043** |

### 交付物

- 可复现的 Docker 推理镜像（阿里云 ACR）
- 完整 GitHub 项目仓库（代码 + 报告 + 配置）
- 实验复盘文档（有效/无效方向、训练曲线、提交记录）

---

## 13. 复盘与反思

### 做得好的

- **发现了识别瓶颈并果断切换 backbone**。ConvNeXt→Swin 的切换带来了最大单次提升（+0.0868），说明架构选择比小修小补更重要
- **建立了可靠的本地评测体系**。931 corrected pipeline eval 比单纯看 val set Top1 准确得多，帮助在本地筛选有效方案
- **版本控制和归档清晰**。每次提交都有独立 tag 和完整的实验记录，任何时候可以回退到历史版本

### 可以改进的

- **前期在 ConvNeXt/ArcFace 方向投入过多时间**。ArcFace 的各种调参收益微小（< 0.005），本质是 backbone 表示能力不够。如果更早切换到 Swin Transformer，可以节省约 2~3 天实验时间
- **本地验证集不能完全代表线上分布**。E25→E30 本地 F1 提升 0.0163，但线上仅提升 0.0007。如果更早构建与线上更接近的验证集（如更多的 pipeline-style evaluation），可以更准确地判断哪些改进是有效的
- **Swin-Small True Hybrid 的探索方式可以更高效**。E10 local F1=0.7100 离 Tiny E30=0.7141 仅差 0.004，但未能突破。如果做更多 epoch 或调整学习率策略，也许可以超越，但时间有限选择了停止

### 如果继续做

- **构建更接近线上分布的验证集**，减少 local-online gap
- **探索更强的识别 backbone**（Swin-Base、ConvNeXt V2 等）
- **半监督或自监督预训练**，利用大量未标注甲骨文图像
- **页面级上下文建模**：同一拓片上相邻字符之间可能存在语义关联
- **多模型融合**：训练多个不同初始化的 Swin-Tiny 进行投票

---

## 14. 简历版项目描述

### 中文简历版（4 bullets）

- 独立设计并实现甲骨文 OCR 两阶段 pipeline（YOLO11s 检测 + Swin-Tiny 识别），处理 3,483 类细粒度古文字分类任务
- 主导识别 backbone 从 ConvNeXt+ArcFace 切换至 Swin-Tiny+CE，线上 Final F1 提升至 0.4956（+0.0947），识别准确率提升至 54.5%
- 系统性完成检测参数搜索（conf/iou）与动态 crop padding 策略设计；对 DINOv2、ConvNeXt-Small、Swin-Small 等进行多方向实验验证
- 完成 Docker 容器化部署与阿里云镜像推送，修复 CRLF 换行、timm 依赖缺失等线上问题，实现可复现推理系统

### 英文简历版（4 bullets）

- Designed and implemented a two-stage OCR pipeline (YOLO11s detection + Swin-Tiny recognition) for 3,483-class oracle bone character classification
- Led the backbone migration from ConvNeXt+ArcFace to Swin-Tiny+CrossEntropy, improving online Final F1 to 0.4956 (+0.0947) and Recognition Accuracy to 54.5%
- Conducted systematic detection hyperparameter search and designed dynamic crop padding strategy; validated multiple recognition backbones including DINOv2, ConvNeXt variants, and Swin-Small through controlled ablation studies
- Containerized the inference system with Docker, resolved deployment issues (CRLF line endings, missing timm dependency), and pushed to Alibaba Cloud Container Registry for competition submission

### 面试口述版（1 分钟）

> 我参加了一个古文字识别的比赛，任务是从甲骨文拓片图像里检测和识别出每个文字。我搭了一个两阶段的系统：先用 YOLO 检测出文字位置，再用一个分类模型去识别具体是哪个字。
>
> 一开始我用 ConvNeXt 加 ArcFace loss 做识别，但发现准确率卡在 44% 左右上不去。后来我分析觉得 CNN 的局部感受野在处理这种三千多个长得特别像的字的时候有瓶颈，就把 backbone 换成了 Swin Transformer，loss 也简化成了普通的交叉熵。这个切换让线上 F1 直接从 0.40 跳到 0.49，是项目最大的提升。
>
> 后面继续训练到第 30 个 epoch，最终线上 F1 稳定在 0.4956，识别准确率到了 54.5%。整个过程我还做了 Docker 封装、修了几个线上部署的坑，最后完整推到了阿里云上。项目代码和实验记录都在 GitHub 上。
