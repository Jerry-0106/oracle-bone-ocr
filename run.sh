#!/bin/bash
# =============================================
# Competition OCR Pipeline
# Detection: YOLO11s @ 1280px (mAP50=0.84)
# Recognition: Swin-Tiny @224 + CE (Top1=74.96%, 3483 classes)
# Output: /saisresult/prediction.json
# =============================================
set -e

echo "============================================"
echo "  Oracle Bone Character OCR"
echo "  Detector: YOLO11s @ ${IMGSZ:-1280}px"
echo "  Recognizer: Swin-Tiny + CE"
echo "============================================"

# DEVICE: 不强制默认值 — 留给 infer.py 自动检测 GPU。
# 如果平台显式设置 DEVICE=cpu 或 DEVICE=cuda，则透传。
# 如果未设置，infer.py 会通过 torch.cuda.is_available() 自动选择。
if [ -n "${DEVICE}" ]; then
    export DEVICE
    echo "DEVICE=$DEVICE (from env)"
else
    echo "DEVICE=auto (will detect GPU via torch.cuda.is_available())"
fi

export CONF="${CONF:-0.20}"
export IOU="${IOU:-0.25}"
export IMGSZ="${IMGSZ:-1280}"

# Competition platform paths
export INPUT_DIR="${INPUT_DIR:-/saisdata}"
export OUTPUT_DIR="${OUTPUT_DIR:-/saisresult}"

echo "CONF=$CONF IOU=$IOU IMGSZ=$IMGSZ"
echo "INPUT_DIR=$INPUT_DIR OUTPUT_DIR=$OUTPUT_DIR"

# Run full detection + recognition pipeline
python /app/scripts/infer.py

echo ""
echo "OCR Pipeline Complete"
echo "============================================"
