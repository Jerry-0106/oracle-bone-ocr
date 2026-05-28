#!/bin/bash
# =============================================
# Competition OCR Pipeline
# Detection: YOLOv8n Character Detector (640px)
# Recognition: EfficientNet-B0 (1588 classes)
# Output: /saisresult/prediction.json
# =============================================
set -e

echo "============================================"
echo "  Oracle Bone Character OCR"
echo "  Detector: YOLOv8n @ ${IMGSZ:-640}px"
echo "  Recognizer: EfficientNet-B0"
echo "============================================"

DEVICE="${DEVICE:-cpu}"
export DEVICE
export CONF="${CONF:-0.32}"
export IOU="${IOU:-0.3}"
export IMGSZ="${IMGSZ:-640}"

echo "CONF=$CONF IOU=$IOU IMGSZ=$IMGSZ"

# Run competition inference
python /app/scripts/infer_competition.py

echo ""
echo "OCR Pipeline Complete"
echo "============================================"
