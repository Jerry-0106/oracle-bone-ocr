#!/usr/bin/env python3
"""
Competition OCR Inference Script.
Reads images from /saisdata/13/eval/images/, outputs /saisresult/prediction.json.

Format: {"image_id": [{"bbox": [x, y, w, h], "text": "char"}, ...], ...}
"""
import json, os, sys, time
from pathlib import Path

import cv2
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"

# ── fixed random seeds ──
import random
random.seed(42)
np.random.seed(42)
import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

from ultralytics import YOLO

# ── paths ──
# __file__ is scripts/infer.py, so parent.parent = project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR  = Path(os.getenv("INPUT_DIR", str(PROJECT_ROOT / "demo")))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(PROJECT_ROOT)))
OUTPUT_JSON = OUTPUT_DIR / "prediction.json"

# ── model paths ──
# Phase 2: YOLO11s detector trained at 1280px (mAP50=0.84)
DETECTOR_PATH   = PROJECT_ROOT / "checkpoints" / "yolo11s_det_1280.pt"
# Phase 6: Swin-Tiny E30 + CE recognizer (corrected F1=0.7141, 3483 classes)
RECOGNIZER_PATH = PROJECT_ROOT / "checkpoints" / "swin_tiny_e30_v13_candidate.pt"
MAPPING_DIR     = PROJECT_ROOT / "mappings"

# ── inference config ──
CONF  = float(os.getenv("CONF", "0.20"))
IOU   = float(os.getenv("IOU", "0.25"))
IMGSZ = int(os.getenv("IMGSZ", "1280"))
DEVICE_ENV = os.getenv("DEVICE")  # None if not set
if DEVICE_ENV:
    # Explicit: use requested device, fallback if unavailable
    requested = DEVICE_ENV
    if requested == "cuda" and not torch.cuda.is_available():
        DEVICE = "cpu"
        print(f"[competition] WARNING: DEVICE=cuda requested but CUDA unavailable, falling back to CPU")
    else:
        DEVICE = requested
else:
    # Auto-detect: GPU first, CPU fallback
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[competition] DEVICE not set → auto-detected: {DEVICE}")
BATCH_SIZE = 32


def load_recognizer():
    """Load ConvNeXt+ArcFace recognizer and class mappings."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.recognizer import ArcFaceRecognizer
    return ArcFaceRecognizer(
        str(RECOGNIZER_PATH),
        mapping_dir=str(MAPPING_DIR),
        device=DEVICE,
    )


def find_images(directory):
    """Find all images in a directory (recursive)."""
    exts = {'.png', '.jpg', '.jpeg'}
    images = []
    for root, _, files in os.walk(directory):
        for f in sorted(files):
            if Path(f).suffix.lower() in exts:
                images.append(Path(root) / f)
    return images


def xyxy_to_xywh(bbox):
    """Convert [x1, y1, x2, y2] to [x, y, w, h]."""
    x1, y1, x2, y2 = bbox
    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]


def main():
    t0 = time.time()
    print(f"[competition] device={DEVICE} conf={CONF} iou={IOU} imgsz={IMGSZ}")

    # ── find images ──
    if not INPUT_DIR.exists():
        # fallback: try alternative paths
        alt_dirs = [
            Path("/saisdata"),
            Path("/input"),
        ]
        found = False
        for alt in alt_dirs:
            if alt.exists():
                images = find_images(alt)
                if images:
                    print(f"[competition] found {len(images)} images under {alt}")
                    found = True
                    break
        if not found:
            print(f"[competition] ERROR: input dir not found")
            sys.exit(1)
    else:
        images = find_images(INPUT_DIR)
        print(f"[competition] found {len(images)} images in {INPUT_DIR}")

    if not images:
        print("[competition] WARNING: no images found, writing empty prediction.json")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False)
        return

    # ── load models ──
    print(f"[competition] loading detector: {DETECTOR_PATH}")
    detector = YOLO(str(DETECTOR_PATH))

    print(f"[competition] loading recognizer: {RECOGNIZER_PATH}")
    recognizer = load_recognizer()

    # ── inference ──
    result = {}

    for img_idx, img_path in enumerate(images):
        image_id = img_path.stem  # filename without .png
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [{img_idx+1}/{len(images)}] SKIP {image_id} (unreadable)")
            result[image_id] = []
            continue

        img_h, img_w = img.shape[:2]

        # Detection
        det_results = detector(
            str(img_path), conf=CONF, iou=IOU, imgsz=IMGSZ,
            augment=True, max_det=300, device=DEVICE, verbose=False,
        )

        char_detections = []
        if det_results and det_results[0].boxes is not None:
            for box in det_results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                # strict clamp to image boundaries
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(img_w, int(x2))
                y2 = min(img_h, int(y2))
                w = x2 - x1
                h = y2 - y1
                if w <= 0 or h <= 0:
                    continue
                # filter abnormally large boxes (>30% of image dimension)
                if w > img_w * 0.3 or h > img_h * 0.3:
                    continue
                # filter tiny boxes (<5px)
                if w < 5 or h < 5:
                    continue
                char_detections.append({
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "confidence": conf,
                })

        # Crop and batch-recognize
        # Dynamic padding: larger bboxes get proportionally more context
        crops = []
        crop_metas = []
        for det in char_detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            w = x2 - x1
            h = y2 - y1

            # Dynamic padding: max(8px, 8% of max bbox dimension)
            pad = max(8, int(0.08 * max(w, h)))

            # Crop from original bbox + dynamic padding
            px1 = max(0, x1 - pad)
            py1 = max(0, y1 - pad)
            px2 = min(img_w, x2 + pad)
            py2 = min(img_h, y2 + pad)
            crop = img[py1:py2, px1:px2]
            if crop.size == 0:
                continue
            crops.append(crop)
            # Output bbox = original YOLO bbox (no shrink)
            crop_metas.append({"bbox_xyxy": [x1, y1, x2, y2]})

        # Batch recognition
        chars = []
        if crops:
            from PIL import Image
            pil_crops = []
            valid_indices = []
            for ci, crop in enumerate(crops):
                try:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil_crops.append(Image.fromarray(crop_rgb))
                    valid_indices.append(ci)
                except Exception:
                    continue

            if pil_crops:
                rec_results = recognizer.predict_batch(pil_crops, topk=1)
                for ri, (vi, meta) in enumerate(zip(valid_indices, [crop_metas[i] for i in valid_indices])):
                    top1 = rec_results[ri][0] if rec_results[ri] else None
                    if top1 and top1.get("char"):
                        chars.append({
                            "bbox": xyxy_to_xywh(meta["bbox_xyxy"]),  # [x, y, w, h] native int
                            "text": top1["char"],
                        })

        result[image_id] = chars

        if (img_idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{img_idx+1}/{len(images)}] {elapsed:.0f}s "
                  f"total_chars={sum(len(v) for v in result.values())}")

    # ── write output ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    total_chars = sum(len(v) for v in result.values())
    num_nonempty = sum(1 for v in result.values() if v)
    print(f"[competition] Done: {len(result)} images, {total_chars} chars, "
          f"{num_nonempty} non-empty, {elapsed:.0f}s")
    print(f"[competition] Output: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
