"""
OCR Pipeline Utilities.
"""

import json, os, time
from pathlib import Path
from datetime import datetime
from PIL import Image
import cv2
import numpy as np


def crop_character(img, bbox, padding=5):
    """Crop a character region from an image with padding.

    Args:
        img: numpy array (H, W, 3) BGR
        bbox: [x1, y1, x2, y2]
        padding: pixels of padding

    Returns:
        cropped PIL Image (RGB), or None if invalid
    """
    img_h, img_w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(img_w, x2 + padding)
    y2 = min(img_h, y2 + padding)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)


def load_image(image_path):
    """Load image as BGR numpy array."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    return img


def find_images(source_dir, exts=('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
    """Recursively find all image files in a directory."""
    source = Path(source_dir)
    if not source.is_dir():
        return []

    images = []
    for ext in exts:
        images.extend(source.rglob(f'*{ext}'))
        images.extend(source.rglob(f'*{ext.upper()}'))
    return sorted(set(images))


def save_outputs(results, output_dir, image_name):
    """Save OCR results as JSON and a formatted text file.

    Args:
        results: list of dicts (merged detection + recognition)
        output_dir: output directory
        image_name: source image name (without extension)

    Returns:
        path to JSON output file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / f"{image_name}_ocr.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "image": image_name,
            "timestamp": datetime.now().isoformat(),
            "num_detections": len(results),
            "detections": results,
        }, f, indent=2, ensure_ascii=False)

    # Text format
    txt_path = output_dir / f"{image_name}_ocr.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"OCR Results: {image_name}\n")
        f.write(f"{'='*60}\n")
        for i, det in enumerate(results):
            rec = det.get("recognition", {})
            top1 = rec.get("top1", {})
            f.write(f"[{i}] bbox={det['bbox']} "
                    f"det_conf={det['det_confidence']:.3f} "
                    f"char={top1.get('char', '?')} "
                    f"rec_conf={top1.get('confidence', 0):.3f}\n")

    return json_path


def compute_stats(detections):
    """Compute summary statistics from detection results."""
    if not detections:
        return {"total_detections": 0, "avg_det_conf": 0, "avg_rec_conf": 0,
                "high_conf_det": 0, "high_conf_rec": 0}

    det_confs = [d["det_confidence"] for d in detections]
    rec_confs = []
    for d in detections:
        rec = d.get("recognition", {})
        top1 = rec.get("top1", {})
        if top1.get("confidence"):
            rec_confs.append(top1["confidence"])

    return {
        "total_detections": len(detections),
        "avg_det_conf": round(np.mean(det_confs), 3),
        "avg_rec_conf": round(np.mean(rec_confs), 3) if rec_confs else 0,
        "high_conf_det": sum(1 for c in det_confs if c > 0.5),
        "high_conf_rec": sum(1 for c in rec_confs if c > 0.5) if rec_confs else 0,
    }
