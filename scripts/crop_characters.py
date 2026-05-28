#!/usr/bin/env python3
"""
Crop Characters for Recognition Pipeline.
Uses detector to crop characters from images, or uses ground truth boxes.
Output organized for recognition training.

Usage:
  python crop_characters.py --source ./images/ --output recognition_dataset/
  python crop_characters.py --source ./images/ --output recognition_dataset/ --use-gt
  python crop_characters.py --source ./images/ --output recognition_dataset/ --padding 10
"""

import os, sys, json, argparse
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PT = PROJECT_ROOT / "checkpoints" / "detector_v5.pt"
VAL_DIR = PROJECT_ROOT / "data" / "yolo" / "images" / "val"
LABEL_DIR = PROJECT_ROOT / "data" / "yolo" / "labels" / "val"

CONFIG = {"conf": 0.10, "iou": 0.3, "imgsz": 640, "augment": True, "device": "mps", "max_det": 300, "verbose": False}
PADDING = 5  # pixels of padding around crop

os.environ["OMP_NUM_THREADS"] = "1"


def parse_xml_utf16(xml_path):
    """Parse UTF-16 XML annotation, return list of bboxes."""
    boxes = []
    try:
        with open(xml_path, 'r', encoding='utf-16') as f:
            content = f.read()
    except Exception:
        return boxes

    import re
    # Match position="x1,y1,x2,y2"
    positions = re.findall(r'position="(\d+),(\d+),(\d+),(\d+)"', content)
    for match in positions:
        x1, y1, x2, y2 = map(int, match)
        boxes.append((x1, y1, x2, y2))
    return boxes


def crop_and_save(img, bbox, output_dir, img_name, char_idx, padding=5):
    """Crop a character from the image and save it."""
    img_h, img_w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(img_w, x2 + padding)
    y2 = min(img_h, y2 + padding)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    crop_name = f"{img_name}_{char_idx:03d}.jpg"
    crop_path = output_dir / crop_name
    cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return crop_name


def run_detector_crop(model, source_dir, output_dir, config, padding=5):
    """Crop characters using detector predictions."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    crops_dir = output_dir / "crops"
    vis_dir = output_dir / "visualization"
    crops_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted([f for f in os.listdir(source_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print(f"Processing {len(img_files)} images with detector...")
    metadata = []
    stats = {"total_crops": 0, "small_crops": 0, "images_with_detections": 0}

    for i, img_name in enumerate(img_files):
        img_path = source_dir / img_name
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        base = os.path.splitext(img_name)[0]

        results = model(img, **config)
        if not results or len(results) == 0 or results[0].boxes is None:
            continue

        boxes = results[0].boxes
        if len(boxes) == 0:
            continue

        stats["images_with_detections"] += 1
        vis = img.copy()

        for j, box in enumerate(boxes):
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            conf = float(box.conf[0])
            bw, bh = x2 - x1, y2 - y1

            if bw < 10 or bh < 10:
                stats["small_crops"] += 1

            crop_name = crop_and_save(img, (x1, y1, x2, y2), crops_dir, base, j, padding)
            if crop_name:
                stats["total_crops"] += 1
                metadata.append({
                    "crop_file": crop_name,
                    "source_image": img_name,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": round(conf, 4),
                    "crop_size": [int(bw), int(bh)],
                    "detector_pred": True,
                })

                # Draw on vis
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"{j}", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        if i < 50:
            cv2.imwrite(str(vis_dir / f"{base}_crops.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(img_files)}, crops: {stats['total_crops']}")

    print(f"\nCrop Summary: {stats['total_crops']} crops from {stats['images_with_detections']} images")
    print(f"  Small crops (<10px): {stats['small_crops']}")

    # Save metadata
    with open(output_dir / "metadata" / "crop_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Analyze crop quality
    sizes = [m["crop_size"] for m in metadata]
    if sizes:
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        areas = [w * h for w, h in sizes]
        quality = {
            "total_crops": len(metadata),
            "avg_width": round(np.mean(widths), 1),
            "avg_height": round(np.mean(heights), 1),
            "min_size": [int(np.min(widths)), int(np.min(heights))],
            "max_size": [int(np.max(widths)), int(np.max(heights))],
            "pct_small_20px": round(sum(1 for a in areas if a < 400) / len(areas) * 100, 1),
            "pct_large_100px": round(sum(1 for a in areas if a > 10000) / len(areas) * 100, 1),
        }
        with open(output_dir / "metadata" / "crop_quality.json", 'w') as f:
            json.dump(quality, f, indent=2)
        print(f"\nAvg crop: {quality['avg_width']}x{quality['avg_height']} px")
        print(f"  <20x20px: {quality['pct_small_20px']}%")
        print(f"  >100x100px: {quality['pct_large_100px']}%")

    return metadata


def run_gt_crop(source_dir, output_dir, padding=5):
    """Crop characters using ground truth bounding boxes."""
    source_dir = Path(source_dir)
    label_dir = source_dir  # XML files are in the same dir
    output_dir = Path(output_dir)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Find images with corresponding XML
    img_files = []
    for f in sorted(os.listdir(source_dir)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            xml_path = source_dir / (os.path.splitext(f)[0] + ".xml")
            if xml_path.exists():
                img_files.append(f)

    print(f"Processing {len(img_files)} images with GT boxes...")
    metadata = []
    stats = {"total_crops": 0}

    for i, img_name in enumerate(img_files):
        img_path = source_dir / img_name
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        base = os.path.splitext(img_name)[0]
        xml_path = source_dir / (base + ".xml")

        gt_boxes = parse_xml_utf16(xml_path)
        for j, bbox in enumerate(gt_boxes):
            x1, y1, x2, y2 = bbox
            crop_name = crop_and_save(img, bbox, crops_dir, base, j, padding)
            if crop_name:
                stats["total_crops"] += 1
                metadata.append({
                    "crop_file": crop_name,
                    "source_image": img_name,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "crop_size": [int(x2 - x1), int(y2 - y1)],
                    "detector_pred": False,
                })

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(img_files)}, crops: {stats['total_crops']}")

    print(f"\nTotal: {stats['total_crops']} crops from {len(img_files)} images")

    with open(output_dir / "metadata" / "crop_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Crop Characters for Recognition")
    parser.add_argument("--source", required=True, help="Directory with images")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--use-gt", action="store_true", help="Use GT boxes instead of detector")
    parser.add_argument("--padding", type=int, default=5, help="Padding pixels around crop")
    parser.add_argument("--conf", type=float, default=None, help="Detection confidence threshold")
    args = parser.parse_args()

    output_dir = Path(args.output)
    for sub in ["crops", "metadata", "visualization"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    config = CONFIG.copy()
    if args.conf is not None:
        config["conf"] = args.conf

    if args.use_gt:
        run_gt_crop(args.source, output_dir, args.padding)
    else:
        if not MODEL_PT.exists():
            print(f"ERROR: Model not found at {MODEL_PT}")
            print("Use --use-gt to crop with ground truth boxes instead.")
            sys.exit(1)
        print(f"Loading detector: {MODEL_PT}")
        model = YOLO(str(MODEL_PT))
        run_detector_crop(model, args.source, output_dir, config, args.padding)


if __name__ == "__main__":
    main()
