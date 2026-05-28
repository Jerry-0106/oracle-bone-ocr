#!/usr/bin/env python3
"""
Competition-grade evaluator — simulates official evaluate.py.

Matches competition scoring:
  - IoU >= 0.5 for detection match
  - Text exact match for TP
  - Greedy one-to-one matching
"""
import json, argparse
from pathlib import Path
from collections import defaultdict


def parse_yolo_label(label_path, img_w, img_h):
    """Parse YOLO-format label: class cx cy w h (normalized). Return [x1,y1,x2,y2]."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy, w, h = map(float, parts[1:5])
            x1 = int((cx - w/2) * img_w)
            y1 = int((cy - h/2) * img_h)
            x2 = int((cx + w/2) * img_w)
            y2 = int((cy + h/2) * img_h)
            boxes.append([max(0,x1), max(0,y1), min(img_w,x2), min(img_h,y2)])
    return boxes


def compute_iou(a, b):
    """IoU between two [x1,y1,x2,y2] boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(0, (a[2]-a[0]) * (a[3]-a[1]))
    area_b = max(0, (b[2]-b[0]) * (b[3]-b[1]))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def xywh_to_xyxy(bbox):
    """Convert [x,y,w,h] to [x1,y1,x2,y2]."""
    x, y, w, h = bbox
    return [x, y, x+w, y+h]


def match_image(gt_boxes, pred_items, iou_thresh=0.5):
    """
    Match predictions to ground truth.
    Returns (tp, fp, fn, details).
    """
    # Build IoU matrix for all pairs
    pairs = []
    for pi, pred in enumerate(pred_items):
        pred_box = xywh_to_xyxy(pred["bbox"])
        for gi, gt_box in enumerate(gt_boxes):
            iou = compute_iou(pred_box, gt_box)
            if iou >= iou_thresh:
                pairs.append((iou, pi, gi))

    pairs.sort(key=lambda x: x[0], reverse=True)

    matched_pred = set()
    matched_gt = set()
    tp = 0
    tp_details = []

    for iou, pi, gi in pairs:
        if pi not in matched_pred and gi not in matched_gt:
            matched_pred.add(pi)
            matched_gt.add(gi)
            # Check text match (skip if GT has no text, e.g. YOLO labels)
            gt_entry = gt_boxes[gi]
            if isinstance(gt_entry, dict) and "text" in gt_entry:
                if pred_items[pi]["text"] != gt_entry["text"]:
                    continue  # text mismatch, not a TP
            tp += 1
            tp_details.append({
                "pred_idx": pi,
                "gt_idx": gi,
                "iou": round(iou, 4),
                "pred_text": pred_items[pi]["text"],
            })

    fp = len(pred_items) - len(matched_pred)
    fn = len(gt_boxes) - len(matched_gt)

    return tp, fp, fn, tp_details


def main():
    parser = argparse.ArgumentParser(description="Competition-grade OCR evaluator")
    parser.add_argument("--prediction", required=True, help="Path to prediction.json")
    parser.add_argument("--gt-dir", required=True, help="Directory with YOLO label .txt files")
    parser.add_argument("--img-dir", required=True, help="Directory with original images")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--output", default=None, help="Save report JSON")
    args = parser.parse_args()

    import cv2

    with open(args.prediction, encoding='utf-8') as f:
        predictions = json.load(f)

    gt_dir = Path(args.gt_dir)
    img_dir = Path(args.img_dir)

    total_tp = total_fp = total_fn = 0
    per_image = []

    for image_id, pred_items in predictions.items():
        # Try to find image and GT
        gt_path = gt_dir / f"{image_id}.txt"
        img_path = img_dir / f"{image_id}.png"
        if not img_path.exists():
            img_path = img_dir / f"{image_id}.jpg"

        if not gt_path.exists() or not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        gt_boxes = parse_yolo_label(gt_path, img_w, img_h)
        # Note: YOLO labels don't have text, so we skip text matching
        # In real competition, GT has text annotations

        tp, fp, fn, _ = match_image(gt_boxes, pred_items)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if tp + fp + fn > 0:
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_img = 2*p*r/(p+r) if (p+r) > 0 else 0
            per_image.append({
                "image": image_id,
                "gt": len(gt_boxes), "pred": len(pred_items),
                "tp": tp, "fp": fp, "fn": fn, "f1": round(f1_img, 4),
            })

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'='*60}")
    print(f"  Competition Evaluation (IoU@{args.iou})")
    print(f"{'='*60}")
    print(f"  Images:          {len(per_image)}")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"{'='*60}")

    # F1 distribution
    dist = defaultdict(int)
    for r in per_image:
        if r["f1"] == 1.0:
            dist["F1=1.0"] += 1
        elif r["f1"] >= 0.5:
            dist["F1>=0.5"] += 1
        elif r["f1"] > 0:
            dist["F1>0"] += 1
        else:
            dist["F1=0"] += 1
    print(f"\n  F1 Distribution:")
    for k in ["F1=1.0", "F1>=0.5", "F1>0", "F1=0"]:
        if k in dist:
            print(f"    {k}: {dist[k]} images")

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({
                "iou_threshold": args.iou,
                "tp": total_tp, "fp": total_fp, "fn": total_fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "per_image": per_image,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n  Report: {args.output}")


if __name__ == "__main__":
    main()
