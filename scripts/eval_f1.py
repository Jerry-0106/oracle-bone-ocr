#!/usr/bin/env python3
"""
F1 Evaluation Script for OCR Detection.
Matches predicted character bboxes against YOLO ground truth labels.
TP = correctly detected chars (IoU > 0.5 match)
FP = extra/spurious detections (no matching GT)
FN = missed chars (GT with no matching prediction)
F1 = 2*TP / (2*TP + FP + FN)
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import cv2


def parse_yolo_label(label_path, img_w, img_h):
    """Parse a YOLO-format label file into absolute bboxes [x1, y1, x2, y2]."""
    bboxes = []
    if not label_path.exists():
        return bboxes
    with open(label_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id, cx, cy, w, h = map(float, parts)
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)
            bboxes.append([x1, y1, x2, y2])
    return bboxes


def compute_iou(box_a, box_b):
    """Compute Intersection over Union between two bboxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    area_a = max(0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def match_boxes(gt_boxes, pred_boxes, iou_thresh=0.5):
    """
    Greedy IoU matching.
    Returns (tp, fp, fn, matches) where matches = [(gt_idx, pred_idx, iou), ...]
    """
    # Build IoU matrix
    ious = []
    for gi, gb in enumerate(gt_boxes):
        for pi, pb in enumerate(pred_boxes):
            iou = compute_iou(gb, pb)
            if iou >= iou_thresh:
                ious.append((iou, gi, pi))

    ious.sort(key=lambda x: x[0], reverse=True)

    matched_gt = set()
    matched_pred = set()
    matches = []

    for iou, gi, pi in ious:
        if gi not in matched_gt and pi not in matched_pred:
            matched_gt.add(gi)
            matched_pred.add(pi)
            matches.append((gi, pi, iou))

    tp = len(matches)
    fp = len(pred_boxes) - len(matched_pred)
    fn = len(gt_boxes) - len(matched_gt)

    return tp, fp, fn, matches


def compute_metrics(total_tp, total_fp, total_fn):
    """Compute Precision, Recall, F1 from aggregate counts."""
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR detection F1")
    parser.add_argument("--pred-dir", required=True, help="Directory with *_ocr.json prediction files")
    parser.add_argument("--gt-dir", required=True, help="Directory with YOLO label .txt files")
    parser.add_argument("--img-dir", required=True, help="Directory with original images (for dimensions)")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold (default: 0.5)")
    parser.add_argument("--output", default=None, help="Optional path to save per-image report JSON")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    img_dir = Path(args.img_dir)

    pred_files = sorted(pred_dir.glob("*_ocr.json"))
    if not pred_files:
        print(f"ERROR: No *_ocr.json files found in {pred_dir}")
        return

    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_image = []

    for pred_path in pred_files:
        image_name = pred_path.stem.replace("_ocr", "")
        gt_path = gt_dir / f"{image_name}.txt"

        with open(pred_path, encoding='utf-8') as f:
            pred_data = json.load(f)

        pred_boxes = [d["char_bbox"] for d in pred_data.get("detections", [])]

        # Get actual image dimensions
        img = cv2.imread(str(img_dir / f"{image_name}.png"))
        if img is None:
            img = cv2.imread(str(img_dir / f"{image_name}.jpg"))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        gt_boxes = parse_yolo_label(gt_path, img_w, img_h)

        tp, fp, fn, matches = match_boxes(gt_boxes, pred_boxes, args.iou)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision, recall, f1 = compute_metrics(tp, fp, fn)

        per_image.append({
            "image": image_name,
            "gt_chars": len(gt_boxes),
            "pred_chars": len(pred_boxes),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })

    agg_precision, agg_recall, agg_f1 = compute_metrics(total_tp, total_fp, total_fn)

    # Print report
    print(f"\n{'='*70}")
    print(f"  OCR Detection F1 Evaluation (IoU@{args.iou})")
    print(f"{'='*70}")
    print(f"  Images evaluated:        {len(per_image)}")
    print(f"  Ground truth characters: {total_tp + total_fn}")
    print(f"  Predicted characters:    {total_tp + total_fp}")
    print(f"  True Positives (TP):     {total_tp}")
    print(f"  False Positives (FP):    {total_fp}")
    print(f"  False Negatives (FN):    {total_fn}")
    print(f"  {'─'*50}")
    print(f"  Precision:  {agg_precision:.4f}  ({agg_precision*100:.2f}%)")
    print(f"  Recall:     {agg_recall:.4f}  ({agg_recall*100:.2f}%)")
    print(f"  F1 Score:   {agg_f1:.4f}  ({agg_f1*100:.2f}%)")
    print(f"{'='*70}")

    # Per-image breakdown for worst/best F1
    per_image.sort(key=lambda x: x["f1"])
    print(f"\n  Bottom 10 images (lowest F1):")
    for r in per_image[:10]:
        print(f"    {r['image']}: GT={r['gt_chars']} Pred={r['pred_chars']} "
              f"TP={r['tp']} FP={r['fp']} FN={r['fn']} F1={r['f1']:.4f}")

    print(f"\n  Top 10 images (highest F1):")
    for r in per_image[-10:]:
        print(f"    {r['image']}: GT={r['gt_chars']} Pred={r['pred_chars']} "
              f"TP={r['tp']} FP={r['fp']} FN={r['fn']} F1={r['f1']:.4f}")

    # F1 distribution
    f1_bands = defaultdict(int)
    for r in per_image:
        if r["gt_chars"] == 0 and r["pred_chars"] == 0:
            f1_bands["perfect (0/0)"] += 1
        elif r["f1"] == 1.0:
            f1_bands["F1=1.0"] += 1
        elif r["f1"] >= 0.8:
            f1_bands["F1>=0.8"] += 1
        elif r["f1"] >= 0.5:
            f1_bands["F1>=0.5"] += 1
        elif r["f1"] > 0:
            f1_bands["F1>0"] += 1
        else:
            f1_bands["F1=0"] += 1

    print(f"\n  F1 Distribution:")
    for band in ["perfect (0/0)", "F1=1.0", "F1>=0.8", "F1>=0.5", "F1>0", "F1=0"]:
        if band in f1_bands:
            print(f"    {band}: {f1_bands[band]} images")

    if args.output:
        report = {
            "iou_threshold": args.iou,
            "aggregate": {
                "total_images": len(per_image),
                "total_gt": total_tp + total_fn,
                "total_pred": total_tp + total_fp,
                "tp": total_tp, "fp": total_fp, "fn": total_fn,
                "precision": round(agg_precision, 4),
                "recall": round(agg_recall, 4),
                "f1": round(agg_f1, 4),
            },
            "per_image": per_image,
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  Report saved to: {args.output}")


if __name__ == "__main__":
    main()
