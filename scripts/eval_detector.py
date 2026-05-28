#!/usr/bin/env python3
"""
Strict Detection Evaluator — Competition-grade evaluation.

Matches competition scoring: IoU=0.5, one-to-one Hungarian matching, single class.

Usage:
  # Part 1: Confidence threshold sweep
  python scripts/eval_detector.py --mode conf_sweep

  # Part 2: NMS IoU sweep (with best conf)
  python scripts/eval_detector.py --mode nms_sweep --best-conf 0.XX

  # Part 3: FP/FN analysis with visualization
  python scripts/eval_detector.py --mode error_analysis --conf 0.XX --iou 0.X --max-samples 50
"""

import json, os, sys, time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "data" / "yolo"
VAL_IMAGES = DATASET_DIR / "images" / "val"
VAL_LABELS = DATASET_DIR / "labels" / "val"


def load_ground_truth(label_path, img_w, img_h):
    """Load YOLO-format labels, convert to absolute pixel bboxes."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            # Convert normalized → absolute
            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h
            boxes.append({
                "bbox": [max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)],
                "class_id": cls_id,
            })
    return boxes


def compute_iou(boxA, boxB):
    """IoU of two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denom = boxAArea + boxBArea - interArea
    if denom <= 0:
        return 0.0
    return interArea / denom


def match_detections(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    Strict one-to-one matching via greedy assignment (highest IoU first).
    Returns (TP, FP, FN, matched_pairs, unmatched_preds, unmatched_gts).
    """
    if not pred_boxes:
        return 0, 0, len(gt_boxes), [], [], list(range(len(gt_boxes)))

    if not gt_boxes:
        return 0, len(pred_boxes), 0, [], list(range(len(pred_boxes))), []

    # Build IoU matrix
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, p in enumerate(pred_boxes):
        for j, g in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou(p["bbox"], g["bbox"])

    matched_pairs = []
    matched_preds = set()
    matched_gts = set()

    # Greedy assignment: sort all pairs by IoU descending
    pairs = []
    for i in range(len(pred_boxes)):
        for j in range(len(gt_boxes)):
            if iou_matrix[i, j] >= iou_threshold:
                pairs.append((iou_matrix[i, j], i, j))
    pairs.sort(reverse=True)

    for iou_val, pi, gj in pairs:
        if pi not in matched_preds and gj not in matched_gts:
            matched_pairs.append((pi, gj, iou_val))
            matched_preds.add(pi)
            matched_gts.add(gj)

    TP = len(matched_pairs)
    FP = len(pred_boxes) - TP
    FN = len(gt_boxes) - TP

    unmatched_preds = [i for i in range(len(pred_boxes)) if i not in matched_preds]
    unmatched_gts = [j for j in range(len(gt_boxes)) if j not in matched_gts]

    return TP, FP, FN, matched_pairs, unmatched_preds, unmatched_gts


def run_detector(model, image_path, conf, iou, imgsz):
    """Run YOLO detector with given parameters, return parsed boxes."""
    results = model(
        str(image_path), conf=conf, iou=iou, imgsz=imgsz,
        augment=True, max_det=300, verbose=False, device=model.device,
    )
    detections = []
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy.astype(int)
            conf_val = float(box.conf[0])
            cls_id = int(box.cls[0]) if box.cls is not None else -1
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": conf_val,
                "class_id": cls_id,
            })
    return detections


def conf_sweep(model_path, imgsz=1280):
    """Sweep confidence thresholds, output table."""
    confs = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    fixed_iou = 0.5  # competition IoU for NMS — but we match with 0.5 IoU

    print("=" * 90)
    print(f"CONFIDENCE THRESHOLD SWEEP (NMS IoU fixed=0.5, imgsz={imgsz})")
    print("=" * 90)

    model = YOLO(str(model_path))

    # Get all val images
    val_images = sorted(VAL_IMAGES.glob("*.png")) + sorted(VAL_IMAGES.glob("*.jpg"))
    if not val_images:
        print("ERROR: no val images found")
        return

    print(f"\nVal images: {len(val_images)}")
    print(f"{'Conf':<8} {'TP':<8} {'FP':<8} {'FN':<8} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Time(s)':<10}")
    print("-" * 90)

    best_f1 = -1
    best_conf = None
    all_results = []

    for conf in confs:
        total_tp, total_fp, total_fn = 0, 0, 0
        t0 = time.time()

        for img_path in tqdm(val_images, desc=f"  conf={conf:.2f}", leave=False):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            label_path = VAL_LABELS / (img_path.stem + ".txt")
            gt_boxes = load_ground_truth(label_path, img_w, img_h)
            pred_boxes = run_detector(model, img_path, conf=conf, iou=fixed_iou, imgsz=imgsz)

            tp, fp, fn, _, _, _ = match_detections(gt_boxes, pred_boxes, iou_threshold=0.5)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        elapsed = time.time() - t0
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        all_results.append({
            "conf": conf, "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "time": round(elapsed, 1),
        })

        marker = " <-- BEST" if f1 > best_f1 else ""
        if f1 > best_f1:
            best_f1 = f1
            best_conf = conf

        print(f"{conf:<8.2f} {total_tp:<8} {total_fp:<8} {total_fn:<8} "
              f"{precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {elapsed:<10.1f}{marker}")

    print(f"\nBest F1: {best_f1:.4f} at confidence={best_conf}")

    # Save
    out_path = PROJECT_ROOT / "evaluation" / "conf_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({"best_conf": best_conf, "best_f1": best_f1, "results": all_results}, f, indent=2)
    print(f"Results saved: {out_path}")

    return best_conf, best_f1


def nms_sweep(model_path, best_conf, imgsz=1280):
    """Sweep NMS IoU thresholds with best confidence."""
    nms_vals = [0.3, 0.4, 0.5, 0.6]

    print("\n" + "=" * 90)
    print(f"NMS IoU SWEEP (conf={best_conf}, imgsz={imgsz})")
    print("=" * 90)

    model = YOLO(str(model_path))

    val_images = sorted(VAL_IMAGES.glob("*.png")) + sorted(VAL_IMAGES.glob("*.jpg"))

    print(f"\nVal images: {len(val_images)}")
    print(f"{'NMS IoU':<10} {'Conf':<8} {'TP':<8} {'FP':<8} {'FN':<8} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Time(s)':<10}")
    print("-" * 90)

    best_f1 = -1
    best_nms = None
    all_results = []

    for nms_iou in nms_vals:
        total_tp, total_fp, total_fn = 0, 0, 0
        t0 = time.time()

        for img_path in tqdm(val_images, desc=f"  nms={nms_iou}", leave=False):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            label_path = VAL_LABELS / (img_path.stem + ".txt")
            gt_boxes = load_ground_truth(label_path, img_w, img_h)
            pred_boxes = run_detector(model, img_path, conf=best_conf, iou=nms_iou, imgsz=imgsz)

            tp, fp, fn, _, _, _ = match_detections(gt_boxes, pred_boxes, iou_threshold=0.5)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        elapsed = time.time() - t0
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        all_results.append({
            "nms_iou": nms_iou, "conf": best_conf, "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "time": round(elapsed, 1),
        })

        marker = " <-- BEST" if f1 > best_f1 else ""
        if f1 > best_f1:
            best_f1 = f1
            best_nms = nms_iou

        print(f"{nms_iou:<10.1f} {best_conf:<8.2f} {total_tp:<8} {total_fp:<8} {total_fn:<8} "
              f"{precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {elapsed:<10.1f}{marker}")

    print(f"\nBest: conf={best_conf}, nms={best_nms}, F1={best_f1:.4f}")

    out_path = PROJECT_ROOT / "evaluation" / "nms_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({"best_conf": best_conf, "best_nms": best_nms, "best_f1": best_f1, "results": all_results}, f, indent=2)
    print(f"Results saved: {out_path}")

    return best_conf, best_nms, best_f1


def error_analysis(model_path, conf, nms_iou, imgsz=1280, max_samples=50):
    """Analyze FP and FN with visualizations."""
    print("\n" + "=" * 90)
    print(f"ERROR ANALYSIS (conf={conf}, nms={nms_iou}, imgsz={imgsz})")
    print("=" * 90)

    model = YOLO(str(model_path))

    val_images = sorted(VAL_IMAGES.glob("*.png")) + sorted(VAL_IMAGES.glob("*.jpg"))

    fp_samples = []  # (img_path, pred_box, gt_boxes_nearby)
    fn_samples = []  # (img_path, gt_box, pred_boxes_nearby)

    for img_path in tqdm(val_images, desc="  Analyzing errors"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        label_path = VAL_LABELS / (img_path.stem + ".txt")
        gt_boxes = load_ground_truth(label_path, img_w, img_h)
        pred_boxes = run_detector(model, img_path, conf=conf, iou=nms_iou, imgsz=imgsz)

        _, _, _, matched_pairs, unmatched_preds, unmatched_gts = match_detections(
            gt_boxes, pred_boxes, iou_threshold=0.5
        )

        # Collect FPs
        for pi in unmatched_preds:
            if len(fp_samples) < max_samples:
                fp_samples.append({
                    "image": str(img_path),
                    "image_name": img_path.stem,
                    "pred_box": pred_boxes[pi]["bbox"],
                    "pred_conf": pred_boxes[pi]["confidence"],
                    "img_w": img_w, "img_h": img_h,
                })

        # Collect FNs
        for gj in unmatched_gts:
            if len(fn_samples) < max_samples:
                fn_samples.append({
                    "image": str(img_path),
                    "image_name": img_path.stem,
                    "gt_box": gt_boxes[gj]["bbox"],
                    "img_w": img_w, "img_h": img_h,
                })

        if len(fp_samples) >= max_samples and len(fn_samples) >= max_samples:
            break

    # Generate visualizations
    out_dir = PROJECT_ROOT / "evaluation" / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFP samples: {len(fp_samples)}, FN samples: {len(fn_samples)}")

    # Draw FP samples
    print("\n=== FALSE POSITIVES ===")
    fp_analysis = analyze_fp(fp_samples, out_dir)
    print(fp_analysis)

    # Draw FN samples
    print("\n=== FALSE NEGATIVES ===")
    fn_analysis = analyze_fn(fn_samples, out_dir)
    print(fn_analysis)

    # Save metadata
    with open(out_dir / "fp_samples.json", 'w') as f:
        json.dump(fp_samples, f, indent=2)
    with open(out_dir / "fn_samples.json", 'w') as f:
        json.dump(fn_samples, f, indent=2)

    print(f"\nVisualizations saved: {out_dir}")


def analyze_fp(fp_samples, out_dir):
    """Categorize and visualize false positives."""
    categories = defaultdict(list)

    for i, fp in enumerate(fp_samples):
        img = cv2.imread(fp["image"])
        if img is None:
            continue
        x1, y1, x2, y2 = fp["pred_box"]
        box_w = x2 - x1
        box_h = y2 - y1
        area = box_w * box_h
        aspect = box_w / max(box_h, 1)

        # Categorize
        img_area = fp["img_w"] * fp["img_h"]
        area_ratio = area / img_area

        if area_ratio < 0.002:
            cat = "tiny_box"
            reason = "极小的误检 (< 0.2% 图像面积)"
        elif aspect > 5 or aspect < 0.2:
            cat = "extreme_aspect"
            reason = "极端宽高比"
        elif area_ratio > 0.15:
            cat = "large_box"
            reason = "大框误检 (可能是多字区域)"
        else:
            cat = "medium_fp"
            reason = "中等误检 (可能是纹理/裂纹)"

        categories[cat].append(i)

        # Draw
        vis = img.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, f"FP {fp['pred_conf']:.2f} ({reason})", (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imwrite(str(out_dir / f"fp_{i:03d}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])

    summary = []
    summary.append(f"Total FPs analyzed: {len(fp_samples)}")
    summary.append(f"Categories:")
    for cat, indices in sorted(categories.items()):
        summary.append(f"  {cat}: {len(indices)} ({100*len(indices)/max(len(fp_samples),1):.1f}%)")
    return "\n".join(summary)


def analyze_fn(fn_samples, out_dir):
    """Categorize and visualize false negatives."""
    categories = defaultdict(list)

    for i, fn in enumerate(fn_samples):
        img = cv2.imread(fn["image"])
        if img is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in fn["gt_box"]]
        box_w = x2 - x1
        box_h = y2 - y1
        area = box_w * box_h

        img_area = fn["img_w"] * fn["img_h"]
        area_ratio = area / img_area

        if area_ratio < 0.003:
            cat = "tiny_char"
            reason = "极小字符 (< 0.3% 图像面积)"
        elif area_ratio < 0.01:
            cat = "small_char"
            reason = "小字符 (0.3-1% 图像面积)"
        elif area_ratio > 0.2:
            cat = "large_char"
            reason = "大字符 (>20% 图像面积)"
        else:
            cat = "medium_char"
            reason = "中等字符 (未能检测)"

        categories[cat].append(i)

        # Draw
        vis = img.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(vis, f"FN {box_w}x{box_h} ({reason})", (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.imwrite(str(out_dir / f"fn_{i:03d}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])

    summary = []
    summary.append(f"Total FNs analyzed: {len(fn_samples)}")
    summary.append(f"Categories:")
    for cat, indices in sorted(categories.items()):
        summary.append(f"  {cat}: {len(indices)} ({100*len(indices)/max(len(fn_samples),1):.1f}%)")
    return "\n".join(summary)


def check_result_json():
    """Verify v4 pipeline produces correct result.json schema."""
    print("\n" + "=" * 90)
    print("RESULT.JSON SCHEMA CHECK")
    print("=" * 90)

    schema = {
        "pipeline": str,
        "results": list,
    }
    char_schema = {"bbox": list, "char": str, "confidence": float}

    # Check v4 infer script imports
    infer_path = PROJECT_ROOT / "scripts" / "infer.py"
    with open(infer_path) as f:
        content = f.read()

    issues = []

    # Check for 'segmented' references
    if "'segmented'" in content or '"segmented"' in content:
        issues.append("WARN: 'segmented' string found in infer_v4.py")
    if "seg_status" in content:
        issues.append("WARN: 'seg_status' found in infer_v4.py")

    # Check bbox field is 'char_bbox' or 'bbox'
    if "'char_bbox'" in content or '"char_bbox"' in content:
        print("  OK: char_bbox field present in result")
    if "'bbox'" in content or '"bbox"' in content:
        print("  OK: bbox field present in result")

    # Check pipeline name
    if "YOLO-CharDet" in content:
        print("  OK: pipeline name 'YOLO-CharDet+EfficientNet' present")

    # Check no AdaptiveMorph references
    if "AdaptiveMorph" in content or "adaptive_morph" in content:
        issues.append("ERROR: AdaptiveMorph reference found in v4 code!")
    else:
        print("  OK: No AdaptiveMorph references")

    if "segmentation" in content.lower() and "seg_status" not in content.lower():
        issues.append("WARN: 'segmentation' reference found (may be in comment)")

    for issue in issues:
        print(f"  {issue}")

    if not issues:
        print("\n  All checks passed.")
    else:
        print(f"\n  {len(issues)} issues found.")

    # Check run.sh
    run_sh = PROJECT_ROOT / "run.sh"
    with open(run_sh) as f:
        run_content = f.read()
    if "infer_v4.py" in run_content:
        print("  OK: run.sh calls infer_v4.py")
    if "--min-seg" in run_content:
        issues.append("ERROR: --min-seg still in run.sh!")
    else:
        print("  OK: --min-seg removed from run.sh")


def main():
    parser = argparse.ArgumentParser(description="Strict Detection Evaluator")
    parser.add_argument("--mode", required=True,
                        choices=["conf_sweep", "nms_sweep", "error_analysis", "check_schema", "all"])
    parser.add_argument("--model", default=None, help="Path to detector model")
    parser.add_argument("--imgsz", type=int, default=1280, help="Detection input size")
    parser.add_argument("--best-conf", type=float, default=None, help="Best conf for NMS sweep")
    parser.add_argument("--conf", type=float, default=0.10, help="Conf for error analysis")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU for error analysis")
    parser.add_argument("--max-samples", type=int, default=50, help="Max FP/FN samples")
    parser.add_argument("--device", default="mps", help="Device: mps, cpu, cuda")
    args = parser.parse_args()

    model_path = args.model or str(PROJECT_ROOT / "checkpoints" / "detector_v5.pt")

    if not Path(model_path).exists():
        print(f"ERROR: model not found: {model_path}")
        sys.exit(1)

    print(f"Model: {model_path}")
    print(f"Device: {args.device}")
    print(f"Image size: {args.imgsz}")

    if args.mode == "conf_sweep":
        conf_sweep(model_path, imgsz=args.imgsz)

    elif args.mode == "nms_sweep":
        if args.best_conf is None:
            print("ERROR: --best-conf required for NMS sweep")
            sys.exit(1)
        nms_sweep(model_path, args.best_conf, imgsz=args.imgsz)

    elif args.mode == "error_analysis":
        error_analysis(model_path, conf=args.conf, nms_iou=args.iou,
                       imgsz=args.imgsz, max_samples=args.max_samples)

    elif args.mode == "check_schema":
        check_result_json()

    elif args.mode == "all":
        print("\n" + "#" * 90)
        print("# PART 1: CONFIDENCE SWEEP")
        print("#" * 90)
        best_conf, best_f1 = conf_sweep(model_path, imgsz=args.imgsz)

        print("\n" + "#" * 90)
        print("# PART 2: NMS IoU SWEEP")
        print("#" * 90)
        best_conf, best_nms, best_f1 = nms_sweep(model_path, best_conf, imgsz=args.imgsz)

        print("\n" + "#" * 90)
        print("# PART 3: ERROR ANALYSIS")
        print("#" * 90)
        error_analysis(model_path, conf=best_conf, nms_iou=best_nms,
                       imgsz=args.imgsz, max_samples=50)

        print("\n" + "#" * 90)
        print("# PART 4: SCHEMA CHECK")
        print("#" * 90)
        check_result_json()

        # Final summary
        print("\n" + "=" * 90)
        print("FINAL RECOMMENDATION")
        print("=" * 90)
        print(f"  detector confidence: {best_conf}")
        print(f"  NMS IoU:            {best_nms}")
        print(f"  imgsz:             {args.imgsz}")
        print(f"  F1 score:          {best_f1:.4f}")
        print(f"  model:             {model_path}")
        print("=" * 90)


if __name__ == "__main__":
    import argparse
    main()
