#!/usr/bin/env python3
"""
Competition-Oriented Optimization for Ancient Character OCR.
Focus: Maximize OCR F1, not just detection mAP.

Phases:
  1. Error analysis visualization
  2. Inference parameter grid search
  3. TTA testing
  4. Hard negative analysis
  5. Convergence analysis

Usage:
  python optimize.py --phase 1    # Error visualization
  python optimize.py --phase 2    # Parameter search
  python optimize.py --phase 3    # TTA testing
  python optimize.py --phase 4    # Hard negative analysis
  python optimize.py --phase 5    # Convergence analysis
  python optimize.py --phase all  # Run all phases
"""

import os
import sys
import json
import random
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
BEST_PT = PROJECT_ROOT / "runs" / "fast_baseline" / "weights" / "best.pt"
VAL_IMG_DIR = PROJECT_ROOT / "dataset_yolo" / "images" / "val"
VAL_LBL_DIR = PROJECT_ROOT / "dataset_yolo" / "labels" / "val"
OPT_DIR = PROJECT_ROOT / "optimization"
ERROR_VIS_DIR = OPT_DIR / "error_vis"
PARAM_SEARCH_DIR = OPT_DIR / "param_search"
TTA_DIR = OPT_DIR / "tta_results"
HN_DIR = OPT_DIR / "hard_neg"
REPORT_DIR = OPT_DIR / "reports"
RESULTS_CSV = PROJECT_ROOT / "runs" / "fast_baseline" / "results.csv"

for d in [ERROR_VIS_DIR, PARAM_SEARCH_DIR, TTA_DIR, HN_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Set random seed
np.random.seed(42)
random.seed(42)

# Color definitions for visualization
COLOR_GT = (0, 255, 0)      # Green: ground truth
COLOR_PRED = (255, 0, 0)    # Blue: prediction
COLOR_TP = (0, 255, 255)    # Yellow: true positive
COLOR_FP = (0, 0, 255)      # Red: false positive
COLOR_FN = (0, 165, 255)    # Orange: false negative


def compute_iou(box1, box2):
    """Compute IoU between two boxes in xyxy format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def xywh2xyxy(box, img_w, img_h):
    """Convert YOLO normalized xywh to pixel xyxy."""
    xc, yc, w, h = box
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return [x1, y1, x2, y2]


def load_gt_boxes(label_path, img_w, img_h):
    """Load ground truth boxes from YOLO label file."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                xyxy = xywh2xyxy([xc, yc, w, h], img_w, img_h)
                boxes.append({'cls': cls_id, 'bbox': xyxy, 'xywh': [xc, yc, w, h]})
    return boxes


def match_predictions(pred_boxes, gt_boxes, iou_thresh=0.5):
    """
    Match predictions to ground truth.
    Returns: tp_preds, fp_preds, matched_gt_indices, unmatched_gt_indices
    """
    matched_gt = set()
    tp_preds = []
    fp_preds = []

    # Sort predictions by confidence (descending)
    sorted_preds = sorted(enumerate(pred_boxes), key=lambda x: x[1].get('conf', 0), reverse=True)

    for pred_idx, pred in sorted_preds:
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_thresh:
            tp_preds.append((pred_idx, pred, best_gt_idx, best_iou))
            matched_gt.add(best_gt_idx)
        else:
            fp_preds.append((pred_idx, pred))

    # Unmatched GT = FN
    fn_boxes = [gt for i, gt in enumerate(gt_boxes) if i not in matched_gt]

    return tp_preds, fp_preds, fn_boxes


def draw_boxes(img, boxes, color, label_prefix="", thickness=2):
    """Draw boxes on image."""
    for box_info in boxes:
        if isinstance(box_info, tuple):
            bbox = box_info[1]['bbox'] if len(box_info) > 1 else box_info[0]['bbox']
        else:
            bbox = box_info['bbox']

        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img.shape[1]-1, x2); y2 = min(img.shape[0]-1, y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


# ================================================================
# PHASE 1: Error Analysis Visualization
# ================================================================
def phase1_error_analysis(num_samples=50, conf=0.25, iou=0.5):
    """Run inference on val set samples and visualize errors."""
    print("=" * 60)
    print("PHASE 1: Error Analysis Visualization")
    print("=" * 60)

    if not BEST_PT.exists():
        print(f"ERROR: Model not found at {BEST_PT}")
        return

    model = YOLO(str(BEST_PT))

    # Get val image list
    val_images = sorted([f for f in os.listdir(VAL_IMG_DIR) if f.lower().endswith(('.png', '.jpg'))])
    if len(val_images) > num_samples:
        sampled = random.sample(val_images, num_samples)
    else:
        sampled = val_images

    print(f"Analyzing {len(sampled)} validation images...")

    all_results = []
    error_stats = {
        'small_miss': 0,      # Small target missed (< 5% image area)
        'dense_miss': 0,      # Dense region missed
        'edge_miss': 0,       # Edge missed
        'crack_fp': 0,        # Crack false positive
        'noise_fp': 0,        # Noise false positive
        'duplicate_fp': 0,    # Duplicate detection
        'total_gt': 0,
        'total_pred': 0,
        'total_tp': 0,
        'total_fp': 0,
        'total_fn': 0,
    }

    for idx, img_name in enumerate(tqdm(sampled, desc="Analyzing errors")):
        img_path = VAL_IMG_DIR / img_name
        base = os.path.splitext(img_name)[0]
        lbl_path = VAL_LBL_DIR / (base + ".txt")

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]
        img_area = img_w * img_h

        # Load GT
        gt_boxes = load_gt_boxes(lbl_path, img_w, img_h)
        error_stats['total_gt'] += len(gt_boxes)

        # Run inference
        results = model(img, conf=conf, iou=iou, verbose=False)
        pred_boxes = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    conf_val = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    pred_boxes.append({'cls': cls_id, 'bbox': xyxy, 'conf': conf_val})

        error_stats['total_pred'] += len(pred_boxes)

        # Match predictions to GT
        tp_preds, fp_preds, fn_boxes = match_predictions(pred_boxes, gt_boxes, iou_thresh=iou)
        error_stats['total_tp'] += len(tp_preds)
        error_stats['total_fp'] += len(fp_preds)
        error_stats['total_fn'] += len(fn_boxes)

        # Analyze error types
        # Small target FN (< 5% image area)
        for fn_box in fn_boxes:
            area = (fn_box['bbox'][2] - fn_box['bbox'][0]) * (fn_box['bbox'][3] - fn_box['bbox'][1])
            if area < img_area * 0.05:
                error_stats['small_miss'] += 1
            # Edge check (near border)
            x1, y1, x2, y2 = fn_box['bbox']
            if x1 < img_w * 0.05 or y1 < img_h * 0.05 or x2 > img_w * 0.95 or y2 > img_h * 0.95:
                error_stats['edge_miss'] += 1

        # Save visualization for first 20 samples
        if idx < 20:
            vis_img = img.copy()

            # Draw GT (green)
            for gt in gt_boxes:
                x1, y1, x2, y2 = [int(v) for v in gt['bbox']]
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), COLOR_GT, 2)

            # Draw FP (red)
            for _, fp in fp_preds:
                x1, y1, x2, y2 = [int(v) for v in fp['bbox']]
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), COLOR_FP, 2)

            # Draw FN (orange, dashed overlay)
            for fn in fn_boxes:
                x1, y1, x2, y2 = [int(v) for v in fn['bbox']]
                # Draw dashed line by using polylines
                pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
                cv2.polylines(vis_img, [pts], True, COLOR_FN, 2)

            # Draw TP (yellow)
            for _, tp, _, _ in tp_preds:
                x1, y1, x2, y2 = [int(v) for v in tp['bbox']]
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), COLOR_TP, 1)

            # Add legend and stats
            cv2.putText(vis_img, f"GT:{len(gt_boxes)} TP:{len(tp_preds)} FP:{len(fp_preds)} FN:{len(fn_boxes)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # Legend
            cv2.putText(vis_img, "GT", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GT, 1)
            cv2.putText(vis_img, "FP", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_FP, 1)
            cv2.putText(vis_img, "FN", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_FN, 1)
            cv2.putText(vis_img, "TP", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TP, 1)

            out_path = ERROR_VIS_DIR / f"error_{idx:03d}_{base}.jpg"
            cv2.imwrite(str(out_path), vis_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

        all_results.append({
            'image': img_name,
            'gt_count': len(gt_boxes),
            'pred_count': len(pred_boxes),
            'tp': len(tp_preds),
            'fp': len(fp_preds),
            'fn': len(fn_boxes),
        })

    # Compute overall metrics
    total_tp = error_stats['total_tp']
    total_fp = error_stats['total_fp']
    total_fn = error_stats['total_fn']
    total_gt = error_stats['total_gt']

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'='*60}")
    print(f"Error Analysis Results (conf={conf}, iou={iou})")
    print(f"{'='*60}")
    print(f"  Total GT:       {total_gt}")
    print(f"  Total Predictions: {error_stats['total_pred']}")
    print(f"  True Positives: {total_tp}")
    print(f"  False Positives:{total_fp}")
    print(f"  False Negatives:{total_fn}")
    print(f"  Precision:      {precision:.4f}")
    print(f"  Recall:         {recall:.4f}")
    print(f"  F1 Score:       {f1:.4f}")
    print(f"\nError Patterns:")
    print(f"  Small target FN: {error_stats['small_miss']}")
    print(f"  Edge FN:         {error_stats['edge_miss']}")

    # Save error report
    report = {
        'config': {'conf': conf, 'iou': iou, 'samples': len(sampled)},
        'metrics': {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
        },
        'counts': {
            'total_gt': total_gt,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
        },
        'error_patterns': {
            'small_target_fn': error_stats['small_miss'],
            'edge_fn': error_stats['edge_miss'],
        },
        'per_image': all_results[:50],
    }

    report_path = REPORT_DIR / "error_analysis.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_path}")
    print(f"Visualizations saved to: {ERROR_VIS_DIR}")

    return report


# ================================================================
# PHASE 2: Parameter Grid Search
# ================================================================
def phase2_param_search(sample_size=200):
    """Grid search over conf, iou, imgsz for optimal OCR F1."""
    print("=" * 60)
    print("PHASE 2: Inference Parameter Grid Search")
    print("=" * 60)

    if not BEST_PT.exists():
        print(f"ERROR: Model not found at {BEST_PT}")
        return

    # Parameter ranges
    conf_values = [0.15, 0.25, 0.35, 0.45, 0.55]
    iou_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    imgsz_values = [512, 640, 768, 960]

    # Use a subset of val images for speed
    val_images = sorted([f for f in os.listdir(VAL_IMG_DIR) if f.lower().endswith(('.png', '.jpg'))])
    if len(val_images) > sample_size:
        sampled = random.sample(val_images, sample_size)
    else:
        sampled = val_images

    # Load GT for all sampled images first (to avoid repeated IO)
    print(f"Loading GT for {len(sampled)} images...")
    gt_data = {}
    for img_name in tqdm(sampled, desc="Loading GT"):
        img_path = VAL_IMG_DIR / img_name
        base = os.path.splitext(img_name)[0]
        lbl_path = VAL_LBL_DIR / (base + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        gt_data[img_name] = {
            'boxes': load_gt_boxes(lbl_path, img_w, img_h),
            'img_w': img_w,
            'img_h': img_h,
        }

    total_gt = sum(len(v['boxes']) for v in gt_data.values())
    print(f"Total GT boxes: {total_gt}")

    # Search
    results_grid = []

    total_combos = len(conf_values) * len(iou_values) * len(imgsz_values)
    print(f"\nSearching {total_combos} parameter combinations...")

    for imgsz_val in imgsz_values:
        # Load model once per imgsz for efficiency
        model = YOLO(str(BEST_PT))

        for conf_val in conf_values:
            for iou_val in iou_values:
                print(f"  imgsz={imgsz_val}, conf={conf_val}, iou={iou_val}")

                tp_total = 0
                fp_total = 0
                fn_total = 0

                for img_name, data in tqdm(gt_data.items(), desc=f"  Infer", leave=False):
                    img_path = VAL_IMG_DIR / img_name
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue

                    gt_boxes = data['boxes']
                    img_w, img_h = data['img_w'], data['img_h']

                    # Run inference
                    results = model(img, conf=conf_val, iou=iou_val, imgsz=imgsz_val, verbose=False)
                    pred_boxes = []
                    if results and len(results) > 0:
                        r = results[0]
                        if r.boxes is not None and len(r.boxes) > 0:
                            for box in r.boxes:
                                xyxy = box.xyxy[0].cpu().numpy().tolist()
                                conf_val_pred = float(box.conf[0])
                                cls_id = int(box.cls[0])
                                pred_boxes.append({'cls': cls_id, 'bbox': xyxy, 'conf': conf_val_pred})

                    # Match
                    tp_preds, fp_preds, fn_boxes = match_predictions(pred_boxes, gt_boxes, iou_thresh=0.5)
                    tp_total += len(tp_preds)
                    fp_total += len(fp_preds)
                    fn_total += len(fn_boxes)

                # Compute metrics (OCR F1 oriented)
                precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
                recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
                f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                results_grid.append({
                    'imgsz': imgsz_val,
                    'conf': conf_val,
                    'iou': iou_val,
                    'precision': round(precision, 4),
                    'recall': round(recall, 4),
                    'f1': round(f1_score, 4),
                    'tp': tp_total,
                    'fp': fp_total,
                    'fn': fn_total,
                })

                print(f"    P={precision:.4f} R={recall:.4f} F1={f1_score:.4f} (FP={fp_total}, FN={fn_total})")

    # Sort by F1
    results_grid.sort(key=lambda x: x['f1'], reverse=True)

    # Save results
    results_path = REPORT_DIR / "param_search_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_grid, f, indent=2)

    # Save CSV
    csv_path = PARAM_SEARCH_DIR / "param_search.csv"
    with open(csv_path, 'w') as f:
        cols = ['imgsz', 'conf', 'iou', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn']
        f.write(','.join(cols) + '\n')
        for r in results_grid:
            f.write(','.join(str(r[c]) for c in cols) + '\n')

    # Print top 10
    print(f"\n{'='*60}")
    print(f"Top 10 Parameter Combinations by F1:")
    print(f"{'='*60}")
    print(f"{'Rank':<5} {'imgsz':<7} {'conf':<7} {'iou':<7} {'P':<8} {'R':<8} {'F1':<8} {'FP':<6} {'FN':<6}")
    print("-" * 62)
    for rank, r in enumerate(results_grid[:10]):
        print(f"{rank+1:<5} {r['imgsz']:<7} {r['conf']:<7} {r['iou']:<7} "
              f"{r['precision']:<8.4f} {r['recall']:<8.4f} {r['f1']:<8.4f} "
              f"{r['fp']:<6} {r['fn']:<6}")

    best = results_grid[0]
    print(f"\nBest config: imgsz={best['imgsz']}, conf={best['conf']}, iou={best['iou']}")
    print(f"  Precision={best['precision']:.4f}, Recall={best['recall']:.4f}, F1={best['f1']:.4f}")

    return results_grid


# ================================================================
# PHASE 3: TTA Testing
# ================================================================
def phase3_tta_test(sample_size=100, conf=0.25, iou=0.5, imgsz=640):
    """Test Test-Time Augmentation strategies."""
    print("=" * 60)
    print("PHASE 3: TTA Testing")
    print("=" * 60)

    if not BEST_PT.exists():
        print(f"ERROR: Model not found at {BEST_PT}")
        return

    model = YOLO(str(BEST_PT))

    val_images = sorted([f for f in os.listdir(VAL_IMG_DIR) if f.lower().endswith(('.png', '.jpg'))])
    if len(val_images) > sample_size:
        sampled = random.sample(val_images, sample_size)
    else:
        sampled = val_images

    # Load GT
    gt_data = {}
    for img_name in tqdm(sampled, desc="Loading GT"):
        img_path = VAL_IMG_DIR / img_name
        base = os.path.splitext(img_name)[0]
        lbl_path = VAL_LBL_DIR / (base + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        gt_data[img_name] = {
            'boxes': load_gt_boxes(lbl_path, img_w, img_h),
            'img_w': img_w, 'img_h': img_h,
        }

    # Test strategies
    strategies = {
        'baseline': {'augment': False},
        'flip_lr': {'augment': False, 'flip_lr': True},  # manual
        'flip_ud': {'augment': False, 'flip_ud': True},
        'augment_flag': {'augment': True},  # Ultralytics built-in augment
    }

    results = {}

    for strategy_name, strategy_cfg in strategies.items():
        print(f"\nTesting: {strategy_name}")
        tp_total = 0
        fp_total = 0
        fn_total = 0

        for img_name, data in tqdm(gt_data.items(), desc=f"  {strategy_name}", leave=False):
            img_path = VAL_IMG_DIR / img_name
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            gt_boxes = data['boxes']
            img_w, img_h = data['img_w'], data['img_h']

            # Manual flip TTA
            if strategy_name == 'flip_lr':
                flipped = cv2.flip(img, 1)
                r1 = model(img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
                r2 = model(flipped, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
                # Combine: merge predictions from both
                all_preds = []
                for r in [r1, r2]:
                    if r and len(r) > 0:
                        rr = r[0]
                        if rr.boxes is not None and len(rr.boxes) > 0:
                            for box in rr.boxes:
                                xyxy = box.xyxy[0].cpu().numpy().tolist()
                                if r is r2:  # flip back
                                    xyxy[0], xyxy[2] = img_w - xyxy[2], img_w - xyxy[0]
                                all_preds.append({'cls': int(box.cls[0]), 'bbox': xyxy, 'conf': float(box.conf[0])})
                # NMS on combined predictions
                all_preds = nms_predictions(all_preds, iou)
            elif strategy_name == 'flip_ud':
                flipped = cv2.flip(img, 0)
                r1 = model(img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
                r2 = model(flipped, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
                all_preds = []
                for r in [r1, r2]:
                    if r and len(r) > 0:
                        rr = r[0]
                        if rr.boxes is not None and len(rr.boxes) > 0:
                            for box in rr.boxes:
                                xyxy = box.xyxy[0].cpu().numpy().tolist()
                                if r is r2:
                                    xyxy[1], xyxy[3] = img_h - xyxy[3], img_h - xyxy[1]
                                all_preds.append({'cls': int(box.cls[0]), 'bbox': xyxy, 'conf': float(box.conf[0])})
                all_preds = nms_predictions(all_preds, iou)
            elif strategy_name == 'augment_flag':
                results_yolo = model(img, conf=conf, iou=iou, imgsz=imgsz, augment=True, verbose=False)
                all_preds = []
                if results_yolo and len(results_yolo) > 0:
                    r = results_yolo[0]
                    if r.boxes is not None and len(r.boxes) > 0:
                        for box in r.boxes:
                            all_preds.append({'cls': int(box.cls[0]),
                                              'bbox': box.xyxy[0].cpu().numpy().tolist(),
                                              'conf': float(box.conf[0])})
            else:  # baseline
                results_yolo = model(img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
                all_preds = []
                if results_yolo and len(results_yolo) > 0:
                    r = results_yolo[0]
                    if r.boxes is not None and len(r.boxes) > 0:
                        for box in r.boxes:
                            all_preds.append({'cls': int(box.cls[0]),
                                              'bbox': box.xyxy[0].cpu().numpy().tolist(),
                                              'conf': float(box.conf[0])})

            # Match
            tp_preds, fp_preds, fn_boxes = match_predictions(all_preds, gt_boxes, iou_thresh=0.5)
            tp_total += len(tp_preds)
            fp_total += len(fp_preds)
            fn_total += len(fn_boxes)

        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[strategy_name] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1_score, 4),
            'tp': tp_total,
            'fp': fp_total,
            'fn': fn_total,
        }
        print(f"  P={precision:.4f} R={recall:.4f} F1={f1_score:.4f} (FP={fp_total}, FN={fn_total})")

    # Print comparison
    print(f"\n{'='*60}")
    print("TTA Comparison:")
    print(f"{'='*60}")
    print(f"{'Strategy':<15} {'P':<10} {'R':<10} {'F1':<10} {'FP':<8} {'FN':<8}")
    print("-" * 55)
    for strategy, metrics in results.items():
        print(f"{strategy:<15} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
              f"{metrics['f1']:<10.4f} {metrics['fp']:<8} {metrics['fn']:<8}")

    # Save
    tta_path = REPORT_DIR / "tta_results.json"
    with open(tta_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def nms_predictions(predictions, iou_threshold):
    """Apply NMS to a list of prediction dicts."""
    if len(predictions) == 0:
        return []

    # Sort by confidence
    predictions = sorted(predictions, key=lambda x: x['conf'], reverse=True)
    keep = []

    while len(predictions) > 0:
        best = predictions.pop(0)
        keep.append(best)
        filtered = []
        for pred in predictions:
            if compute_iou(best['bbox'], pred['bbox']) < iou_threshold:
                filtered.append(pred)
        predictions = filtered

    return keep


# ================================================================
# PHASE 4: Hard Negative Analysis
# ================================================================
def phase4_hard_negative_analysis(sample_size=100, conf=0.25, iou=0.5):
    """Analyze false positive patterns for hard negative mining."""
    print("=" * 60)
    print("PHASE 4: Hard Negative Analysis")
    print("=" * 60)

    if not BEST_PT.exists():
        print(f"ERROR: Model not found at {BEST_PT}")
        return

    model = YOLO(str(BEST_PT))

    val_images = sorted([f for f in os.listdir(VAL_IMG_DIR) if f.lower().endswith(('.png', '.jpg'))])
    if len(val_images) > sample_size:
        sampled = random.sample(val_images, sample_size)
    else:
        sampled = val_images

    fp_crops = []
    fp_confidences = []
    fp_sizes = []
    fp_positions = []  # relative position in image
    tp_sizes = []

    for img_name in tqdm(sampled, desc="Collecting hard negatives"):
        img_path = VAL_IMG_DIR / img_name
        base = os.path.splitext(img_name)[0]
        lbl_path = VAL_LBL_DIR / (base + ".txt")

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        img_area = img_w * img_h

        gt_boxes = load_gt_boxes(lbl_path, img_w, img_h)

        # Run inference
        results = model(img, conf=conf, iou=iou, verbose=False)
        pred_boxes = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    pred_boxes.append({
                        'cls': int(box.cls[0]),
                        'bbox': box.xyxy[0].cpu().numpy().tolist(),
                        'conf': float(box.conf[0]),
                    })

        # Match
        tp_preds, fp_preds, fn_boxes = match_predictions(pred_boxes, gt_boxes, iou_thresh=0.5)

        # Collect FP info
        for _, fp in fp_preds:
            bbox = fp['bbox']
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            fp_confidences.append(fp['conf'])
            fp_sizes.append((w * h) / img_area * 100)  # percentage
            fp_positions.append({
                'cx': (bbox[0] + bbox[2]) / 2 / img_w,
                'cy': (bbox[1] + bbox[3]) / 2 / img_h,
            })

            # Extract FP crop (for visual analysis)
            x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
            x2 = min(img_w-1, x2)
            y2 = min(img_h-1, y2)
            if x2 > x1 and y2 > y1:
                crop = img[y1:y2, x1:x2]
                fp_crops.append({
                    'crop': crop,
                    'conf': fp['conf'],
                    'size_ratio': (w * h) / img_area * 100,
                    'aspect_ratio': w / h if h > 0 else 0,
                })

        # Collect TP sizes for comparison
        for _, tp, _, _ in tp_preds:
            bbox = tp['bbox']
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            tp_sizes.append((w * h) / img_area * 100)

    # Analysis
    print(f"\nHard Negative Analysis Results:")
    print(f"  Total FP samples collected: {len(fp_crops)}")
    print(f"  Mean FP confidence: {np.mean(fp_confidences):.4f}" if fp_confidences else "  No FPs found")
    print(f"  Median FP confidence: {np.median(fp_confidences):.4f}" if fp_confidences else "")
    print(f"  Mean FP size (% of image): {np.mean(fp_sizes):.2f}%" if fp_sizes else "")
    print(f"  Mean TP size (% of image): {np.mean(tp_sizes):.2f}%" if tp_sizes else "")

    # Save top FP crops for visual inspection
    if fp_crops:
        # Sort by confidence (highest first)
        fp_crops.sort(key=lambda x: x['conf'], reverse=True)
        for i, fp in enumerate(fp_crops[:30]):
            crop_img = fp['crop']
            if crop_img.size > 0:
                out_path = HN_DIR / f"fp_crop_{i:03d}_conf{fp['conf']:.2f}_size{fp['size_ratio']:.1f}.jpg"
                cv2.imwrite(str(out_path), crop_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Save analysis
    analysis = {
        'total_fp': len(fp_crops),
        'mean_conf': float(np.mean(fp_confidences)) if fp_confidences else 0,
        'median_conf': float(np.median(fp_confidences)) if fp_confidences else 0,
        'mean_fp_size_pct': float(np.mean(fp_sizes)) if fp_sizes else 0,
        'mean_tp_size_pct': float(np.mean(tp_sizes)) if tp_sizes else 0,
        'conf_distribution': [float(c) for c in fp_confidences],
        'size_distribution': [float(s) for s in fp_sizes],
    }

    analysis_path = REPORT_DIR / "hard_negative_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\nHard negative analysis saved to: {analysis_path}")
    print(f"FP crops saved to: {HN_DIR}")

    return analysis


# ================================================================
# PHASE 5: Convergence Analysis
# ================================================================
def phase5_convergence_analysis():
    """Analyze training curves to determine if more training would help."""
    print("=" * 60)
    print("PHASE 5: Convergence Analysis")
    print("=" * 60)

    if not RESULTS_CSV.exists():
        print(f"ERROR: results.csv not found at {RESULTS_CSV}")
        return

    import csv
    epochs = []
    box_loss = []
    cls_loss = []
    val_box_loss = []
    val_cls_loss = []
    mAP50 = []
    mAP50_95 = []
    precision = []
    recall = []

    with open(RESULTS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            box_loss.append(float(row['train/box_loss']))
            cls_loss.append(float(row['train/cls_loss']))
            val_box_loss.append(float(row['val/box_loss']))
            val_cls_loss.append(float(row['val/cls_loss']))
            mAP50.append(float(row['metrics/mAP50(B)']))
            mAP50_95.append(float(row['metrics/mAP50-95(B)']))
            precision.append(float(row['metrics/precision(B)']))
            recall.append(float(row['metrics/recall(B)']))

    # Analysis
    mAP50_trend = np.polyfit(range(len(mAP50)), mAP50, 1)[0]
    mAP50_95_trend = np.polyfit(range(len(mAP50_95)), mAP50_95, 1)[0]
    box_loss_trend = np.polyfit(range(len(box_loss)), box_loss, 1)[0]

    # Check if converging
    recent_mAP50_improvement = mAP50[-1] - mAP50[-3] if len(mAP50) >= 3 else 0
    recent_loss_reduction = box_loss[-3] - box_loss[-1] if len(box_loss) >= 3 else 0

    # Overfitting check: val loss trend vs train loss
    val_loss_trend = np.polyfit(range(len(val_box_loss)), val_box_loss, 1)[0]
    overfitting_signal = val_loss_trend > 0  # val loss increasing = overfitting

    analysis = {
        'epochs_completed': len(epochs),
        'final_mAP50': mAP50[-1],
        'final_mAP50_95': mAP50_95[-1],
        'final_precision': precision[-1],
        'final_recall': recall[-1],
        'mAP50_trend_per_epoch': round(mAP50_trend, 5),
        'mAP50_95_trend_per_epoch': round(mAP50_95_trend, 5),
        'box_loss_trend_per_epoch': round(box_loss_trend, 4),
        'recent_mAP50_improvement': round(recent_mAP50_improvement, 4),
        'recent_loss_reduction': round(recent_loss_reduction, 4),
        'overfitting_detected': bool(overfitting_signal),
        'is_converged': bool(abs(recent_mAP50_improvement) < 0.01 and abs(recent_loss_reduction) < 0.05),
    }

    # Recommendation
    if overfitting_signal:
        recommendation = "STOP - overfitting detected (val loss increasing)"
    elif analysis['is_converged']:
        recommendation = "STOP - model has converged"
    elif mAP50_trend > 0.01 and box_loss_trend < -0.02:
        recommendation = "CONTINUE - model still improving"
        extra_epochs = min(10, int((0.01 / mAP50_trend) if mAP50_trend > 0 else 10))
        analysis['recommended_extra_epochs'] = max(3, extra_epochs)
        analysis['recommended_lr'] = 'current_lr * 0.5 (cosine schedule)'
        analysis['recommended_imgsz'] = 'increase to 640 or 768'
        analysis['recommended_augment'] = 'enable mild mosaic=0.3, mixup=0.1'
    else:
        recommendation = "MARGINAL - can try 5-10 more epochs with lower LR"

    analysis['recommendation'] = recommendation

    print(f"\nConvergence Analysis:")
    print(f"  Epochs completed:    {len(epochs)}")
    print(f"  Final mAP50:         {mAP50[-1]:.4f}")
    print(f"  Final mAP50-95:      {mAP50_95[-1]:.4f}")
    print(f"  mAP50 trend/epoch:   {mAP50_trend:.5f}")
    print(f"  Box loss trend/epoch:{box_loss_trend:.4f}")
    print(f"  Recent mAP50 gain:   {recent_mAP50_improvement:.4f}")
    print(f"  Overfitting:         {overfitting_signal}")
    print(f"\n  Recommendation: {recommendation}")

    # Generate plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, box_loss, 'b-o', label='Train Box Loss', markersize=4)
    ax.plot(epochs, val_box_loss, 'r-o', label='Val Box Loss', markersize=4)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Box Loss'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, cls_loss, 'b-o', label='Train CLS Loss', markersize=4)
    ax.plot(epochs, val_cls_loss, 'r-o', label='Val CLS Loss', markersize=4)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Classification Loss'); ax.legend(); ax.grid(True, alpha=0.3)

    # mAP curves
    ax = axes[0, 2]
    ax.plot(epochs, mAP50, 'g-o', label='mAP50', markersize=6)
    ax.plot(epochs, mAP50_95, 'y-o', label='mAP50-95', markersize=6)
    ax.set_xlabel('Epoch'); ax.set_ylabel('mAP')
    ax.set_title('Detection mAP'); ax.legend(); ax.grid(True, alpha=0.3)

    # P/R curve trend
    ax = axes[1, 0]
    ax.plot(epochs, precision, 'c-o', label='Precision', markersize=4)
    ax.plot(epochs, recall, 'm-o', label='Recall', markersize=4)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Score')
    ax.set_title('Precision & Recall'); ax.legend(); ax.grid(True, alpha=0.3)

    # F1 evolution
    f1_values = [2 * p * r / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision, recall)]
    ax = axes[1, 1]
    ax.plot(epochs, f1_values, 'r-o', label='OCR F1', markersize=6)
    ax.set_xlabel('Epoch'); ax.set_ylabel('F1 Score')
    ax.set_title('OCR F1 Score'); ax.legend(); ax.grid(True, alpha=0.3)

    # Combined view
    ax = axes[1, 2]
    ax2 = ax.twinx()
    ax.plot(epochs, box_loss, 'b-o', label='Box Loss', markersize=4)
    ax2.plot(epochs, mAP50, 'g-s', label='mAP50', markersize=6)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss', color='b')
    ax2.set_ylabel('mAP50', color='g')
    ax.set_title('Combined: Loss & mAP50')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Training Convergence Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = REPORT_DIR / "convergence_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nConvergence plots saved to: {plot_path}")

    analysis_path = REPORT_DIR / "convergence_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    return analysis


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="all",
                        choices=["1", "2", "3", "4", "5", "all"])
    parser.add_argument("--samples", type=int, default=200,
                        help="Number of val samples for param search")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for error analysis")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="NMS IoU for error analysis")
    args = parser.parse_args()

    all_results = {}

    if args.phase in ("1", "all"):
        all_results['error_analysis'] = phase1_error_analysis(conf=args.conf, iou=args.iou)

    if args.phase in ("2", "all"):
        all_results['param_search'] = phase2_param_search(sample_size=args.samples)

    if args.phase in ("3", "all"):
        all_results['tta'] = phase3_tta_test(sample_size=min(args.samples, 100))

    if args.phase in ("4", "all"):
        all_results['hard_negative'] = phase4_hard_negative_analysis(sample_size=min(args.samples, 100))

    if args.phase in ("5", "all"):
        all_results['convergence'] = phase5_convergence_analysis()

    print(f"\n{'='*60}")
    print("All optimization phases complete!")
    print(f"Results saved to: {REPORT_DIR}")
    print(f"{'='*60}")
