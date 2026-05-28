#!/usr/bin/env python3
"""
Stage 1: Detector Parameter Sweep (detection-only F1, no recognition).
Strategy: Run YOLO once with permissive settings, then post-filter
for each conf/iou/max_det combination. This avoids re-running the
model 20-60 times on 1193 images.
"""
import json, sys, time, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

VAL_IMG_DIR = PROJECT_ROOT / "dataset_yolo" / "images" / "val"
VAL_LBL_DIR = PROJECT_ROOT / "dataset_yolo" / "labels" / "val"
CACHE_FILE  = PROJECT_ROOT / "evaluation" / "raw_detections.json"

# ── detection-only evaluation from competition evaluate.py ──

def parse_yolo_label(label_path, img_w, img_h):
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
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(0, (a[2]-a[0]) * (a[3]-a[1]))
    area_b = max(0, (b[2]-b[0]) * (b[3]-b[1]))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def greedy_nms(boxes, scores, iou_thresh):
    """Greedy NMS: returns indices of kept boxes."""
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=np.float64)
    scores = np.array(scores, dtype=np.float64)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        ious = np.array([compute_iou(boxes[i], boxes[j]) for j in order[1:]])
        inds = np.where(ious < iou_thresh)[0]
        order = order[inds + 1]
    return keep


def load_image_dims(img_dir):
    """Pre-load image dimensions using PIL (avoids cv2 libpng spam)."""
    from PIL import Image
    dims = {}
    for img_path in sorted(Path(img_dir).glob("*")):
        if img_path.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
            continue
        try:
            with Image.open(img_path) as im:
                dims[img_path.stem] = {"w": im.width, "h": im.height}
        except Exception:
            continue
    return dims


def evaluate_detections(all_preds, gt_dir, img_dims, iou_thresh=0.5):
    """Evaluate detection-only (no text matching). Returns metrics dict."""
    total_tp = total_fp = total_fn = 0
    per_image = []

    for image_id, pred_boxes in all_preds.items():
        gt_path = Path(gt_dir) / f"{image_id}.txt"

        if not gt_path.exists() or image_id not in img_dims:
            continue

        img_w = img_dims[image_id]["w"]
        img_h = img_dims[image_id]["h"]

        gt_boxes_raw = parse_yolo_label(gt_path, img_w, img_h)

        # pred_boxes are [x1,y1,x2,y2]
        pairs = []
        for pi, pb in enumerate(pred_boxes):
            for gi, gb in enumerate(gt_boxes_raw):
                iou_val = compute_iou(pb, gb)
                if iou_val >= iou_thresh:
                    pairs.append((iou_val, pi, gi))

        pairs.sort(key=lambda x: x[0], reverse=True)
        matched_pred = set()
        matched_gt = set()
        tp = 0

        for iou_val, pi, gi in pairs:
            if pi not in matched_pred and gi not in matched_gt:
                matched_pred.add(pi)
                matched_gt.add(gi)
                tp += 1

        fp = len(pred_boxes) - len(matched_pred)
        fn = len(gt_boxes_raw) - len(matched_gt)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if tp + fp + fn > 0:
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_img = 2*p*r/(p+r) if (p+r) > 0 else 0
            per_image.append({"image": image_id, "gt": len(gt_boxes_raw),
                              "pred": len(pred_boxes), "tp": tp, "fp": fp,
                              "fn": fn, "f1": round(f1_img, 4)})

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    f1_dist = defaultdict(int)
    for r in per_image:
        if r["f1"] == 1.0:    f1_dist["F1=1.0"] += 1
        elif r["f1"] >= 0.5:  f1_dist["F1>=0.5"] += 1
        elif r["f1"] > 0:     f1_dist["F1>0"] += 1
        else:                 f1_dist["F1=0"] += 1

    return {
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "f1_dist": dict(f1_dist),
        "n_images": len(per_image), "per_image": per_image,
    }


# ── main sweep ──

def run_raw_detection(detector_path, img_dir, imgsz, device):
    """Run YOLO with permissive settings, cache raw boxes."""
    from ultralytics import YOLO
    import cv2

    model = YOLO(str(detector_path))
    all_raw = {}

    images = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    if not images:
        print("[ERROR] No images found")
        return all_raw

    print(f"Running raw detection on {len(images)} images (conf=0.01, iou=0.3, augment=True)...")
    t0 = time.time()

    for idx, img_path in enumerate(images):
        image_id = img_path.stem
        results = model(
            str(img_path), conf=0.01, iou=0.3, imgsz=imgsz,
            augment=True, max_det=2000, device=device, verbose=False,
        )

        boxes = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
                conf = float(box.conf[0])
                boxes.append([float(x1), float(y1), float(x2), float(y2), float(conf)])

        all_raw[image_id] = boxes

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{len(images)}] {elapsed:.0f}s, last={image_id}")

    elapsed = time.time() - t0
    total_boxes = sum(len(v) for v in all_raw.values())
    print(f"  Done: {len(all_raw)} images, {total_boxes} raw boxes, {elapsed:.0f}s")
    return all_raw


def sweep(raw_detections, confs, ious, max_dets, gt_dir, img_dims):
    """Run full grid search over conf×iou×max_det."""
    results = []
    total_combos = len(confs) * len(ious) * len(max_dets)
    combo_idx = 0

    for md in max_dets:
        for iou_nms in ious:
            for conf_thr in confs:
                combo_idx += 1
                label = f"conf={conf_thr:.2f}_iou={iou_nms:.2f}_maxdet={md}"
                print(f"\n[{combo_idx}/{total_combos}] {label}")

                # Build filtered predictions
                all_preds = {}
                for image_id, raw_boxes in raw_detections.items():
                    # Filter by confidence
                    filtered = [(b[:4], b[4]) for b in raw_boxes if b[4] >= conf_thr]
                    if not filtered:
                        all_preds[image_id] = []
                        continue
                    boxes_arr = [f[0] for f in filtered]
                    scores_arr = [f[1] for f in filtered]
                    keep = greedy_nms(boxes_arr, scores_arr, iou_nms)
                    # Sort kept by confidence desc, limit to max_det
                    kept_boxes = [(boxes_arr[i], scores_arr[i]) for i in keep]
                    kept_boxes.sort(key=lambda x: x[1], reverse=True)
                    kept_boxes = kept_boxes[:md]
                    all_preds[image_id] = [b[0] for b in kept_boxes]

                metrics = evaluate_detections(all_preds, gt_dir, img_dims)
                metrics["conf"] = conf_thr
                metrics["iou_nms"] = iou_nms
                metrics["max_det"] = md
                metrics["label"] = label
                results.append(metrics)

                print(f"  TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} "
                      f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f} "
                      f"F1_dist={metrics['f1_dist']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Detector parameter sweep")
    parser.add_argument("--detector", default=None,
                        help="Path to detector .pt (default: competition_final/weights/best.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for YOLO")
    parser.add_argument("--device", default=None, help="Device (auto-detect if not set)")
    parser.add_argument("--cache-only", action="store_true",
                        help="Only run detection + save cache, skip sweep")
    parser.add_argument("--sweep-only", action="store_true",
                        help="Skip detection, load cache and sweep only")
    parser.add_argument("--output", default=None, help="Save sweep results JSON")
    parser.add_argument("--confs", default=None, help="Comma-separated conf values (default: 0.05,0.08,0.10,0.12,0.15)")
    parser.add_argument("--ious", default=None, help="Comma-separated iou values (default: 0.4,0.5,0.6,0.7)")
    parser.add_argument("--max-dets", default=None, help="Comma-separated max_det values (default: 300,500,1000)")
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    detector_path = args.detector or str(PROJECT_ROOT / "checkpoints" / "detector_v5.pt")
    if not Path(detector_path).exists():
        print(f"[ERROR] Detector not found: {detector_path}")
        sys.exit(1)

    # ── Step 1: raw detection (or load cache) ──
    if args.sweep_only and CACHE_FILE.exists():
        print(f"Loading cached raw detections from {CACHE_FILE}")
        with open(CACHE_FILE, encoding='utf-8') as f:
            raw_detections = json.load(f)
    else:
        raw_detections = run_raw_detection(detector_path, VAL_IMG_DIR, args.imgsz, device)
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving raw detections to {CACHE_FILE}")
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(raw_detections, f, ensure_ascii=False)

    if args.cache_only:
        print("Cache saved. Exiting (--cache-only).")
        return

    # ── Step 2: sweep ──
    confs  = [float(x) for x in args.confs.split(",")] if args.confs else [0.05, 0.08, 0.10, 0.12, 0.15]
    ious   = [float(x) for x in args.ious.split(",")] if args.ious else [0.4, 0.5, 0.6, 0.7]
    max_dets = [int(x) for x in args.max_dets.split(",")] if args.max_dets else [300, 500, 1000]

    print(f"\n{'='*60}")
    print(f"  Grid Search: {len(confs)} conf × {len(ious)} iou × {len(max_dets)} max_det")
    print(f"  = {len(confs)*len(ious)*len(max_dets)} combinations")
    print(f"{'='*60}")

    # Pre-load image dimensions once (avoids 71K+ cv2.imread calls)
    print("Loading image dimensions...")
    img_dims = load_image_dims(VAL_IMG_DIR)
    print(f"Loaded dims for {len(img_dims)} images")

    results = sweep(raw_detections, confs, ious, max_dets,
                    gt_dir=VAL_LBL_DIR, img_dims=img_dims)

    # ── Sort and display ──
    results.sort(key=lambda x: x["f1"], reverse=True)

    print(f"\n{'='*80}")
    print(f"  TOP 10 CONFIGURATIONS (by F1)")
    print(f"{'='*80}")
    print(f"  {'Rank':<5} {'Config':<42} {'F1':>8} {'P':>8} {'R':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    print(f"  {'-'*80}")

    for rank, r in enumerate(results[:10]):
        label = r["label"]
        f1_pct = r["f1"]*100
        p_pct = r["precision"]*100
        r_pct = r["recall"]*100
        print(f"  {rank+1:<5} {label:<42} {f1_pct:>7.2f}% {p_pct:>7.2f}% {r_pct:>7.2f}% "
              f"{r['tp']:>6} {r['fp']:>6} {r['fn']:>6}")

    print(f"\n  Baseline (conf=0.10, iou=0.3): not in sweep range")

    # ── Save ──
    output_path = args.output or str(PROJECT_ROOT / "evaluation" / "sweep_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {output_path}")

    # ── Summary ──
    best = results[0]
    print(f"\nBest config: {best['label']}  =>  F1={best['f1']*100:.2f}%")

    # Delta from baseline (estimated at 56.42%)
    baseline_f1 = 0.5642
    delta = (best['f1'] - baseline_f1) / baseline_f1 * 100
    print(f"Improvement over baseline (56.42%): {delta:+.1f}%")


if __name__ == "__main__":
    main()
