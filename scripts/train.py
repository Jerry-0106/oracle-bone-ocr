#!/usr/bin/env python3
"""
YOLOv8 Training Script for Ancient Character OCR.

Usage:
  python train.py --mode mini          # Tiny validation
  python train.py --mode fast          # Fast baseline (new training)
  python train.py --mode fast --resume # Resume from last checkpoint
  python train.py --mode full          # Full training
"""

import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_ROOT / "logs"
RUNS_DIR = PROJECT_ROOT / "runs"

if os.environ.get("HTTP_PROXY") is None:
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

os.environ["OMP_NUM_THREADS"] = "1"


def train_mini():
    """Tiny validation with yolov8n on mini dataset."""
    print("=" * 60)
    print("MINI TRAINING - Pipeline Smoke Test")
    print("=" * 60)

    data_yaml = str(PROJECT_ROOT / "data" / "yolo" / "data.yaml")
    if not Path(data_yaml).exists():
        print("ERROR: data.yaml not found")
        sys.exit(1)

    print("Model: yolov8n.pt | Epochs: 3 | imgsz: 640 | Batch: 2 | Workers: 0")

    model = YOLO("yolov8n.pt")
    return model.train(
        data=data_yaml, epochs=3, imgsz=640, batch=2,
        device="mps", workers=0, cache=False,
        project=str(RUNS_DIR), name="mini_validate",
        exist_ok=True, verbose=True, amp=False,
        close_mosaic=0, cos_lr=False, patience=100,
    )


def train_fast(resume=False):
    """Fast baseline: full data, yolov8n, max 10 epochs, early stopping."""
    print("=" * 60)
    print("FAST BASELINE TRAINING" + (" (RESUME)" if resume else ""))
    print("=" * 60)

    data_yaml = str(PROJECT_ROOT / "data" / "yolo" / "data.yaml")
    if not Path(data_yaml).exists():
        print("ERROR: data.yaml not found")
        sys.exit(1)

    run_dir = RUNS_DIR / "fast_baseline"

    if resume:
        last_pt = run_dir / "weights" / "last.pt"
        if not last_pt.exists():
            print(f"ERROR: no checkpoint at {last_pt}")
            sys.exit(1)
        print(f"Resuming from: {last_pt}")
        model = YOLO(str(last_pt))
    else:
        print("Model: yolov8n.pt | Epochs: 10 (patience=3) | imgsz: 416 | Batch: 8 | Workers: 0")
        model = YOLO("yolov8n.pt")

    print("Epochs: 10 (patience=3) | imgsz: 416 | Batch: 8 | Workers: 0")

    try:
        results = model.train(
            data=data_yaml,
            epochs=10,
            patience=3,
            imgsz=416,
            batch=8,
            device="mps",
            workers=0,
            cache=False,
            project=str(RUNS_DIR),
            name="fast_baseline",
            exist_ok=True,
            verbose=True,
            resume=resume,
            amp=False,
            cos_lr=True,
            close_mosaic=0,
            hsv_h=0.015, hsv_s=0.5, hsv_v=0.3,
            degrees=3.0, translate=0.1, scale=0.3,
            flipud=0.0, fliplr=0.0,
            mosaic=0.0, mixup=0.0,
        )
    except RuntimeError as e:
        print(f"\n{'='*60}")
        print(f"MPS RuntimeError caught (known ultralytics MPS bug):")
        print(f"  {e}")
        print(f"Checkpoints are saved in: {run_dir}/weights/")
        print(f"  last.pt  - resume from here with: python scripts/train.py --mode fast --resume")
        print(f"  best.pt  - best model so far")
        print(f"{'='*60}")
        traceback.print_exc()
        return None

    return results


def train_full(resume=False):
    """Full training with yolov8m."""
    print("=" * 60)
    print("FULL TRAINING" + (" (RESUME)" if resume else ""))
    print("=" * 60)

    data_yaml = str(PROJECT_ROOT / "data" / "yolo" / "data.yaml")
    if not Path(data_yaml).exists():
        print("ERROR: data.yaml not found")
        sys.exit(1)

    run_dir = RUNS_DIR / "full_train"

    if resume:
        last_pt = run_dir / "weights" / "last.pt"
        if not last_pt.exists():
            print(f"ERROR: no checkpoint at {last_pt}")
            sys.exit(1)
        print(f"Resuming from: {last_pt}")
        model = YOLO(str(last_pt))
    else:
        print("Model: yolov8m.pt | Epochs: 20 (patience=10) | imgsz: 640 | Batch: 4 | Workers: 0")
        model = YOLO("yolov8m.pt")

    print("Epochs: 20 (patience=10) | imgsz: 640 | Batch: 4 | Workers: 0")

    try:
        results = model.train(
            data=data_yaml,
            epochs=20,
            patience=10,
            imgsz=640,
            batch=4,
            device="mps",
            workers=0,
            cache=False,
            project=str(RUNS_DIR),
            name="full_train",
            exist_ok=True,
            verbose=True,
            resume=resume,
            amp=False,
            cos_lr=True,
            close_mosaic=5,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            degrees=5.0, translate=0.1, scale=0.3, shear=2.0,
            flipud=0.0, fliplr=0.0,
            mosaic=0.3, mixup=0.1,
        )
    except RuntimeError as e:
        print(f"\n{'='*60}")
        print(f"MPS RuntimeError caught:")
        print(f"  {e}")
        print(f"Checkpoints saved in: {run_dir}/weights/")
        print(f"  last.pt  - resume: python scripts/train.py --mode full --resume")
        print(f"  best.pt  - best model so far")
        print(f"{'='*60}")
        traceback.print_exc()
        return None

    return results


def print_results(results, mode_name):
    """Print training summary."""
    print(f"\n{'='*60}")
    print(f"Training: {mode_name}")
    print(f"{'='*60}")

    if results and hasattr(results, 'save_dir'):
        save_dir = Path(results.save_dir)
        best_pt = save_dir / "weights" / "best.pt"
        if best_pt.exists():
            print(f"Best model: {best_pt}")

        results_csv = save_dir / "results.csv"
        if results_csv.exists():
            import pandas as pd
            df = pd.read_csv(results_csv)
            print(f"\nEpochs completed: {len(df)}")
            print(f"\n{'Epoch':<8} {'mAP50':<10} {'mAP50-95':<10} {'P':<10} {'R':<10} {'box_loss':<10} {'val_loss':<10}")
            print("-" * 68)
            for _, row in df.iterrows():
                ep = int(row['epoch'])
                print(f"{ep:<8} {row['metrics/mAP50(B)']:<10.4f} {row['metrics/mAP50-95(B)']:<10.4f} "
                      f"{row['metrics/precision(B)']:<10.4f} {row['metrics/recall(B)']:<10.4f} "
                      f"{row['train/box_loss']:<10.4f} {row['val/box_loss']:<10.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mini", "fast", "full"], default="fast")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    if args.mode == "mini":
        results = train_mini()
    elif args.mode == "fast":
        results = train_fast(resume=args.resume)
    else:
        results = train_full(resume=args.resume)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")

    if results:
        print_results(results, args.mode)
    else:
        print("\nTraining interrupted but checkpoints saved.")
