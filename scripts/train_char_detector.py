#!/usr/bin/env python3
"""
Character-Level YOLOv8m Detector Training for Oracle Bone Characters.

Each bbox = one character. No more region detection + segmentation.
Input size 1280px with small-target optimizations.

Usage:
  python scripts/train_char_detector.py
  python scripts/train_char_detector.py --resume
"""

import os, sys, time, traceback
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_ROOT / "logs"
RUNS_DIR = PROJECT_ROOT / "runs"

if os.environ.get("HTTP_PROXY") is None:
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

os.environ["OMP_NUM_THREADS"] = "1"


def train_char_detector(resume=False):
    print("=" * 60)
    print("CHARACTER-LEVEL DETECTOR TRAINING (v4)")
    print("Model: YOLOv8m | imgsz: 1280 | Character-level labels")
    print("=" * 60)

    data_yaml = str(PROJECT_ROOT / "data" / "yolo" / "data.yaml")
    if not Path(data_yaml).exists():
        print("ERROR: data.yaml not found at", data_yaml)
        sys.exit(1)

    run_dir = RUNS_DIR / "char_detector"

    if resume:
        last_pt = run_dir / "weights" / "last.pt"
        if not last_pt.exists():
            print(f"ERROR: no checkpoint at {last_pt}")
            sys.exit(1)
        print(f"Resuming from: {last_pt}")
        model = YOLO(str(last_pt))
    else:
        model = YOLO("yolov8m.pt")

    print("Epochs: 50 (patience=15) | imgsz: 1280 | Batch: 2")
    print("Aug: mosaic=0.5 copy_paste=0.3 degrees=2 scale=0.3")
    print("Auto-anchor: enabled")

    try:
        results = model.train(
            data=data_yaml,
            epochs=50,
            patience=15,
            imgsz=1280,
            batch=2,
            device="mps",
            workers=0,
            cache=False,
            project=str(RUNS_DIR),
            name="char_detector",
            exist_ok=True,
            verbose=True,
            resume=resume,
            amp=False,
            cos_lr=True,
            close_mosaic=10,
            # No flip — oracle bone characters are orientation-sensitive
            hsv_h=0.015, hsv_s=0.5, hsv_v=0.3,
            degrees=2.0, translate=0.1, scale=0.3, shear=0.0,
            flipud=0.0, fliplr=0.0,
            # Small target augmentations
            mosaic=0.5, copy_paste=0.3, mixup=0.0,
            # Auto-anchor clustering (default True)
            auto_anchor=True,
        )
    except RuntimeError as e:
        print(f"\n{'='*60}")
        print(f"MPS RuntimeError caught (known ultralytics MPS bug):")
        print(f"  {e}")
        print(f"Checkpoints saved in: {run_dir}/weights/")
        print(f"  last.pt  - resume with: python scripts/train_char_detector.py --resume")
        print(f"  best.pt  - best model so far")
        print(f"{'='*60}")
        traceback.print_exc()
        return None

    return results


def print_results(results):
    print(f"\n{'='*60}")
    print("Character Detector Training Complete")
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
            print(f"\n{'Epoch':<8} {'mAP50':<10} {'mAP50-95':<10} {'P':<10} {'R':<10}")
            print("-" * 48)
            for _, row in df.iterrows():
                ep = int(row['epoch'])
                print(f"{ep:<8} {row['metrics/mAP50(B)']:<10.4f} "
                      f"{row['metrics/mAP50-95(B)']:<10.4f} "
                      f"{row['metrics/precision(B)']:<10.4f} "
                      f"{row['metrics/recall(B)']:<10.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train character-level YOLOv8m detector")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    results = train_char_detector(resume=args.resume)
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")

    if results:
        print_results(results)
    else:
        print("\nTraining interrupted but checkpoints saved.")
