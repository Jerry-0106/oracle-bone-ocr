#!/usr/bin/env python3
"""
Resume Training with Optimized Parameters.
Based on convergence analysis: model not converged, mAP50 still improving +0.024/epoch.

Usage:
  python resume_train.py
"""

import os
import sys
import time
import traceback
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs"
DATA_YAML = PROJECT_ROOT / "data" / "yolo" / "data.yaml"

LAST_PT = RUNS_DIR / "fast_baseline" / "weights" / "last.pt"

if os.environ.get("HTTP_PROXY") is None:
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

os.environ["OMP_NUM_THREADS"] = "1"


def main():
    if not LAST_PT.exists():
        print(f"ERROR: Checkpoint not found at {LAST_PT}")
        sys.exit(1)

    print("=" * 60)
    print("RESUME TRAINING - Competition Optimization")
    print("=" * 60)
    print(f"Resuming from: {LAST_PT}")
    print(f"Extra epochs: 15 (total 25)")
    print(f"imgsz: 640 (up from 416 for small target recall)")
    print(f"Batch: 4, Workers: 0, Device: mps")
    print(f"Patience: 8 (early stopping)")
    print(f"Mosaic: 0.3, Mixup: 0.1 (mild augment)")

    model = YOLO(str(LAST_PT))

    try:
        results = model.train(
            data=str(DATA_YAML),
            epochs=25,           # Total epochs (resumed from 10)
            patience=8,          # Early stopping
            imgsz=640,           # Larger for small targets
            batch=4,
            device="mps",
            workers=0,
            cache=False,
            project=str(RUNS_DIR),
            name="competition_final",
            exist_ok=True,
            verbose=True,
            resume=True,
            amp=False,
            cos_lr=True,
            close_mosaic=3,      # Disable mosaic for last epochs
            # Moderate augmentation
            hsv_h=0.015,
            hsv_s=0.5,
            hsv_v=0.3,
            degrees=5.0,
            translate=0.1,
            scale=0.3,
            shear=2.0,
            flipud=0.0,
            fliplr=0.0,          # No flip for ancient characters
            mosaic=0.3,
            mixup=0.1,
        )
    except RuntimeError as e:
        print(f"\nMPS RuntimeError caught:")
        print(f"  {e}")
        print(f"Checkpoints saved in: {RUNS_DIR}/competition_final/weights/")
        print(f"  last.pt  - resume: python resume_train.py")
        print(f"  best.pt  - best model so far")
        traceback.print_exc()
        return

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")

    if results and hasattr(results, 'save_dir'):
        best_pt = Path(results.save_dir) / "weights" / "best.pt"
        if best_pt.exists():
            print(f"Best model: {best_pt}")

        results_csv = Path(results.save_dir) / "results.csv"
        if results_csv.exists():
            import pandas as pd
            df = pd.read_csv(results_csv)
            print(f"\nFinal metrics:")
            last = df.iloc[-1]
            print(f"  mAP50:     {last['metrics/mAP50(B)']:.4f}")
            print(f"  mAP50-95:  {last['metrics/mAP50-95(B)']:.4f}")
            print(f"  Precision: {last['metrics/precision(B)']:.4f}")
            print(f"  Recall:    {last['metrics/recall(B)']:.4f}")
            print(f"  box_loss:  {last['train/box_loss']:.4f}")
            print(f"  val_loss:  {last['val/box_loss']:.4f}")


if __name__ == "__main__":
    main()
