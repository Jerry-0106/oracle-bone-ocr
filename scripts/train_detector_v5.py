#!/usr/bin/env python3
"""
V5 Detector Training: Domain-Augmented for Cross-Source Generalization.

Key changes from v4:
  - imgsz=640 (not 1280) — matches current deployment
  - YOLOv8n (not v8m) — matches current detector.pt
  - Domain-augmented dataset (not raw)
  - Reduced mosaic/copy_paste — focus on appearance, not geometry
  - 30 epochs (not 50) — short, focused training

Usage:
  python scripts/train_detector_v5.py
  python scripts/train_detector_v5.py --resume
"""
import os, sys, time, traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs"
LOGS_DIR = PROJECT_ROOT / "logs"

# ── config ──
DATA_YAML = PROJECT_ROOT / "data" / "data.yaml"
RUN_NAME  = "detector_v5"
EPOCHS    = 30
PATIENCE  = 10
IMGSZ     = 640
BATCH     = 8
DEVICE    = "cuda"  # override to "cpu" if no GPU


def main():
    print("=" * 60)
    print("V5 DETECTOR TRAINING — Domain-Augmented")
    print(f"Model: YOLOv8n | imgsz: {IMGSZ} | Epochs: {EPOCHS}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Device: {DEVICE} | Batch: {BATCH}")
    print("=" * 60)

    if not DATA_YAML.exists():
        print(f"\nERROR: {DATA_YAML} not found!")
        print("Check dataset_yolo/data.yaml exists")
        sys.exit(1)

    from ultralytics import YOLO

    # Count images
    import glob
    n_train = len(glob.glob(str(PROJECT_ROOT / "data" / "yolo" / "images" / "train" / "*")))
    n_val   = len(glob.glob(str(PROJECT_ROOT / "data" / "yolo" / "images" / "val" / "*")))
    print(f"Train images: {n_train}  Val images: {n_val}")

    # Auto-resume if checkpoints exist
    ckpt_path = RUNS_DIR / RUN_NAME / "weights" / "last.pt"
    resume_training = ckpt_path.exists()
    if resume_training:
        print(f"Resuming from {ckpt_path}")
        model = YOLO(str(ckpt_path))
    else:
        model = YOLO("yolov8n.pt")

    try:
        results = model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            patience=PATIENCE,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            workers=2,
            cache=False,
            project=str(RUNS_DIR),
            name=RUN_NAME,
            exist_ok=True,
            verbose=True,
            resume=resume_training,
            amp=True if DEVICE == "cuda" else False,
            cos_lr=True,
            close_mosaic=5,
            # No flip
            hsv_h=0.02, hsv_s=0.7, hsv_v=0.5,
            degrees=2.0, translate=0.1, scale=0.3, shear=0.0,
            flipud=0.0, fliplr=0.0,
            # Reduced mosaic, no copy-paste
            mosaic=0.3, copy_paste=0.0, mixup=0.0,
            # Optimizer
            lr0=0.01, lrf=0.01,
        )
    except RuntimeError as e:
        print(f"\nRuntimeError: {e}")
        traceback.print_exc()
        print(f"\nCheckpoints saved in: {RUNS_DIR / RUN_NAME / 'weights'}/")
        results = None

    if results:
        save_dir = Path(results.save_dir)
        best_pt = save_dir / "weights" / "best.pt"
        if best_pt.exists():
            print(f"\nBest model: {best_pt}")
            # Copy to project models
            dst = PROJECT_ROOT / "checkpoints" / "detector_v5.pt"
            import shutil
            shutil.copy2(best_pt, dst)
            print(f"Copied to: {dst}")

    print("\nV5 training complete.")


if __name__ == '__main__':
    main()
