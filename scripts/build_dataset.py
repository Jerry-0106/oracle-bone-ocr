#!/usr/bin/env python3
"""
Build YOLO dataset structure from converted labels.

Usage:
  python build_dataset.py --mode mini   # 200 samples for validation
  python build_dataset.py --mode full   # full dataset for training
"""

import os
import sys
import random
import shutil
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
IMG_DIR = PROJECT_ROOT / "data" / "raw" / "train"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
OUTPUT_DIR = PROJECT_ROOT / "data" / "yolo"
LOGS_DIR = PROJECT_ROOT / "logs"

SEED = 42


def build_dataset(mode="mini", sample_size=200, train_ratio=0.8):
    """
    Build YOLO dataset structure.
    mode='mini': sample sample_size images for quick validation
    mode='full': use all images
    """
    random.seed(SEED)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Building {mode.upper()} Dataset")
    print("=" * 60)

    # Collect all valid image-label pairs
    pairs = []
    label_files = set(f for f in os.listdir(LABELS_DIR) if f.endswith(".txt"))

    for label_file in sorted(label_files):
        base = label_file.rsplit(".", 1)[0]
        # Find matching image
        for ext in [".png", ".PNG"]:
            img_path = IMG_DIR / (base + ext)
            if img_path.exists():
                # Check if label is non-empty
                label_path = LABELS_DIR / label_file
                with open(label_path, "r") as f:
                    lines = f.readlines()
                if len(lines) > 0 and any(l.strip() for l in lines):
                    pairs.append((str(img_path), str(label_path), base))
                break

    print(f"Total valid pairs (with annotations): {len(pairs)}")

    # Sample if in mini mode
    if mode == "mini":
        if len(pairs) < sample_size:
            print(f"Warning: only {len(pairs)} pairs available, using all")
            sample_size = len(pairs)
        pairs = random.sample(pairs, sample_size)
        print(f"Sampled {sample_size} pairs")

    # Shuffle and split
    random.shuffle(pairs)
    split_idx = int(len(pairs) * train_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # Create directory structure
    for subset in ["train", "val"]:
        (OUTPUT_DIR / "images" / subset).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / subset).mkdir(parents=True, exist_ok=True)

    # Copy/symlink files
    def setup_subset(pairs_list, subset_name):
        for i, (img_src, lbl_src, base) in enumerate(pairs_list):
            if (i + 1) % 500 == 0:
                print(f"  {subset_name}: {i+1}/{len(pairs_list)}")

            # Symlink images (save disk space)
            img_dst = OUTPUT_DIR / "images" / subset_name / (base + os.path.splitext(img_src)[1])
            lbl_dst = OUTPUT_DIR / "labels" / subset_name / (base + ".txt")

            if not img_dst.exists():
                os.symlink(img_src, img_dst)
            if not lbl_dst.exists():
                shutil.copy2(lbl_src, lbl_dst)

    print("Linking train set...")
    setup_subset(train_pairs, "train")
    print("Linking val set...")
    setup_subset(val_pairs, "val")

    # Generate data.yaml
    yaml_path = OUTPUT_DIR / "data.yaml"
    yaml_content = f"""# YOLO Dataset Configuration
# Generated: {datetime.now().isoformat()}
# Mode: {mode}

path: {OUTPUT_DIR}
train: images/train
val: images/val

nc: 1
names:
  0: character
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\nDataset built at: {OUTPUT_DIR}")
    print(f"  data.yaml: {yaml_path}")
    print(f"  Train images: {len(train_pairs)}")
    print(f"  Val images:   {len(val_pairs)}")

    # Save log
    log_path = LOGS_DIR / f"build_dataset_{mode}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset build: {datetime.now().isoformat()}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Train: {len(train_pairs)}\n")
        f.write(f"Val: {len(val_pairs)}\n")
        f.write(f"Sample train IDs: {[p[2] for p in train_pairs[:10]]}\n")
        f.write(f"Sample val IDs: {[p[2] for p in val_pairs[:10]]}\n")

    print(f"Log saved to: {log_path}")
    return train_pairs, val_pairs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mini", "full"], default="mini")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()
    build_dataset(args.mode, args.samples, args.train_ratio)
