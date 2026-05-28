#!/usr/bin/env python3
"""
Domain Augmentation — creates augmented copies of training images in-place.
Images saved with _aug1, _aug2 suffixes in the SAME directory as originals.
No file copying — originals stay where they are.

Bboxes preserved — only pixel-level appearance changes.
"""
import sys, argparse, random
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "data" / "yolo"


def build_transforms(severity="medium"):
    """Return list of (name, Albumentations transform) tuples."""
    import albumentations as A

    if severity == "light":
        bc = (0.15, 0.15); gamma = (90, 110); blur = (3, 5)
        noise = (0.02, 0.08); jpeg_q = (60, 95); ds = (0.8, 0.95)
    elif severity == "strong":
        bc = (0.35, 0.35); gamma = (60, 140); blur = (3, 9)
        noise = (0.04, 0.20); jpeg_q = (30, 85); ds = (0.4, 0.85)
    else:
        bc = (0.25, 0.25); gamma = (75, 125); blur = (3, 7)
        noise = (0.02, 0.14); jpeg_q = (50, 90); ds = (0.6, 0.9)

    return [
        A.RandomBrightnessContrast(brightness_limit=bc[0], contrast_limit=bc[1], p=0.7),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
        A.RandomGamma(gamma_limit=gamma, p=0.5),
        A.GaussianBlur(blur_limit=blur, p=0.5),
        A.MotionBlur(blur_limit=blur, p=0.3),
        A.GaussNoise(std_range=noise, p=0.5),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.4),
        A.ImageCompression(compression_type='jpeg', quality_range=jpeg_q, p=0.5),
        A.Downscale(scale_range=ds, p=0.4),
    ]


def augment_image(image, transforms, min_augs=2, max_augs=5):
    """Apply random subset of transforms."""
    n = random.randint(min_augs, min(max_augs, len(transforms)))
    selected = random.sample(transforms, n)
    import albumentations as A
    pipeline = A.Compose(selected)
    return pipeline(image=image)["image"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--severity", default="medium", choices=["light", "medium", "strong"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--clean", action="store_true", help="Remove old _aug images first")
    args = parser.parse_args()

    img_dir = DATASET_DIR / "images" / "train"
    lbl_dir = DATASET_DIR / "labels" / "train"

    # Clean old augmentations
    if args.clean:
        old = list(img_dir.glob("*_aug[0-9]*")) + list(lbl_dir.glob("*_aug[0-9]*"))
        for f in old:
            f.unlink()
        print(f"Cleaned {len(old)} old augmented files")

    images = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
    # Skip previously augmented images
    images = [p for p in images if "_aug" not in p.stem]
    if args.limit > 0:
        images = images[:args.limit]

    print(f"Images: {len(images)}, Variants: {args.variants}, Severity: {args.severity}")
    transforms = build_transforms(args.severity)

    total = 0
    for img_path in tqdm(images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lbl_path = lbl_dir / f"{img_path.stem}.txt"

        for v in range(args.variants):
            aug = augment_image(img_rgb, transforms)
            aug_bgr = cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)
            aug_name = f"{img_path.stem}_aug{v+1}{img_path.suffix}"
            cv2.imwrite(str(img_dir / aug_name), aug_bgr)

            if lbl_path.exists():
                import shutil
                shutil.copy2(lbl_path, lbl_dir / f"{img_path.stem}_aug{v+1}.txt")
            total += 1

    print(f"Done: {total} augmented images created in {img_dir}")
    # Show total image count
    all_imgs = list(img_dir.glob("*"))
    print(f"Total training images now: {len(all_imgs)}")


if __name__ == "__main__":
    main()
