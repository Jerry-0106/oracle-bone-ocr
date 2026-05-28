#!/usr/bin/env python3
"""
Prepare Recognition Data from HUST-OBC Download.
Extracts and organizes the deciphered character dataset for training.

Usage:
  # After downloading HUST-OBC from https://figshare.com/s/8a9c0420312d94fc01e3
  python prepare_data.py --source ~/Downloads/HUST-OBC --output ../recognition_data
"""

import os, sys, json, argparse, shutil
from pathlib import Path
from collections import Counter


def prepare_dataset(source_dir, output_dir):
    source = Path(source_dir)
    output = Path(output_dir)

    # Find deciphered directory
    deciphered_src = None
    for candidate in [source / "deciphered", source / "HUST-OBC" / "deciphered", source]:
        if candidate.exists() and any((candidate / d).is_dir() for d in os.listdir(candidate) if (candidate / d).is_dir()):
            deciphered_src = candidate
            break

    if not deciphered_src:
        print("ERROR: Could not find deciphered directory.")
        print(f"Searched: {source}")
        print("Expected structure: HUST-OBC/deciphered/ID1/..., ID2/...")
        sys.exit(1)

    print(f"Found deciphered data at: {deciphered_src}")

    # Load mappings if available
    chinese_to_id = {}
    id_to_chinese = {}
    for mapping_file in ["chinese_to_ID.json", "chinese_to_id.json"]:
        mp = deciphered_src / mapping_file
        if mp.exists():
            with open(mp, 'r', encoding='utf-8') as f:
                chinese_to_id = json.load(f)
            break
    for mapping_file in ["ID_to_chinese.json", "id_to_chinese.json"]:
        mp = deciphered_src / mapping_file
        if mp.exists():
            with open(mp, 'r', encoding='utf-8') as f:
                id_to_chinese = json.load(f)
            break

    # Create output structure
    output_deciphered = output / "deciphered"
    output_deciphered.mkdir(parents=True, exist_ok=True)

    mappings_dir = output / "mappings"
    mappings_dir.mkdir(parents=True, exist_ok=True)

    # Copy/symlink class directories
    class_dirs = sorted([d for d in deciphered_src.iterdir() if d.is_dir()])
    print(f"Found {len(class_dirs)} class directories")

    class_stats = {}
    total_images = 0

    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        images = list(cls_dir.glob("*.[jpJP][pnPN]*[gG]")) + \
                 list(cls_dir.glob("*.[bB][mM][pP]")) + \
                 list(cls_dir.glob("*.[tT][iI][fF]*"))
        if not images:
            continue

        dest_dir = output_deciphered / cls_name
        dest_dir.mkdir(exist_ok=True)

        for img in images:
            dest = dest_dir / img.name
            if not dest.exists():
                shutil.copy2(img, dest)

        class_stats[cls_name] = len(images)
        total_images += len(images)

    print(f"Total: {total_images} images across {len(class_stats)} classes")

    # Save mappings
    if chinese_to_id:
        with open(mappings_dir / "chinese_to_id.json", 'w', encoding='utf-8') as f:
            json.dump(chinese_to_id, f, indent=2, ensure_ascii=False)
    if id_to_chinese:
        with open(mappings_dir / "id_to_chinese.json", 'w', encoding='utf-8') as f:
            json.dump(id_to_chinese, f, indent=2, ensure_ascii=False)

    # Build class index
    classes = sorted(class_stats.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {str(i): c for i, c in enumerate(classes)}
    with open(mappings_dir / "class_to_idx.json", 'w') as f:
        json.dump(class_to_idx, f, indent=2, ensure_ascii=False)
    with open(mappings_dir / "idx_to_class.json", 'w') as f:
        json.dump(idx_to_class, f, indent=2, ensure_ascii=False)

    # Save statistics
    counts = list(class_stats.values())
    stats = {
        "num_classes": len(classes),
        "total_images": total_images,
        "avg_per_class": round(total_images / len(classes), 1),
        "min_per_class": min(counts),
        "max_per_class": max(counts),
        "classes_with_1_sample": sum(1 for c in counts if c == 1),
        "classes_with_lt_5": sum(1 for c in counts if c < 5),
        "classes_with_lt_10": sum(1 for c in counts if c < 10),
    }
    with open(output / "stats" / "dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    (output / "stats").mkdir(parents=True, exist_ok=True)

    print(f"\nDataset Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print(f"\nData prepared at: {output}")
    print(f"Ready for training: python recognition/train.py --data_dir {output}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Recognition Data")
    parser.add_argument("--source", required=True, help="Path to downloaded HUST-OBC dataset")
    parser.add_argument("--output", required=True, help="Output directory for prepared data")
    args = parser.parse_args()
    prepare_dataset(args.source, args.output)


if __name__ == "__main__":
    main()
