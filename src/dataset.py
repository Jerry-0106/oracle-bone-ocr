"""
Recognition Dataset for HUST-OBC Deciphered Characters.
Supports both raw deciphered/ structure and pre-split train/val/ structure.
"""

import os, json
from pathlib import Path
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


class HUSTOBCDataset(Dataset):
    """HUST-OBC deciphered character classification dataset."""

    def __init__(self, root_dir, transform=None, split='train', val_ratio=0.2, seed=42):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split

        # Check for pre-split directories first
        train_dir = self.root_dir / 'train'
        val_dir = self.root_dir / 'val'
        deciphered_dir = self.root_dir / 'deciphered'

        if train_dir.exists() and val_dir.exists():
            # Use pre-split data
            if split == 'train':
                source_dir = train_dir
            elif split == 'val':
                source_dir = val_dir
            else:
                source_dir = deciphered_dir if deciphered_dir.exists() else self.root_dir
            self._build_from_dir(source_dir, is_split=True)
        elif deciphered_dir.exists():
            self._build_from_dir(deciphered_dir, is_split=False)
        else:
            raise FileNotFoundError(
                f"No data found at {root_dir}\n"
                f"Expected train/val/ or deciphered/ subdirectories.\n"
                f"Download HUST-OBC from: https://figshare.com/s/8a9c0420312d94fc01e3"
            )

        # Train/val split for raw deciphered data
        if not self._pre_split:
            self._apply_split(val_ratio, seed)

    def _build_from_dir(self, source_dir, is_split=False):
        """Build class index and sample list from a directory, recursing into subdirs."""
        self.classes = sorted([d.name for d in source_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        # Recursively collect all images
        self.samples = []
        for cls_name in self.classes:
            cls_path = source_dir / cls_name
            for img_path in cls_path.rglob('*'):
                if img_path.is_file() and img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}:
                    self.samples.append((str(img_path), self.class_to_idx[cls_name]))

        self._pre_split = is_split
        self._all_samples = self.samples[:] if not is_split else None

    def _apply_split(self, val_ratio, seed):
        """Create train/val split internally."""
        generator = torch.Generator().manual_seed(seed)
        n_val = int(len(self.samples) * val_ratio)
        n_train = len(self.samples) - n_val
        train_idx, val_idx = random_split(
            range(len(self.samples)), [n_train, n_val], generator=generator
        )

        if self.split == 'train':
            self.samples = [self.samples[i] for i in train_idx]
        elif self.split == 'val':
            self.samples = [self.samples[i] for i in val_idx]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (255, 255, 255))

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self):
        counter = Counter()
        for _, label in self.samples:
            counter[self.idx_to_class[label]] += 1
        return counter


def get_transforms(config, is_train=True):
    """Build transforms from config."""
    if is_train:
        aug = config.get("augmentation", {})
        tfms = []
        if aug.get("random_resized_crop", True):
            tfms.append(transforms.RandomResizedCrop(
                config["input_size"], scale=(0.8, 1.0), ratio=(0.9, 1.1)))
        else:
            tfms.append(transforms.Resize(256))
            tfms.append(transforms.CenterCrop(config["input_size"]))

        if aug.get("randaugment", True):
            from torchvision.transforms import RandAugment
            tfms.append(RandAugment(
                num_ops=aug.get("randaugment_n", 2),
                magnitude=aug.get("randaugment_m", 9)))

        if aug.get("rotation", 0) > 0:
            tfms.append(transforms.RandomRotation(aug["rotation"]))

        cj = aug.get("color_jitter", {})
        if cj:
            tfms.append(transforms.ColorJitter(**cj))

        tfms.append(transforms.ToTensor())
        tfms.append(transforms.Normalize(
            mean=aug.get("normalize_mean", [0.485, 0.456, 0.406]),
            std=aug.get("normalize_std", [0.229, 0.224, 0.225])))
        return transforms.Compose(tfms)
    else:
        aug = config.get("val_augmentation", {})
        tfms = [
            transforms.Resize(aug.get("resize", 256)),
            transforms.CenterCrop(aug.get("center_crop", config["input_size"])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=aug.get("normalize_mean", [0.485, 0.456, 0.406]),
                std=aug.get("normalize_std", [0.229, 0.224, 0.225])),
        ]
        return transforms.Compose(tfms)


def get_dataloaders(data_dir, config, num_workers=0):
    """Create train and val dataloaders."""
    train_tfms = get_transforms(config, is_train=True)
    val_tfms = get_transforms(config, is_train=False)

    full_dataset = HUSTOBCDataset(data_dir, transform=None, split='all')

    train_dataset = HUSTOBCDataset(data_dir, transform=train_tfms, split='train')
    val_dataset = HUSTOBCDataset(data_dir, transform=val_tfms, split='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, full_dataset


def analyze_dataset(data_dir, output_dir):
    """Analyze dataset and save statistics."""
    dataset = HUSTOBCDataset(data_dir, split='all')
    dist = dataset.get_class_distribution()
    counts = list(dist.values())

    stats = {
        "num_classes": dataset.num_classes,
        "total_samples": len(dataset.samples),
        "avg_per_class": round(len(dataset.samples) / dataset.num_classes, 1) if dataset.num_classes else 0,
        "min_samples": min(counts) if counts else 0,
        "max_samples": max(counts) if counts else 0,
        "median_samples": sorted(counts)[len(counts)//2] if counts else 0,
        "classes_with_1_sample": sum(1 for c in counts if c == 1),
        "classes_with_lt_5_samples": sum(1 for c in counts if c < 5),
        "classes_with_lt_10_samples": sum(1 for c in counts if c < 10),
        "class_imbalance_ratio": round(max(counts) / max(min(counts), 1), 1),
    }

    print(f"\nDataset Analysis:")
    print(f"  Classes: {stats['num_classes']}")
    print(f"  Total images: {stats['total_samples']}")
    print(f"  Avg/class: {stats['avg_per_class']}")
    print(f"  Min/class: {stats['min_samples']}, Max/class: {stats['max_samples']}")
    print(f"  Classes with 1 sample: {stats['classes_with_1_sample']}")
    print(f"  Classes with <5 samples: {stats['classes_with_lt_5_samples']}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    with open(output_dir / "class_to_idx.json", 'w') as f:
        json.dump(dataset.class_to_idx, f, indent=2, ensure_ascii=False)
    with open(output_dir / "idx_to_class.json", 'w') as f:
        json.dump(dataset.idx_to_class, f, indent=2, ensure_ascii=False)

    return stats, dataset
