#!/usr/bin/env python3
"""
Recognition Training — Oracle Bone Character Classification.
EfficientNet-B0 baseline, MPS-compatible.

Usage:
  python train.py --data_dir ../recognition_data --output_dir ./weights
  python train.py --data_dir ../recognition_data --model resnet18 --epochs 50
"""

import os, sys, json, time, argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import confusion_matrix, classification_report

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_dataloaders, analyze_dataset, HUSTOBCDataset
from src.models import build_model, count_parameters


def train_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    n_batches = len(loader)
    t0 = time.time()
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)
        if (batch_idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (batch_idx + 1) * (n_batches - batch_idx - 1)
            print(f"  [{epoch}/{total_epochs}] batch {batch_idx+1}/{n_batches} | "
                  f"loss: {total_loss/total:.4f} | acc: {correct/total:.4f} | "
                  f"ETA: {eta:.0f}s", flush=True)
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    acc = correct / total
    top5 = compute_topk_accuracy(torch.tensor(all_preds), torch.tensor(all_labels), k=5)
    return total_loss / total, acc, top5, all_preds, all_labels


def compute_topk_accuracy(preds, labels, k=5):
    correct = preds.eq(labels).sum().item()
    return correct / len(labels)


def main():
    parser = argparse.ArgumentParser(description="Train Recognition Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to HUST-OBC data")
    parser.add_argument("--output_dir", type=str, default="./weights", help="Output for model weights")
    parser.add_argument("--config", type=str, default=None, help="Config JSON path")
    parser.add_argument("--model", type=str, default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Load config
    config_path = args.config or str(PROJECT_ROOT / "configs" / "baseline_config.json")
    with open(config_path) as f:
        config = json.load(f)

    config["model"] = args.model
    if args.epochs: config["epochs"] = args.epochs
    if args.batch_size: config["batch_size"] = args.batch_size
    if args.lr: config["lr"] = args.lr
    if args.device: config["device"] = args.device

    device = torch.device(config["device"] if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Analyze dataset first
    data_dir = args.data_dir
    stats_dir = Path(args.output_dir).parent / "reports"
    data_stats, _ = analyze_dataset(data_dir, stats_dir)
    config["num_classes"] = data_stats["num_classes"]

    print(f"\nModel: {config['model']}")
    print(f"Classes: {config['num_classes']}")
    print(f"Data: {data_dir}")

    # Data
    train_loader, val_loader, _ = get_dataloaders(data_dir, config)

    # Model
    model = build_model(config["model"], num_classes=config["num_classes"], pretrained=True)
    model = model.to(device)
    print(f"Parameters: {count_parameters(model):.1f}M")

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.1))

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 1e-4))

    # LR schedule: warmup + cosine
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=config.get("warmup_epochs", 3))
    cosine = CosineAnnealingLR(optimizer, T_max=config["epochs"] - config.get("warmup_epochs", 3))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                            milestones=[config.get("warmup_epochs", 3)])

    start_epoch = 0
    best_acc = 0.0
    patience_counter = 0
    history = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_pt = output_dir / "best.pt"
    last_pt = output_dir / "last.pt"

    print(f"\n{'='*50}")
    print(f"Training {config['epochs']} epochs (patience={config['early_stopping_patience']})")
    print(f"Batches: train={len(train_loader)}, val={len(val_loader)}")
    print(f"{'='*50}")

    for epoch in range(start_epoch, config["epochs"]):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch+1, config["epochs"])
        val_loss, val_acc, val_top5, all_preds, all_labels = validate(model, val_loader, criterion, device)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "val_top5_acc": round(val_top5, 6),
            "lr": lr,
            "time": round(elapsed, 1),
        })

        print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
              f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | "
              f"top5: {val_top5:.4f} | lr: {lr:.6f} | {elapsed:.0f}s", flush=True)

        # Checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
                "config": config,
                "history": history,
            }, best_pt)
            print(f"  -> Best model saved (acc={best_acc:.4f})", flush=True)
        else:
            patience_counter += 1

        # Save last
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
            "config": config,
            "history": history,
        }, last_pt)

        if patience_counter >= config["early_stopping_patience"]:
            print(f"\nEarly stopping at epoch {epoch+1}", flush=True)
            break

    print(f"\n{'='*50}")
    print(f"Training Complete. Best val_acc: {best_acc:.4f}")
    print(f"{'='*50}")

    # Save history
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    # Confusion matrix for best model
    best_ckpt = torch.load(best_pt, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    _, _, _, all_preds, all_labels = validate(model, val_loader, criterion, device)

    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    top_errors = []
    for i in range(min(len(cm), 50)):
        for j in range(min(len(cm), 50)):
            if i != j and cm[i][j] > 0:
                top_errors.append((int(i), int(j), int(cm[i][j])))
    top_errors.sort(key=lambda x: x[2], reverse=True)
    top_errors = top_errors[:20]

    print(f"\nTop 20 Confused Pairs:")
    for i, j, count in top_errors:
        print(f"  Class {i} -> Class {j}: {count} errors")

    # Save confusion analysis
    confusion_data = {
        "top_confused_pairs": [{"from": int(i), "to": int(j), "count": int(c)} for i, j, c in top_errors],
        "val_accuracy": float(best_acc),
    }
    with open(output_dir / "confusion_analysis.json", 'w') as f:
        json.dump(confusion_data, f, indent=2)

    # Save inference-ready model
    model.eval()
    torch.save(model.state_dict(), output_dir / "model_state_dict.pt")


if __name__ == "__main__":
    main()
