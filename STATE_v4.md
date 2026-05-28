# v4 Character-Level Detector — Current State

**Updated: 2026-05-27 00:20 | V5 training epoch 7/30 in progress, mAP50=0.548**

## Active: V5 Detector Training (Domain-Augmented)

**Status**: Running, epoch 7/30, resumed from epoch 6 checkpoint after terminal disconnect.

**Training config**:
- Model: YOLOv8n pretrained, nc=1, 3,011,043 params
- Data: 14,749 train (4,925 original + 9,824 domain-augmented), 1,193 val
- imgsz=640, batch=8, epochs=30, patience=10
- Aug: degrees=2, translate=0.1, scale=0.3, shear=0, flip OFF
- HSV: h=0.02, s=0.7, v=0.5, mosaic=0.3, close_mosaic=5
- Optimizer: AdamW(lr=0.002), cos_lr=True, AMP
- Workers=2, KMP_DUPLICATE_LIB_OK=TRUE (Windows workarounds)

**Progress**:
| Epoch | mAP50 | P | R | mAP50-95 | box_loss |
|-------|-------|-----|-----|-----------|----------|
| 1 | 0.401 | 0.503 | 0.423 | 0.205 | 1.952 |
| 2 | 0.440 | 0.538 | 0.449 | 0.230 | 1.823 |
| 3 | 0.471 | 0.543 | 0.503 | 0.240 | 1.790 |
| 4 | 0.492 | 0.534 | 0.517 | 0.267 | 1.742 |
| 5 | 0.541 | 0.588 | 0.552 | 0.295 | 1.689 |
| 6 | 0.548 | 0.600 | 0.550 | 0.304 | 1.664 |

**Epoch 5 Per-Source F1 (tracked sources)**:
- HANJINWENLU6: 0.0% | HANJINWENLU7: 0.0% | OUMISOUCHU: 0.0%
- HANJINWENLU5: 5.3% | HANJINWENLU4: 7.1% | HANJINWENLU1: 7.4%
- MINGWENXUAN: 18.4% | HANJINWENLU3: 28.1%
- DONGYANGWENKU: 49.5% (baseline 37.7%, +11.8pp)
- HANJINWENLU2: 50.8% (baseline 40.0%, +10.8pp)

**Auto-scheduled**:
- Checkpoints saved at epoch 10, 15, 20 → `runs/detector_v5/weights/epoch{N}.pt`
- Per-source eval at epoch 10, 15, 20, 25, 30 → `evaluation/per_source/v5_epoch{N}_per_source.json`
- Monitor script: `scripts/monitor_training.py` (PID varies, restart if needed)

## Running Processes

```
python scripts/train_detector_v5.py   (→ logs/train_detector_v5.log)
python scripts/monitor_training.py    (→ logs/monitor.log)
```

## Resume Commands

```bash
# Check training progress
tail -c 1000 logs/train_detector_v5.log | strings | grep -E "^\d+/30" | tail -1
cat runs/detector_v5/results.csv | tail -1

# Check monitor
cat logs/monitor.log

# Kill and resume (if crash)
taskkill //F //IM python.exe
export KMP_DUPLICATE_LIB_OK=TRUE
cd E:/orc_project/orc_project
./venv/Scripts/python.exe scripts/train_detector_v5.py >> logs/train_detector_v5.log 2>&1 &
./venv/Scripts/python.exe scripts/monitor_training.py > logs/monitor.log 2>&1 &

# Per-source eval (manual)
./venv/Scripts/python.exe scripts/eval_per_source.py \
  --detector runs/detector_v5/weights/best.pt \
  --min-samples 3 --output v5_epoch{N}_per_source.json

# Per-source eval vs baseline
./venv/Scripts/python.exe scripts/eval_per_source.py \
  --detector project/models/detector.pt \
  --min-samples 3 --output baseline_per_source.json
```

## Key Files

```
runs/detector_v5/
  weights/best.pt          (epoch 6, 24.5MB)
  weights/last.pt          (epoch 6, 24.5MB)
  results.csv              (epochs 1-6)
evaluation/per_source/
  baseline_per_source.json (competition_final detector F1)
  v5_epoch3_per_source.json
  v5_epoch5_per_source.json
scripts/
  train_detector_v5.py     (auto-resume, workers=2)
  monitor_training.py      (auto-eval + checkpoint save)
  augment_dataset.py       (domain augmentation)
  eval_per_source.py       (per-source F1 with --output)
  sweep_detector.py        (conf/iou/max_det sweep)
  analyze_stage3.py        (compound region analysis)
```

## Windows-Specific Fixes Applied

1. `if __name__ == '__main__':` guard (multiprocessing spawn)
2. `KMP_DUPLICATE_LIB_OK=TRUE` (OpenMP conflict)
3. `workers=2` (stable, faster than 0)
4. `Albumentations` API 2.0.8: `GaussNoise(std_range=...)`, `ImageCompression(compression_type='jpeg')`
5. Auto-resume: loads `last.pt` as model, sets `resume=True`

## Previous Stages (Completed)

- Stage 1: conf=0.18 optimal (F1=59.15%, +4.8%)
- Rec conf filtering: DEAD END
- Stage 3: Compound adhesion NOT main error (84.4% FP are large boxes)
- Per-source baseline: HANJINWENLU6/7, OUMISOUCHU at 0% F1
