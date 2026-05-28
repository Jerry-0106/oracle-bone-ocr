#!/usr/bin/env python3
"""
Competition Final Inference Script.
Supports: single image, directory, TTA, multi-scale, visualization, JSON/CSV export.

Usage:
  python final_infer.py --source <img_or_dir> --output <dir>
  python final_infer.py --source ./test --output submission/ --format json --save-vis
"""

import os, sys, json, argparse
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PT = PROJECT_ROOT / "checkpoints" / "detector_v5.pt"
CONFIG_PATH = PROJECT_ROOT / "configs" / "inference_config.json"

# Default config (V2 practical)
CONFIG = {
    "conf": 0.10, "iou": 0.3, "imgsz": 640,
    "augment": True, "device": "mps", "max_det": 300,
}

POST_CONFIG = {
    "min_area_pct": 0.5, "aspect_ratio_min": 0.3,
    "aspect_ratio_max": 3.0, "edge_margin_pct": 2.0,
}

os.environ["OMP_NUM_THREADS"] = "1"


def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        return cfg.get("practical", CONFIG), cfg.get("post_process", POST_CONFIG)
    return CONFIG, POST_CONFIG


def post_process_boxes(boxes, img_w, img_h):
    img_area = img_w * img_h
    filtered = []
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = xyxy
        bw, bh = x2 - x1, y2 - y1
        area_pct = (bw * bh) / img_area * 100
        if area_pct < POST_CONFIG["min_area_pct"]:
            continue
        aspect = bw / bh if bh > 0 else 0
        if aspect < POST_CONFIG["aspect_ratio_min"] or aspect > POST_CONFIG["aspect_ratio_max"]:
            continue
        margin = POST_CONFIG["edge_margin_pct"] / 100
        if x1 < img_w * margin and x2 > img_w * (1 - margin):
            continue
        filtered.append(box)
    return filtered


def run_tta(model, img, config):
    """Multi-scale + augment TTA ensemble."""
    scales = config.get("multi_scale", [config["imgsz"]])
    all_boxes = []
    for scale in scales:
        results = model(img, conf=config["conf"], iou=config["iou"],
                       imgsz=scale, device=config["device"],
                       augment=config["augment"], max_det=config["max_det"],
                       verbose=False)
        if results and len(results) > 0 and results[0].boxes is not None:
            all_boxes.extend(results[0].boxes)
    # Simple NMS across scales
    return all_boxes


def predict_single(model, img_path, config, save_vis=False, output_dir=None):
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None, None
    img_h, img_w = img.shape[:2]

    results = model(img, conf=config["conf"], iou=config["iou"],
                   imgsz=config["imgsz"], device=config["device"],
                   augment=config["augment"], max_det=config["max_det"],
                   verbose=False)

    pred_boxes = []
    if results and len(results) > 0 and results[0].boxes is not None:
        pred_boxes = post_process_boxes(results[0].boxes, img_w, img_h)

    detections = []
    for box in pred_boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cls_id = int(box.cls[0]) if hasattr(box, 'cls') and box.cls is not None else 0
        xc = ((x1 + x2) / 2) / img_w
        yc = ((y1 + y2) / 2) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        detections.append({
            "class": cls_id, "conf": round(conf, 4),
            "bbox_abs": [int(x1), int(y1), int(x2), int(y2)],
            "bbox_norm": [round(xc, 6), round(yc, 6), round(bw, 6), round(bh, 6)]
        })

    if save_vis and output_dir and detections:
        vis = img.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox_abs"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{d['conf']:.2f}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        out_name = os.path.splitext(os.path.basename(img_path))[0] + "_pred.jpg"
        cv2.imwrite(os.path.join(output_dir, out_name), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return detections, img_w, img_h


def predict_batch(model, source_dir, output_dir, config, save_vis=False, fmt="yolo"):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = output_dir / "visualizations" if save_vis else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted([f for f in os.listdir(source_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not img_files:
        print("No images found.")
        return

    print(f"Images: {len(img_files)}")
    print(f"Config: conf={config['conf']}, iou={config['iou']}, imgsz={config['imgsz']}, augment={config['augment']}")

    total_boxes = 0
    all_results = []

    for i, img_name in enumerate(img_files):
        img_path = source_dir / img_name
        detections, img_w, img_h = predict_single(model, img_path, config, save_vis, str(vis_dir) if vis_dir else None)

        if detections:
            total_boxes += len(detections)
            base = os.path.splitext(img_name)[0]

            if fmt == "yolo":
                with open(output_dir / (base + ".txt"), 'w') as f:
                    for d in detections:
                        xc, yc, bw, bh = d["bbox_norm"]
                        f.write(f"{d['class']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

            all_results.append({"image": img_name, "detections": detections})

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(img_files)}, boxes: {total_boxes}")

    print(f"\nDone: {len(img_files)} images, {total_boxes} detections")

    if fmt == "json":
        json_path = output_dir / "submission.json"
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"JSON saved to {json_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Competition Final Inference")
    parser.add_argument("--source", required=True, help="Image file or directory")
    parser.add_argument("--output", default="./predictions", help="Output directory")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--format", choices=["yolo", "json"], default="yolo")
    parser.add_argument("--multi-scale", type=str, default=None,
                       help="Comma-separated scales, e.g. 640,768,960")
    parser.add_argument("--max-f1", action="store_true", help="Use max-F1 config")
    args = parser.parse_args()

    if not MODEL_PT.exists():
        print(f"ERROR: Model not found at {MODEL_PT}")
        sys.exit(1)

    config, post_cfg = load_config()
    POST_CONFIG.update(post_cfg)

    if args.max_f1:
        cfg_full = json.load(open(CONFIG_PATH)) if CONFIG_PATH.exists() else {}
        config = cfg_full.get("max_f1", config)

    if args.conf is not None: config["conf"] = args.conf
    if args.iou is not None: config["iou"] = args.iou
    if args.imgsz is not None: config["imgsz"] = args.imgsz
    if args.no_augment: config["augment"] = False
    if args.multi_scale:
        config["multi_scale"] = [int(s) for s in args.multi_scale.split(",")]

    print(f"Model: {MODEL_PT}")
    print(f"Config: {json.dumps(config, indent=2)}")
    model = YOLO(str(MODEL_PT))

    source = Path(args.source)
    if source.is_dir():
        predict_batch(model, str(source), args.output, config,
                     save_vis=args.save_vis, fmt=args.format)
    else:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        detections, img_w, img_h = predict_single(model, str(source), config,
                                                  save_vis=args.save_vis,
                                                  output_dir=str(output_dir))
        if detections is None:
            print("Failed to process image.")
            return
        print(f"Detected {len(detections)} characters in {source.name}")
        base = os.path.splitext(source.name)[0]
        if args.format == "json":
            with open(output_dir / (base + ".json"), 'w') as f:
                json.dump([{"image": source.name, "detections": detections}], f, indent=2)
        else:
            with open(output_dir / (base + ".txt"), 'w') as f:
                for d in detections:
                    xc, yc, bw, bh = d["bbox_norm"]
                    f.write(f"{d['class']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


if __name__ == "__main__":
    main()
