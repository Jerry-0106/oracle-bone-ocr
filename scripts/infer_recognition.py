#!/usr/bin/env python3
"""
Recognition Inference Script.
Supports single image, crop directory, and full pipeline.

Usage:
  python infer.py --image crop.jpg --model weights/best.pt
  python infer.py --dir crops/ --model weights/best.pt
"""

import os, sys, json, argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model


def load_model(model_path, num_classes, model_name="efficientnet_b0", device="mps"):
    device = torch.device(device if torch.backends.mps.is_available() else "cpu")
    model = build_model(model_name, num_classes=num_classes, pretrained=False)

    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        config = ckpt.get("config", {})
    else:
        model.load_state_dict(ckpt)
        config = {}

    model = model.to(device)
    model.eval()
    return model, device, config


def get_transform(input_size=224):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def predict(model, image_path, transform, device, topk=5, idx_to_class=None):
    """Predict top-k classes for an image."""
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        return None, str(e)

    tensor = transform(image).unsqueeze(0).to(device)
    outputs = model(tensor)
    probs = F.softmax(outputs, dim=1)
    topk_probs, topk_indices = torch.topk(probs, topk, dim=1)

    results = []
    for i in range(topk):
        idx = topk_indices[0][i].item()
        prob = topk_probs[0][i].item()
        cls_name = idx_to_class.get(str(idx), str(idx)) if idx_to_class else str(idx)
        results.append({"rank": i + 1, "class_id": idx, "class_name": cls_name, "confidence": round(prob, 4)})

    return results, None


def predict_batch(model, image_dir, transform, device, topk=5, idx_to_class=None):
    """Predict for all images in a directory."""
    image_dir = Path(image_dir)
    img_files = sorted([f for f in os.listdir(image_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    results = {}
    for i, img_name in enumerate(img_files):
        img_path = image_dir / img_name
        pred, err = predict(model, img_path, transform, device, topk, idx_to_class)
        if pred:
            results[img_name] = pred
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(img_files)}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Recognition Inference")
    parser.add_argument("--image", type=str, default=None, help="Single image path")
    parser.add_argument("--dir", type=str, default=None, help="Directory of crop images")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--num_classes", type=int, default=1588)
    parser.add_argument("--model_name", type=str, default="efficientnet_b0")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--mapping", type=str, default=None, help="idx_to_class.json path")
    parser.add_argument("--output", type=str, default="recognition_results.json")
    args = parser.parse_args()

    # Load class mapping
    idx_to_class = None
    if args.mapping and Path(args.mapping).exists():
        with open(args.mapping) as f:
            idx_to_class = json.load(f)

    model, device, config = load_model(args.model, args.num_classes, args.model_name, args.device)
    print(f"Model loaded: {args.model} on {device}")

    tfm = get_transform(config.get("input_size", 224))

    if args.image:
        results, err = predict(model, args.image, tfm, device, args.topk, idx_to_class)
        if err:
            print(f"Error: {err}")
            return
        print(f"\nPredictions for {args.image}:")
        for r in results:
            print(f"  {r['rank']}. {r['class_name']} (conf={r['confidence']})")

    elif args.dir:
        print(f"Predicting images in {args.dir}...")
        results = predict_batch(model, args.dir, tfm, device, args.topk, idx_to_class)
        print(f"\nProcessed {len(results)} images")

        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")
    else:
        print("Specify --image or --dir")


if __name__ == "__main__":
    main()
