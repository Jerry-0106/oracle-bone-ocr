"""
Recognition Wrapper for Oracle Bone Character Classification.
Self-contained: loads mappings from ocr_pipeline/mappings/.
"""

import json
from pathlib import Path
import torch
import torch.nn.functional as F

PIPELINE_ROOT = Path(__file__).resolve().parent.parent  # project root


class Recognizer:
    """EfficientNet-based character classifier with top-k support."""

    def __init__(self, model_path, mapping_dir=None, device="cpu", model_name="efficientnet_b0"):
        self.model_path = str(model_path)
        # Safety: never use cuda if not actually available
        if device in ("cuda", "cuda:0") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        torch_device = torch.device(self.device)
        self.model_name = model_name

        print(f"Loading recognizer: {model_path} on {self.device}")
        ckpt = torch.load(model_path, map_location=torch_device, weights_only=True)

        self.config = ckpt.get("config", {})
        self.num_classes = self.config.get("num_classes", 1588)
        self.input_size = self.config.get("input_size", 224)

        self.model = self._build_model()
        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model.load_state_dict(ckpt)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.idx_to_class = {}
        self.class_to_chinese = {}
        self._load_mappings(mapping_dir)

        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _build_model(self):
        from torchvision import models
        import torch.nn as nn

        name = self.model_name.lower()
        if name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
        elif name == "efficientnet_b1":
            model = models.efficientnet_b1(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
        elif name == "resnet18":
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        else:
            raise ValueError(f"Unknown model: {name}")
        return model

    def _load_mappings(self, mapping_dir):
        mapping_dir = Path(mapping_dir) if mapping_dir else PIPELINE_ROOT / "mappings"
        idx_path = mapping_dir / "idx_to_class.json"
        id_path = mapping_dir / "ID_to_chinese.json"

        if idx_path.exists():
            with open(idx_path, encoding='utf-8') as f:
                self.idx_to_class = json.load(f)

        if id_path.exists():
            with open(id_path, encoding='utf-8') as f:
                id_to_chinese = json.load(f)
                for idx, class_id in self.idx_to_class.items():
                    chinese = id_to_chinese.get(class_id)
                    if chinese is None and '_' in class_id:
                        for sub_id in class_id.split('_'):
                            chinese = id_to_chinese.get(sub_id)
                            if chinese:
                                break
                    self.class_to_chinese[idx] = chinese if chinese else class_id

        print(f"Loaded {len(self.idx_to_class)} class mappings, {len(self.class_to_chinese)} Chinese mappings")

    @torch.no_grad()
    def predict(self, image, topk=5):
        from PIL import Image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            from torchvision.transforms.functional import to_pil_image
            image = to_pil_image(image)

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        outputs = self.model(tensor)
        probs = F.softmax(outputs, dim=1)
        topk_probs, topk_indices = torch.topk(probs, min(topk, self.num_classes), dim=1)

        results = []
        for i in range(len(topk_indices[0])):
            idx = str(topk_indices[0][i].item())
            prob = topk_probs[0][i].item()
            cls_id = self.idx_to_class.get(idx, idx)
            chinese = self.class_to_chinese.get(idx, cls_id)
            results.append({
                "rank": i + 1,
                "class_index": int(idx),
                "class_id": cls_id,
                "char": chinese,
                "confidence": round(prob, 4),
            })
        return results

    @torch.no_grad()
    def predict_batch(self, images, topk=5):
        tensors = torch.stack([self.transform(img) for img in images]).to(self.device)
        outputs = self.model(tensors)
        probs = F.softmax(outputs, dim=1)
        topk_probs, topk_indices = torch.topk(probs, min(topk, self.num_classes), dim=1)

        batch_results = []
        for b in range(len(images)):
            results = []
            for i in range(len(topk_indices[b])):
                idx = str(topk_indices[b][i].item())
                prob = topk_probs[b][i].item()
                cls_id = self.idx_to_class.get(idx, idx)
                chinese = self.class_to_chinese.get(idx, cls_id)
                results.append({
                    "rank": i + 1,
                    "class_index": int(idx),
                    "class_id": cls_id,
                    "char": chinese,
                    "confidence": round(prob, 4),
                })
            batch_results.append(results)
        return batch_results
