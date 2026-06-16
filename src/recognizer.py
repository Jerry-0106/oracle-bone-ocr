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


class ArcFaceRecognizer:
    """
    ConvNeXt-Tiny + ArcFace character classifier for competition inference.

    Loads checkpoints produced by Phase 4 training (train_arcface.py).
    Uses cosine logits (no angular margin) for inference.
    """

    def __init__(self, model_path, mapping_dir=None, device="cpu", input_size=None):
        self.model_path = str(model_path)
        if device in ("cuda", "cuda:0") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        print(f"Loading ArcFace recognizer: {model_path} on {self.device}")

        from src.arcface_model import build_arcface_model, load_arcface_checkpoint

        # Determine model config from checkpoint (before building)
        map_loc = torch.device(self.device)
        ckpt_meta = torch.load(model_path, map_location=map_loc, weights_only=False)
        num_classes = ckpt_meta.get('num_classes', 3483)
        embedding_dim = ckpt_meta.get('embedding_dim', 512)
        arc_s = ckpt_meta.get('arc_s', 30.0)
        arc_m = ckpt_meta.get('arc_m', 0.5)

        # input_size: explicit arg > checkpoint saved value > default 224
        if input_size is not None:
            self.input_size = input_size
        else:
            self.input_size = ckpt_meta.get('input_size', 224)

        # Resize value: keep same aspect-ratio-preserving ratio as 256/224
        self._resize_size = int(self.input_size * 256 / 224)

        print(f"  Config: num_classes={num_classes}, embedding_dim={embedding_dim}, s={arc_s}, m={arc_m}")
        print(f"  Input: {self.input_size}x{self.input_size} (resize={self._resize_size})")

        # Build model matching training architecture
        self.model = build_arcface_model(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            s=arc_s,
            m=arc_m,
        )
        self.model, self.meta = load_arcface_checkpoint(self.model, model_path, map_loc)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.num_classes = num_classes

        print(f"  Loaded: epoch={self.meta['epoch']}, best_top1={self.meta['best_top1']:.4f}")

        # Load mappings
        self.idx_to_class = {}
        self.class_to_char = {}
        self._load_mappings(mapping_dir)

        # Preprocessing — matches val_tfm from train_arcface.py EXACTLY when input_size=224
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize(self._resize_size),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_mappings(self, mapping_dir):
        """Load idx→char mappings.

        Two mapping formats are supported:
        1. ArcFace (new): idx_to_class maps index→Chinese character directly.
           e.g., {"0": "一", "1": "丁", ...}
        2. Legacy (old): idx_to_class maps index→competition ID, then
           ID_to_chinese.json maps competition ID→Chinese character.
           e.g., {"0": "0081", ...} + {"0081": "一", ...}
        """
        from pathlib import Path
        mapping_dir = Path(mapping_dir) if mapping_dir else Path(__file__).resolve().parent.parent / "mappings"

        # Load idx_to_class (model output index → class identifier)
        idx_path = mapping_dir / "idx_to_class.json"
        id_path = mapping_dir / "ID_to_chinese.json"

        if idx_path.exists():
            with open(idx_path, encoding='utf-8') as f:
                raw = json.load(f)
                for k, v in raw.items():
                    self.idx_to_class[str(k)] = v

        # Determine mapping type and resolve characters
        if not self.idx_to_class:
            print("  WARNING: No idx_to_class mapping found!")
            return

        # Check if values are Chinese characters (ArcFace) or competition IDs (legacy)
        sample_val = next(iter(self.idx_to_class.values()))
        is_direct_char = len(sample_val) == 1 and ord(sample_val) > 127

        if is_direct_char:
            # ArcFace format: values ARE the Chinese characters
            self.class_to_char = dict(self.idx_to_class)
            print(f"  Mapping type: DIRECT (index → Chinese char)")

        elif id_path.exists():
            # Legacy format: values are competition IDs, need ID→char lookup
            with open(id_path, encoding='utf-8') as f:
                id_to_chinese = json.load(f)
            for idx_str, class_id in self.idx_to_class.items():
                chinese = id_to_chinese.get(class_id)
                if chinese is None and '_' in class_id:
                    for sub_id in class_id.split('_'):
                        chinese = id_to_chinese.get(sub_id)
                        if chinese:
                            break
                self.class_to_char[idx_str] = chinese if chinese else class_id
            print(f"  Mapping type: LEGACY (index → competition ID → Chinese char)")

        else:
            # Fallback: use class_id as char
            print("  WARNING: No character mappings found; outputting class IDs as text")
            for idx_str, class_id in self.idx_to_class.items():
                self.class_to_char[idx_str] = class_id

        print(f"  Loaded {len(self.idx_to_class)} class mappings, {len(self.class_to_char)} char mappings")

    @torch.no_grad()
    def predict(self, image, topk=5):
        """Predict single PIL image."""
        from PIL import Image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            from torchvision.transforms.functional import to_pil_image
            image = to_pil_image(image)

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        outputs = self.model(tensor, label=None)  # Inference: cosine logits
        logits = outputs['logits']
        probs = F.softmax(logits, dim=1)
        topk_probs, topk_indices = torch.topk(probs, min(topk, self.num_classes), dim=1)

        results = []
        for i in range(len(topk_indices[0])):
            idx = str(topk_indices[0][i].item())
            prob = topk_probs[0][i].item()
            cls_id = self.idx_to_class.get(idx, idx)
            chinese = self.class_to_char.get(idx, cls_id)
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
        """Predict batch of PIL images."""
        tensors = torch.stack([self.transform(img) for img in images]).to(self.device)
        outputs = self.model(tensors, label=None)  # Inference: cosine logits
        logits = outputs['logits']
        probs = F.softmax(logits, dim=1)
        topk_probs, topk_indices = torch.topk(probs, min(topk, self.num_classes), dim=1)

        batch_results = []
        for b in range(len(images)):
            results = []
            for i in range(len(topk_indices[b])):
                idx = str(topk_indices[b][i].item())
                prob = topk_probs[b][i].item()
                cls_id = self.idx_to_class.get(idx, idx)
                chinese = self.class_to_char.get(idx, cls_id)
                results.append({
                    "rank": i + 1,
                    "class_index": int(idx),
                    "class_id": cls_id,
                    "char": chinese,
                    "confidence": round(prob, 4),
                })
            batch_results.append(results)
        return batch_results
