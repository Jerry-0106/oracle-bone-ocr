"""
YOLO Detection Wrapper for Oracle Bone Characters.
"""

from pathlib import Path
import torch
from ultralytics import YOLO


class Detector:
    """YOLOv8 detector with TTA and configurable thresholds."""

    def __init__(self, model_path, device=None, config=None):
        self.model_path = str(model_path)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # Safety: never use cuda/mps if not actually available
        if device in ("cuda", "cuda:0") and not torch.cuda.is_available():
            device = "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        self.device = device
        self.config = config or {}

        print(f"Loading detector: {model_path} on {self.device}")
        self.model = YOLO(self.model_path)

        # Defaults
        self.conf = self.config.get("conf", 0.10)
        self.iou = self.config.get("iou", 0.3)
        self.imgsz = self.config.get("imgsz", 640)
        self.augment = self.config.get("augment", True)
        self.max_det = self.config.get("max_det", 300)
        self.verbose = self.config.get("verbose", False)

    def detect(self, image_path):
        """Run detection on a single image path or numpy array."""
        results = self.model(
            image_path,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            augment=self.augment,
            max_det=self.max_det,
            verbose=self.verbose,
            device=self.device,
        )
        return results

    def parse_results(self, results, img_w, img_h):
        """Extract normalized detection results from YOLO output."""
        detections = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy.astype(int)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0]) if box.cls is not None else -1

            # Clamp to image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(conf, 4),
                "class_id": cls_id,
            })

        return detections

    def set_config(self, **kwargs):
        """Update detection config on the fly."""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
