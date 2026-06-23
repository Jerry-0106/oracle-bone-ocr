"""
ArcFace Recognition Model for Inference.

Self-contained ArcFace model definition matching the training architecture
in _training/train_arcface.py. Loads checkpoints produced by Phase 4 training.

Architecture:
    ConvNeXt-Tiny (backbone, classifier replaced with Identity)
    → BN + Linear(768 → 512) + BN (embedding)
    → ArcMarginProduct(512, num_classes, s, m) (head)

Inference: model(x, label=None) → cosine logits (no angular margin)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ArcMarginProduct(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss head.

    Args:
        in_features:   embedding dimension (e.g. 512)
        out_features:  number of classes (e.g. 3483)
        s:             scale factor (default 30.0)
        m:             angular margin in radians (default 0.5)
        easy_margin:   if True, use cos(theta) when cos(theta+m) would be non-monotonic
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.5, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        """
        Args:
            input: L2-normalized embeddings  [B, in_features]
            label: class indices [B] or None for inference

        Returns:
            If label=None: cosine similarity logits * scale (inference mode)
            If label given: ArcFace logits with angular margin (training mode)
        """
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        if label is None:
            # Inference: pure cosine similarity (no margin)
            return cosine * self.s

        # Training: apply angular margin
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        output = one_hot * phi + (1.0 - one_hot) * cosine
        output *= self.s
        return output


class ArcFaceModel(nn.Module):
    """
    Full ArcFace recognition model.

    ConvNeXt-Tiny backbone → embedding projection → ArcFace head.
    """

    def __init__(self, backbone, backbone_dim=768, embedding_dim=512,
                 num_classes=3483, s=30.0, m=0.5):
        super().__init__()
        self.backbone = backbone
        self.backbone_dim = backbone_dim
        self.embedding_dim = embedding_dim

        self.embedding = nn.Sequential(
            nn.BatchNorm1d(backbone_dim),
            nn.Linear(backbone_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        self.arcface = ArcMarginProduct(
            embedding_dim, num_classes, s=s, m=m
        )

    def forward(self, x, label=None):
        """
        Args:
            x:     images [B, C, H, W]
            label: class indices [B] or None for inference

        Returns:
            dict with 'logits', 'embedding', 'features'
            Inference: logits = cosine similarity * s (no margin)
        """
        features = self.backbone(x)  # [B, backbone_dim]
        embedding = self.embedding(features)  # [B, embedding_dim]
        logits = self.arcface(embedding, label)

        return {
            'logits': logits,
            'embedding': embedding,
            'features': features,
        }

    @torch.no_grad()
    def get_embedding(self, x):
        """Extract L2-normalized embedding for feature analysis."""
        features = self.backbone(x)
        embedding = self.embedding(features)
        return F.normalize(embedding)


def build_arcface_model(num_classes=3483, embedding_dim=512, s=30.0, m=0.5):
    """
    Build a fresh ArcFace model with ConvNeXt-Tiny backbone.

    Matches the training architecture exactly.

    Args:
        num_classes:    number of output classes (3483 for competition)
        embedding_dim:  embedding dimension (512)
        s:              ArcFace scale
        m:              ArcFace margin in radians

    Returns:
        ArcFaceModel ready for weight loading
    """
    # ConvNeXt-Tiny with classifier replaced by Identity
    backbone = models.convnext_tiny(weights=None)
    backbone.classifier[2] = nn.Identity()

    model = ArcFaceModel(
        backbone=backbone,
        backbone_dim=768,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        s=s,
        m=m,
    )
    return model


def build_arcface_model_small_hybrid(competition_ckpt_path, num_classes=3483,
                                      embedding_dim=512, s=30.0, m=0.5):
    """
    Build ConvNeXt-Small ArcFace model with hybrid initialization.

    Strategy:
      1. Create ConvNeXt-Small with ImageNet pretrained weights
      2. Load convnext_competition_best.pt (Tiny backbone, competition finetuned)
      3. Copy all name+shape matching layers from competition ckpt → Small
      4. New Stage3 blocks (features.5.9–26) keep ImageNet weights
      5. Replace classifier[2] with Identity
      6. Build ArcFaceModel on top

    Expected transfer: ~56.3% of backbone params from competition checkpoint.

    Args:
        competition_ckpt_path: path to convnext_competition_best.pt
        num_classes:           number of output classes
        embedding_dim:         embedding dimension
        s:                     ArcFace scale
        m:                     ArcFace margin

    Returns:
        ArcFaceModel with hybrid-initialized backbone, plus transfer stats dict
    """
    import torch.nn as nn

    # Step 1: Build Small with ImageNet weights (covers new S3 blocks)
    backbone = models.convnext_small(weights='IMAGENET1K_V1')

    # Step 2: Load competition Tiny checkpoint
    comp_ckpt = torch.load(competition_ckpt_path, map_location='cpu',
                           weights_only=False)
    comp_state = comp_ckpt.get('model_state_dict', comp_ckpt)

    # Step 3: Transfer matching layers
    bb_state = backbone.state_dict()
    transferred = 0
    transferred_params = 0
    skipped = 0

    for k, v in comp_state.items():
        if k in bb_state and bb_state[k].shape == v.shape:
            bb_state[k].copy_(v)
            transferred += 1
            transferred_params += v.numel()
        else:
            skipped += 1

    backbone.load_state_dict(bb_state)

    # Step 4: Replace classifier with Identity
    backbone.classifier[2] = nn.Identity()

    # Step 5: Build ArcFace model
    model = ArcFaceModel(
        backbone=backbone,
        backbone_dim=768,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        s=s,
        m=m,
    )

    # Stats
    total_backbone = sum(p.numel() for p in backbone.parameters())
    stats = {
        'transferred_layers': transferred,
        'skipped_layers': skipped,
        'transferred_params': transferred_params,
        'total_backbone_params': total_backbone,
        'transfer_ratio': round(transferred_params / total_backbone * 100, 1),
        'competition_epoch': comp_ckpt.get('epoch', '?'),
        'competition_top1': comp_ckpt.get('best_top1', 0.0),
    }

    return model, stats


def load_arcface_checkpoint(model, checkpoint_path, device='cpu'):
    """
    Load ArcFace checkpoint, supporting both direct model state_dict
    and full training checkpoint dict.

    Args:
        model:           ArcFaceModel instance (already built)
        checkpoint_path: path to .pt checkpoint
        device:          torch device

    Returns:
        model with loaded weights, and checkpoint metadata dict
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in ckpt:
        # Full training checkpoint
        model.load_state_dict(ckpt['model_state_dict'])
        metadata = {
            'epoch': ckpt.get('epoch', '?'),
            'best_top1': ckpt.get('best_top1', 0.0),
            'best_top5': ckpt.get('best_top5', 0.0),
            'num_classes': ckpt.get('num_classes', 3483),
            'embedding_dim': ckpt.get('embedding_dim', 512),
            'arc_s': ckpt.get('arc_s', 30.0),
            'arc_m': ckpt.get('arc_m', 0.5),
        }
    else:
        # Direct state_dict
        model.load_state_dict(ckpt)
        metadata = {'epoch': '?', 'best_top1': 0.0}

    return model, metadata
