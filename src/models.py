"""
Recognition Models for Oracle Bone Character Classification.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model(model_name="efficientnet_b0", num_classes=1588, pretrained=True):
    """Build a classification model.

    Supported models:
    - efficientnet_b0, efficientnet_b1, efficientnet_b2
    - efficientnet_v2_s, efficientnet_v2_m
    - resnet18, resnet34, resnet50
    - convnext_tiny, convnext_small
    """
    model_name = model_name.lower()

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "efficientnet_b1":
        model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "efficientnet_b2":
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "efficientnet_v2_m":
        model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    elif model_name == "convnext_small":
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def count_parameters(model):
    """Return number of trainable parameters in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
