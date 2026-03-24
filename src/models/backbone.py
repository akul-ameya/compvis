"""ImageNet-pretrained backbones for Tiny ImageNet and CIFAR-100 (224×224)."""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torchvision import models


def build_backbone(arch: str, num_classes: int) -> nn.Module:
    arch = arch.lower().replace("-", "_")
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        m._arch_name = "resnet18"  # type: ignore
        return m
    if arch in ("mobilenet_v3_small", "mobilenetv3_small"):
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        m._arch_name = "mobilenet_v3_small"  # type: ignore
        return m
    if arch in ("mobilenet_v3", "mobilenet_v3_large", "mobilenetv3_large"):
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
        m._arch_name = "mobilenet_v3_large"  # type: ignore
        return m
    raise ValueError(f"Unknown backbone: {arch}")


def forward_features(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (logits, penultimate) for supported architectures."""
    name = getattr(model, "_arch_name", "")
    if name == "resnet18" or isinstance(model.fc, nn.Linear) and hasattr(model, "layer4"):
        z = model.conv1(x)
        z = model.bn1(z)
        z = model.relu(z)
        z = model.maxpool(z)
        z = model.layer1(z)
        z = model.layer2(z)
        z = model.layer3(z)
        z = model.layer4(z)
        z = model.avgpool(z)
        feat = torch.flatten(z, 1)
        logits = model.fc(feat)
        return logits, feat
    if "mobilenet" in name or hasattr(model, "features") and hasattr(model, "classifier"):
        feat = model.features(x)
        feat = nn.functional.adaptive_avg_pool2d(feat, 1).flatten(1)
        logits = model.classifier(feat)
        return logits, feat
    logits = model(x)
    return logits, logits


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
