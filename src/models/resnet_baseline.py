"""Backward-compatible wrapper; prefer src.models.backbone.build_backbone."""
from typing import Tuple

import torch.nn as nn

from src.models.backbone import build_backbone, count_parameters, forward_features


class TinyImageNetModel(nn.Module):
    """Delegates to build_backbone(arch, num_classes)."""

    def __init__(self, arch: str = "resnet18", num_classes: int = 200) -> None:
        super().__init__()
        if arch == "mobilenet_v3":
            arch = "mobilenet_v3_small"
        self.inner = build_backbone(arch, num_classes)
        self.arch = getattr(self.inner, "_arch_name", arch)

    def forward(self, x, return_features: bool = False):
        if not return_features:
            return self.inner(x)
        logits, feat = forward_features(self.inner, x)
        return logits, feat


ResNet18TinyImageNet = TinyImageNetModel

__all__ = ["TinyImageNetModel", "ResNet18TinyImageNet", "count_parameters", "build_backbone", "forward_features"]
