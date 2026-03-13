from typing import Tuple

import torch
from torch import nn
from torchvision import models


class TinyImageNetModel(nn.Module):
    """ResNet-18 or MobileNetV3 adapted for 200-class Tiny ImageNet classification."""

    def __init__(self, arch: str = "resnet18", num_classes: int = 200) -> None:
        super().__init__()
        self.arch = arch
        if arch == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        elif arch == "mobilenet_v3":
            self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            self.backbone.classifier[3] = nn.Linear(self.backbone.classifier[3].in_features, num_classes)
        else:
            raise ValueError(f"Unknown architecture: {arch}")

    def forward(self, x: torch.Tensor, return_features: bool = False):
        if not return_features:
            return self.backbone(x)

        if self.arch == "resnet18":
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            x = self.backbone.avgpool(x)
            features = torch.flatten(x, 1)
            logits = self.backbone.fc(features)
        else:
            features = self.backbone.features(x)
            features = nn.functional.adaptive_avg_pool2d(features, 1).flatten(1)
            logits = self.backbone.classifier(features)
        return logits, features


# Backward compatibility alias
ResNet18TinyImageNet = TinyImageNetModel


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
