from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int = 10, img_size: int = 32) -> nn.Module:
    """构建 ResNet-18，根据分辨率选择是否使用 CIFAR 风格 stem。"""
    model = models.resnet18(weights=None)
    if img_size <= 64:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_densenet121(num_classes: int = 10, cifar_like: bool = True) -> nn.Module:
    """
    构建 DenseNet121。
    - cifar_like=True 时，适配小分辨率（如 32x32, 64x64），修改 conv0 和 pool0
    - cifar_like=False 时，使用原版（适合 224x224）
    """
    model = models.densenet121(pretrained=False)
    if cifar_like:
        model.features.conv0 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.features.pool0 = nn.Identity()

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = (pred == target).sum().item()
        return correct / target.size(0)
