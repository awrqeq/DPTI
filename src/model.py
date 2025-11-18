from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int = 10) -> nn.Module:
    model = models.resnet18(weights=None)
    # Adapt for CIFAR-size inputs: smaller conv kernel and no maxpool.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = (pred == target).sum().item()
        return correct / target.size(0)
