"""Shared ResNet-50 encoder used throughout the vision branch."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torchvision.models import resnet50


class ResNet50Encoder(nn.Module):
    """ResNet-50 without its pooling and ImageNet classification head."""

    output_dim = 2048

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet50(weights=None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward_with_skips(self, image: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        stem = self.stem(image)
        layer1 = self.layer1(self.pool(stem))
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return stem, layer1, layer2, layer3, layer4

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.forward_with_skips(image)[-1]
