"""Topology-aware visual pre-training and dual-biomarker grading models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50


class ResNet50Encoder(nn.Module):
    """ResNet-50 feature extractor shared by visual adaptation and fusion."""

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


class _DecoderBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels + skip_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.layers(torch.cat([value, skip], dim=1))


class TopologyAutoencoder(nn.Module):
    """Three-channel ResNet-50 U-Net for image and two-label reconstruction."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = ResNet50Encoder()
        self.decoder4 = _DecoderBlock(2048, 1024, 512)
        self.decoder3 = _DecoderBlock(512, 512, 256)
        self.decoder2 = _DecoderBlock(256, 256, 128)
        self.decoder1 = _DecoderBlock(128, 64, 64)
        self.output = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 3, 1))

    def forward(self, corrupted_input: torch.Tensor) -> torch.Tensor:
        stem, layer1, layer2, layer3, layer4 = self.encoder.forward_with_skips(corrupted_input)
        value = self.decoder4(layer4, layer3)
        value = self.decoder3(value, layer2)
        value = self.decoder2(value, layer1)
        value = self.decoder1(value, stem)
        value = F.interpolate(value, size=corrupted_input.shape[-2:], mode="bilinear", align_corners=False)
        return self.output(value)


def load_checkpoint(module: nn.Module, checkpoint: str | Path, prefixes: Tuple[str, ...] = ()) -> Dict[str, list[str]]:
    """Load a plain or Lightning checkpoint and report unmatched keys."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    cleaned = {}
    for key, value in state.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        cleaned[key] = value
    result = module.load_state_dict(cleaned, strict=False)
    return {"missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys)}


class DualTaskVisionClassifier(nn.Module):
    """Fine-tuning model with global average pooling and two four-level heads."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.encoder = ResNet50Encoder()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.nerve_classifier = nn.Linear(self.encoder.output_dim, num_classes)
        self.cell_classifier = nn.Linear(self.encoder.output_dim, num_classes)

    @classmethod
    def from_pretrained_autoencoder(cls, checkpoint: str | Path, num_classes: int = 4) -> "DualTaskVisionClassifier":
        model = cls(num_classes=num_classes)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload)
        encoder_state = {}
        for key, value in state.items():
            marker = "encoder."
            if marker in key:
                encoder_state[key.split(marker, 1)[1]] = value
        if not encoder_state:
            raise ValueError("The checkpoint does not contain topology-autoencoder encoder weights.")
        model.encoder.load_state_dict(encoder_state, strict=True)
        return model

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.pool(self.extract_features(image)).flatten(1)
        return self.nerve_classifier(features), self.cell_classifier(features)
