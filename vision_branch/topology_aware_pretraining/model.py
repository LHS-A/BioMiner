"""Reconstruction network and the L_TPLC/L_TGIC objectives."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from vision_branch.backbone import ResNet50Encoder


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
    """Reconstruct an image and its nerve/cell labels from topology-guided corruptions."""

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


def _masked_squared_l2(error: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = mask.expand_as(error)
    return (error.square() * expanded_mask).sum() / expanded_mask.sum().clamp_min(1.0)


def topology_reconstruction_loss(
    prediction: torch.Tensor,
    image: torch.Tensor,
    labels: torch.Tensor,
    image_corruption_mask: torch.Tensor,
    label_corruption_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the topology-preserving label and topology-guided image losses."""
    if prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError("prediction must have shape [batch, 3, height, width].")
    reconstructed_image, reconstructed_labels = prediction[:, :1], prediction[:, 1:]
    l_tplc = _masked_squared_l2(reconstructed_labels - labels, label_corruption_mask)
    l_tgic = _masked_squared_l2(reconstructed_image - image, image_corruption_mask)
    l_tgic = l_tgic + _masked_squared_l2(reconstructed_image - image, label_corruption_mask)
    return l_tplc + l_tgic, {"l_tplc": l_tplc, "l_tgic": l_tgic}
