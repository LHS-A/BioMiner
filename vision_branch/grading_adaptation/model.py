"""Visual grading model obtained after discarding the reconstruction decoder."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch import nn

from vision_branch.backbone import ResNet50Encoder


class DualTaskVisionClassifier(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.encoder = ResNet50Encoder()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.nerve_classifier = nn.Linear(self.encoder.output_dim, num_classes)
        self.cell_classifier = nn.Linear(self.encoder.output_dim, num_classes)

    @classmethod
    def from_pretrained_autoencoder(cls, checkpoint: str | Path, num_classes: int = 4) -> "DualTaskVisionClassifier":
        model = cls(num_classes)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload)
        encoder_state = {key.split("encoder.", 1)[1]: value for key, value in state.items() if "encoder." in key}
        if not encoder_state:
            raise ValueError("The checkpoint contains no topology-autoencoder encoder weights.")
        model.encoder.load_state_dict(encoder_state, strict=True)
        return model

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.pool(self.extract_features(image)).flatten(1)
        return self.nerve_classifier(features), self.cell_classifier(features)


def load_vision_checkpoint(model: DualTaskVisionClassifier, checkpoint: str | Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("state_dict", payload), strict=True)
