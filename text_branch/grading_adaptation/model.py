"""T5 encoder with a shared projection and two four-level grading heads."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch import nn

from .encoder import TaskAdaptedTextEncoder


class DualTaskTextClassifier(nn.Module):
    def __init__(
        self,
        base_model: str | Path,
        generative_checkpoint: str | Path | None = None,
        feature_dim: int = 256,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = TaskAdaptedTextEncoder(base_model, generative_checkpoint)
        self.projection = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.nerve_classifier = nn.Linear(feature_dim, num_classes)
        self.cell_classifier = nn.Linear(feature_dim, num_classes)

    def extract_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        sequence = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(sequence.dtype)
        pooled = (sequence * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.projection(pooled)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(input_ids, attention_mask)
        return self.nerve_classifier(features), self.cell_classifier(features)

    @classmethod
    def from_checkpoint(
        cls, checkpoint: str | Path, base_model: str | Path, feature_dim: int = 256, num_classes: int = 4
    ) -> "DualTaskTextClassifier":
        model = cls(base_model=base_model, feature_dim=feature_dim, num_classes=num_classes)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload.get("state_dict", payload), strict=True)
        return model
