"""Bi-directional cross-modal alignment for joint biomarker grading."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class _CrossAttention(nn.Module):
    def __init__(self, dimension: int, heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(dimension, heads, batch_first=True)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output, _ = self.attention(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return output


class BioMinerFusion(nn.Module):
    """Paper-aligned feature fusion with frozen visual and text encoders."""

    def __init__(
        self,
        visual_encoder: nn.Module,
        text_encoder: nn.Module,
        visual_dim: int = 2048,
        text_dim: int = 768,
        fusion_dim: int = 512,
        num_queries: int = 12,
        num_heads: int = 8,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        if num_queries <= 0:
            raise ValueError("num_queries must be positive.")
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.visual_projection = nn.Linear(visual_dim, fusion_dim)
        self.text_projection = nn.Linear(text_dim, fusion_dim)
        self.queries = nn.Parameter(torch.empty(1, num_queries, fusion_dim))
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        self.visual_alignment = _CrossAttention(fusion_dim, num_heads)
        self.text_alignment = _CrossAttention(fusion_dim, num_heads)
        self.visual_calibration = _CrossAttention(fusion_dim, num_heads)
        self.text_calibration = _CrossAttention(fusion_dim, num_heads)
        self.visual_norm = nn.LayerNorm(fusion_dim)
        self.text_norm = nn.LayerNorm(fusion_dim)
        self.calibrated_visual_norm = nn.LayerNorm(fusion_dim)
        self.calibrated_text_norm = nn.LayerNorm(fusion_dim)
        self.nerve_classifier = nn.Linear(fusion_dim, num_classes)
        self.cell_classifier = nn.Linear(fusion_dim, num_classes)
        self.freeze_encoders()

    def freeze_encoders(self) -> None:
        for encoder in (self.visual_encoder, self.text_encoder):
            encoder.requires_grad_(False)
            encoder.eval()

    def train(self, mode: bool = True) -> "BioMinerFusion":
        super().train(mode)
        self.visual_encoder.eval()
        self.text_encoder.eval()
        return self

    @staticmethod
    def _masked_average(feature_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        resized_mask = torch.nn.functional.interpolate(mask.float(), size=feature_map.shape[-2:], mode="nearest")
        denominator = resized_mask.sum(dim=(-2, -1)).clamp_min(1.0)
        return (feature_map * resized_mask).sum(dim=(-2, -1)) / denominator

    def _visual_embeddings(self, image: torch.Tensor, segmentation_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feature_map = self.visual_encoder(image)
        local = self._masked_average(feature_map, segmentation_mask)
        global_representation = feature_map.mean(dim=(-2, -1))
        return self.visual_projection(torch.stack([local, global_representation], dim=1))

    def _text_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence = output.last_hidden_state if hasattr(output, "last_hidden_state") else output
        return self.text_projection(sequence)

    def extract_fused_features(
        self,
        image: torch.Tensor,
        segmentation_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        visual_embeddings = self._visual_embeddings(image, segmentation_mask)
        text_embeddings = self._text_embeddings(input_ids, attention_mask)
        queries = self.queries.expand(image.shape[0], -1, -1)
        aligned_visual = self.visual_norm(queries + self.visual_alignment(queries, visual_embeddings))
        text_padding_mask = ~attention_mask.bool()
        aligned_text = self.text_norm(
            queries + self.text_alignment(queries, text_embeddings, key_padding_mask=text_padding_mask)
        )
        calibrated_visual = self.calibrated_visual_norm(
            aligned_visual + self.visual_calibration(aligned_visual, aligned_text)
        )
        calibrated_text = self.calibrated_text_norm(
            aligned_text + self.text_calibration(aligned_text, aligned_visual)
        )
        return torch.cat([calibrated_visual, calibrated_text], dim=1).mean(dim=1)

    def forward(
        self,
        image: torch.Tensor,
        segmentation_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shared = self.extract_fused_features(image, segmentation_mask, input_ids, attention_mask)
        return self.nerve_classifier(shared), self.cell_classifier(shared)
