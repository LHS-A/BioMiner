from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from fusion_branch.model import BioMinerFusion


class DummyVisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Conv2d(3, 8, 1)

    def forward(self, image):
        return self.layer(torch.nn.functional.adaptive_avg_pool2d(image, (4, 4)))


class DummyTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 6)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class FusionTests(unittest.TestCase):
    def test_fusion_shape_and_encoder_freezing(self):
        model = BioMinerFusion(
            DummyVisualEncoder(), DummyTextEncoder(), visual_dim=8, text_dim=6, fusion_dim=16, num_queries=4, num_heads=4
        )
        model.train()
        self.assertFalse(model.visual_encoder.training)
        self.assertFalse(model.text_encoder.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.visual_encoder.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.text_encoder.parameters()))
        nerve, cell = model(
            torch.rand(2, 3, 16, 16),
            torch.ones(2, 1, 16, 16),
            torch.randint(0, 32, (2, 7)),
            torch.ones(2, 7, dtype=torch.long),
        )
        self.assertEqual(nerve.shape, (2, 4))
        self.assertEqual(cell.shape, (2, 4))
