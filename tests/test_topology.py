from __future__ import annotations

import numpy as np
import torch
import unittest

from biominer.topology import TopologyCorruptor, non_simple_points, topology_reconstruction_loss


def _cross(offset: int = 0):
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[32 + offset, 8:56] = 1
    mask[8:56, 32 + offset] = 1
    return mask


class TopologyTests(unittest.TestCase):
    def test_non_simple_point_extraction_finds_junction(self):
        critical = non_simple_points(_cross())
        self.assertTrue(critical[32, 32])
        self.assertFalse(critical[0, 0])

    def test_corruptor_matches_paper_configuration_and_is_repeatable(self):
        image = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
        first = TopologyCorruptor(seed=7)(image, _cross(-4), _cross(4))
        second = TopologyCorruptor(seed=7)(image, _cross(-4), _cross(4))
        self.assertEqual(first["input"].shape, (3, 64, 64))
        self.assertEqual(first["labels"].shape, (2, 64, 64))
        self.assertTrue(first["image_corruption_mask"].any())
        self.assertTrue(first["label_corruption_mask"].any())
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])

    def test_reconstruction_loss_is_zero_for_exact_targets(self):
        image = torch.rand(2, 1, 8, 8)
        labels = torch.rand(2, 2, 8, 8)
        prediction = torch.cat([image, labels], dim=1)
        mask = torch.ones(2, 1, 8, 8)
        loss, components = topology_reconstruction_loss(prediction, image, labels, mask, mask)
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(components["l_tplc"].item(), 0.0)
        self.assertEqual(components["l_tgic"].item(), 0.0)
