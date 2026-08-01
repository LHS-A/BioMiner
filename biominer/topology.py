"""Topology-aware corruption and reconstruction losses from the paper."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
import torch
from scipy import ndimage
from skimage.measure import label
from skimage.morphology import skeletonize


def _is_simple_point(patch: np.ndarray) -> bool:
    """Return whether toggling the center preserves foreground/background topology."""
    before = patch.astype(np.uint8, copy=True)
    after = before.copy()
    after[1, 1] = 1 - after[1, 1]
    foreground_before = label(before, connectivity=2).max()
    foreground_after = label(after, connectivity=2).max()
    background_before = label(1 - before, connectivity=1).max()
    background_after = label(1 - after, connectivity=1).max()
    return foreground_before == foreground_after and background_before == background_after


def _build_non_simple_lut() -> np.ndarray:
    lut = np.zeros(512, dtype=np.uint8)
    for code in range(512):
        bits = np.array([int(bit) for bit in f"{code:09b}"], dtype=np.uint8)
        patch = bits.reshape(3, 3)
        if patch[1, 1] and not _is_simple_point(patch):
            lut[code] = 1
    return lut


_NON_SIMPLE_LUT = _build_non_simple_lut()
_CODE_KERNEL = (2 ** np.arange(8, -1, -1)).reshape(3, 3)


def non_simple_points(binary_skeleton: np.ndarray) -> np.ndarray:
    """Extract topology-critical skeleton points with the paper's digital-topology test."""
    skeleton = (binary_skeleton > 0).astype(np.uint8)
    codes = ndimage.correlate(skeleton.astype(np.int64), _CODE_KERNEL, mode="constant", cval=0)
    return _NON_SIMPLE_LUT[codes].astype(bool)


def _trace_segment(skeleton: np.ndarray, seed: Tuple[int, int], length: int) -> Iterable[Tuple[int, int]]:
    """Trace a connected skeleton segment from one seed with breadth-first search."""
    height, width = skeleton.shape
    queue = deque([seed])
    visited = {seed}
    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    while queue and len(visited) <= length:
        y, x = queue.popleft()
        yield y, x
        for dy, dx in offsets:
            point = (y + dy, x + dx)
            py, px = point
            if 0 <= py < height and 0 <= px < width and skeleton[py, px] and point not in visited:
                visited.add(point)
                queue.append(point)
                if len(visited) >= length:
                    break


def _sample_seeds(points: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) == 0:
        raise ValueError("A skeleton contains no non-simple points; topology-guided corruption is undefined.")
    indices = rng.choice(len(points), size=count, replace=len(points) < count)
    return points[indices]


@dataclass(frozen=True)
class TopologyCorruptionConfig:
    num_seeds: int = 12
    segment_length: Tuple[int, int] = (60, 100)
    image_dilation: int = 7
    label_dilation: int = 11
    noise_std: float = 0.1

    def __post_init__(self) -> None:
        if self.num_seeds <= 0 or self.num_seeds % 2:
            raise ValueError("num_seeds must be a positive even number.")
        if self.segment_length[0] <= 0 or self.segment_length[0] > self.segment_length[1]:
            raise ValueError("segment_length must be a positive inclusive interval.")
        for value in (self.image_dilation, self.label_dilation):
            if value <= 0 or value % 2 == 0:
                raise ValueError("Dilation kernels must be positive odd integers.")


class TopologyCorruptor:
    """Generate the paired image- and label-domain corruptions described in Methods."""

    def __init__(self, config: TopologyCorruptionConfig | None = None, seed: int | None = None):
        self.config = config or TopologyCorruptionConfig()
        self.rng = np.random.default_rng(seed)

    def __call__(self, image: np.ndarray, nerve_mask: np.ndarray, cell_mask: np.ndarray) -> Dict[str, np.ndarray]:
        image = np.asarray(image, dtype=np.float32)
        if image.ndim != 2:
            raise ValueError("image must be a two-dimensional grayscale array.")
        masks = [(np.asarray(mask) > 0).astype(np.uint8) for mask in (nerve_mask, cell_mask)]
        if any(mask.shape != image.shape for mask in masks):
            raise ValueError("image, nerve_mask, and cell_mask must have identical shapes.")

        skeletons = [skeletonize(mask).astype(np.uint8) for mask in masks]
        critical = [non_simple_points(skeleton) for skeleton in skeletons]
        label_seed_mask = np.logical_or(*critical).astype(np.uint8)
        label_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.config.label_dilation, self.config.label_dilation)
        )
        label_corruption_mask = cv2.dilate(label_seed_mask, label_kernel).astype(bool)

        traced = np.zeros_like(label_seed_mask)
        seeds_per_structure = self.config.num_seeds // 2
        for skeleton, points in zip(skeletons, critical):
            seeds = _sample_seeds(np.argwhere(points), seeds_per_structure, self.rng)
            for seed in seeds:
                length = int(self.rng.integers(self.config.segment_length[0], self.config.segment_length[1] + 1))
                for y, x in _trace_segment(skeleton, tuple(seed), length):
                    traced[y, x] = 1
        image_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.config.image_dilation, self.config.image_dilation)
        )
        image_corruption_mask = cv2.dilate(traced, image_kernel).astype(bool)

        noise = self.rng.normal(0.0, self.config.noise_std, image.shape).astype(np.float32)
        corrupted_image = np.where(image_corruption_mask, noise, image).astype(np.float32)
        corrupted_labels = np.stack(
            [np.where(label_corruption_mask, 0, mask).astype(np.float32) for mask in masks]
        )
        return {
            "input": np.concatenate([corrupted_image[None], corrupted_labels], axis=0),
            "image": image[None],
            "labels": np.stack(masks).astype(np.float32),
            "image_corruption_mask": image_corruption_mask[None].astype(np.float32),
            "label_corruption_mask": label_corruption_mask[None].astype(np.float32),
        }


def _masked_squared_l2(error: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = mask.expand_as(error)
    denominator = expanded_mask.sum().clamp_min(1.0)
    return ((error.square()) * expanded_mask).sum() / denominator


def topology_reconstruction_loss(
    prediction: torch.Tensor,
    image: torch.Tensor,
    labels: torch.Tensor,
    image_corruption_mask: torch.Tensor,
    label_corruption_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute L_TPLC + L_TGIC exactly over the paper-defined masked regions."""
    if prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError("prediction must have shape [batch, 3, height, width].")
    reconstructed_image, reconstructed_labels = prediction[:, :1], prediction[:, 1:]
    l_tplc = _masked_squared_l2(reconstructed_labels - labels, label_corruption_mask)
    l_tgic = _masked_squared_l2(reconstructed_image - image, image_corruption_mask)
    l_tgic = l_tgic + _masked_squared_l2(reconstructed_image - image, label_corruption_mask)
    return l_tplc + l_tgic, {"l_tplc": l_tplc, "l_tgic": l_tgic}
