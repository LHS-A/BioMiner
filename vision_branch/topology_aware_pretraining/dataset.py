"""Dataset for topology-aware reconstruction pre-training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .topology import TopologyCorruptor


def _load_manifest(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError("The image manifest must be a non-empty JSON list.")
    return records


def _grayscale(path: Path, size: int, binary: bool = False) -> np.ndarray:
    resampling = Image.Resampling.NEAREST if binary else Image.Resampling.BILINEAR
    array = np.asarray(Image.open(path).convert("L").resize((size, size), resampling), dtype=np.float32) / 255.0
    return (array > 0.5).astype(np.float32) if binary else array


class TopologyPretrainingDataset(Dataset):
    def __init__(self, manifest: str | Path, root: str | Path, corruptor: TopologyCorruptor, image_size: int = 384):
        self.records = _load_manifest(manifest)
        self.root = Path(root)
        self.corruptor = corruptor
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        corrupted = self.corruptor(
            _grayscale(self.root / record["image"], self.image_size),
            _grayscale(self.root / record["nerve_mask"], self.image_size, binary=True),
            _grayscale(self.root / record["cell_mask"], self.image_size, binary=True),
        )
        return {key: torch.from_numpy(value) for key, value in corrupted.items()}
