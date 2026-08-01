"""Image datasets used by topology pre-training and visual grading adaptation."""

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
        raise ValueError("An image manifest must be a non-empty JSON list.")
    return records


def _grayscale(path: Path, size: int, binary: bool = False) -> np.ndarray:
    image = Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST if binary else Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return (array > 0.5).astype(np.float32) if binary else array


class TopologyPretrainingDataset(Dataset):
    """Return corrupted image/label inputs and their reconstruction targets."""

    def __init__(
        self,
        manifest: str | Path,
        root: str | Path,
        corruptor: TopologyCorruptor,
        image_size: int = 384,
    ) -> None:
        self.records = _load_manifest(manifest)
        self.root = Path(root)
        self.corruptor = corruptor
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        image = _grayscale(self.root / record["image"], self.image_size)
        nerve = _grayscale(self.root / record["nerve_mask"], self.image_size, binary=True)
        cell = _grayscale(self.root / record["cell_mask"], self.image_size, binary=True)
        return {key: torch.from_numpy(value) for key, value in self.corruptor(image, nerve, cell).items()}


class VisionGradingDataset(Dataset):
    """Load original CCM images and simultaneous four-level grading labels."""

    def __init__(self, manifest: str | Path, root: str | Path, image_size: int = 384) -> None:
        self.records = _load_manifest(manifest)
        self.root = Path(root)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        image = _grayscale(self.root / record["image"], self.image_size)
        image_tensor = torch.from_numpy(image).repeat(3, 1, 1)
        return {
            "image": (image_tensor - 0.339) / 0.138,
            "nerve_label": torch.tensor(int(record["nerve_label"]), dtype=torch.long),
            "cell_label": torch.tensor(int(record["cell_label"]), dtype=torch.long),
        }
