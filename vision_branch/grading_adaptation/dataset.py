"""Dataset for topology-informed visual grading adaptation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class VisionGradingDataset(Dataset):
    def __init__(self, manifest: str | Path, root: str | Path, image_size: int = 384) -> None:
        with Path(manifest).open("r", encoding="utf-8") as handle:
            self.records: List[Dict[str, Any]] = json.load(handle)
        if not self.records:
            raise ValueError("The grading manifest must be a non-empty JSON list.")
        self.root = Path(root)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        image = Image.open(self.root / record["image"]).convert("L").resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).repeat(3, 1, 1)
        return {
            "image": (image_tensor - 0.339) / 0.138,
            "nerve_label": torch.tensor(int(record["nerve_label"]), dtype=torch.long),
            "cell_label": torch.tensor(int(record["cell_label"]), dtype=torch.long),
        }
