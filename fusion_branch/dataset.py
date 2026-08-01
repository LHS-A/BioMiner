"""Paired image, segmentation-mask, and clinical-narrative dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

from text_branch.clinical_narrative import load_text_samples


class FusionDataset(Dataset):
    def __init__(self, json_path: str | Path, image_root: str | Path, tokenizer: Any, image_size: int = 384, max_text_length: int = 512):
        self.samples = load_text_samples(json_path)
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.image_transform = v2.Compose([
            v2.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.339] * 3, std=[0.138] * 3),
        ])
        self.mask_transform = v2.Compose([
            v2.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
            v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        for key in ("image", "nerve_mask", "cell_mask"):
            if key not in sample:
                raise ValueError(f"Fusion sample {index} is missing the {key} path.")
        image = self.image_transform(Image.open(self.image_root / sample["image"]).convert("RGB"))
        nerve = self.mask_transform(Image.open(self.image_root / sample["nerve_mask"]).convert("L"))
        cell = self.mask_transform(Image.open(self.image_root / sample["cell_mask"]).convert("L"))
        encoded = self.tokenizer(sample["input"], max_length=self.max_text_length, padding="max_length", truncation=True, return_tensors="pt")
        return {
            "image": image,
            "segmentation_mask": ((nerve + cell) > 0).float(),
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "nerve_label": torch.tensor(sample["nerve_label"], dtype=torch.long),
            "cell_label": torch.tensor(sample["cell_label"], dtype=torch.long),
            "sample_id": sample.get("id", str(index)),
        }
