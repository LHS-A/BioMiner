"""Tokenized clinical narratives for dual-biomarker grading adaptation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import Dataset

from text_branch.clinical_narrative import load_text_samples


class TextGradingDataset(Dataset):
    def __init__(self, json_path: str | Path, tokenizer: Any, max_length: int = 512, require_labels: bool = True):
        self.samples = load_text_samples(json_path, require_labels=require_labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.require_labels = require_labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        encoded = self.tokenizer(
            sample["input"], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        result = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "sample_id": sample.get("id", sample.get("name", str(index))),
        }
        if self.require_labels:
            result.update({
                "nerve_label": torch.tensor(sample["nerve_label"], dtype=torch.long),
                "cell_label": torch.tensor(sample["cell_label"], dtype=torch.long),
            })
        return result
