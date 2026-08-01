"""Generate text-branch grades for unlabeled clinical-narrative JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .dataset import TextGradingDataset
from .model import DualTaskTextClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-dim", type=int, default=256)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer or args.base_model), use_fast=False)
    loader = DataLoader(TextGradingDataset(args.input_json, tokenizer, require_labels=False), batch_size=args.batch_size)
    model = DualTaskTextClassifier.from_checkpoint(args.checkpoint, args.base_model, args.feature_dim).to(device).eval()
    output = []
    with torch.no_grad():
        for batch in loader:
            nerve, cell = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            nerve_prob, cell_prob = nerve.softmax(1).cpu(), cell.softmax(1).cpu()
            for sample_id, np_, cp_ in zip(batch["sample_id"], nerve_prob, cell_prob):
                output.append({
                    "id": sample_id,
                    "nerve_grade": int(np_.argmax()), "cell_grade": int(cp_.argmax()),
                    "nerve_probabilities": np_.tolist(), "cell_probabilities": cp_.tolist(),
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
