"""Evaluate the adapted text branch with paper-reported ACC and one-vs-rest AUC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from evaluation.metrics import grading_metrics
from .dataset import TextGradingDataset
from .model import DualTaskTextClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer or args.base_model), use_fast=False)
    dataset = TextGradingDataset(args.test_json, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = DualTaskTextClassifier.from_checkpoint(args.checkpoint, args.base_model, args.feature_dim).to(device).eval()
    records, labels = [], {"nerve": [], "cell": []}
    probabilities = {"nerve": [], "cell": []}
    with torch.no_grad():
        for batch in loader:
            nerve_logits, cell_logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            nerve_probs, cell_probs = nerve_logits.softmax(1).cpu().numpy(), cell_logits.softmax(1).cpu().numpy()
            labels["nerve"].extend(batch["nerve_label"].numpy())
            labels["cell"].extend(batch["cell_label"].numpy())
            probabilities["nerve"].extend(nerve_probs)
            probabilities["cell"].extend(cell_probs)
            for sample_id, nerve_prob, cell_prob in zip(batch["sample_id"], nerve_probs, cell_probs):
                records.append({"id": sample_id, "nerve_probabilities": nerve_prob.tolist(), "cell_probabilities": cell_prob.tolist()})
    metrics = {task: grading_metrics(labels[task], np.asarray(probabilities[task])) for task in ("nerve", "cell")}
    print(json.dumps(metrics, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"metrics": metrics, "predictions": records}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
