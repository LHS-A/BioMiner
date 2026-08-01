"""Evaluate a generatively pre-trained BioMiner text model with ROUGE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source-column", default="observation")
    parser.add_argument("--target-column", default="forecast")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-source-length", type=int, default=1024)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device).eval()
    raw = load_dataset("json", data_files={"test": str(args.data)})["test"]

    def tokenize(batch):
        encoded = tokenizer(batch[args.source_column], max_length=args.max_source_length, truncation=True)
        encoded["labels"] = tokenizer(text_target=batch[args.target_column], max_length=args.max_target_length, truncation=True)["input_ids"]
        return encoded

    dataset = raw.map(tokenize, batched=True, remove_columns=raw.column_names)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=DataCollatorForSeq2Seq(tokenizer, model=model))
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {name: [] for name in ("rouge1", "rouge2", "rougeL")}
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            generated = model.generate(**batch, max_length=args.max_target_length, num_beams=args.num_beams)
            references = np.where(labels.numpy() != -100, labels.numpy(), tokenizer.pad_token_id)
            predictions = tokenizer.batch_decode(generated.cpu(), skip_special_tokens=True)
            targets = tokenizer.batch_decode(references, skip_special_tokens=True)
            for prediction, target in zip(predictions, targets):
                result = scorer.score(target.strip(), prediction.strip())
                for name in scores:
                    scores[name].append(result[name].fmeasure)
    metrics = {name: float(np.mean(values)) for name, values in scores.items()}
    rendered = json.dumps(metrics, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
