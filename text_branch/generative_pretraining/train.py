"""Generatively pre-train T5 on the paper's clinical question-answering corpus."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, get_scheduler


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"dataset_path", "dataset_name", "model_name_or_path", "tokenizer_name", "checkpoint_path"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Missing configuration fields: {missing}.")
    return config


def prepare_datasets(config: dict, tokenizer):
    root = Path(config["dataset_path"])
    name = config["dataset_name"]
    files = {"train": str(root / f"{name}_train.json"), "validation": str(root / f"{name}_val.json")}
    raw = load_dataset("json", data_files=files, cache_dir=config.get("cache_dir"))
    source_column, target_column = config.get("source_column", "observation"), config.get("target_column", "forecast")
    padding = "max_length" if config.get("pad_to_max_length", False) else False

    def tokenize(batch):
        model_inputs = tokenizer(
            batch[source_column], max_length=config.get("max_source_length", 1024), padding=padding, truncation=True
        )
        labels = tokenizer(
            text_target=batch[target_column], max_length=config.get("max_target_length", 128), padding=padding, truncation=True
        )["input_ids"]
        if padding == "max_length":
            labels = [[token if token != tokenizer.pad_token_id else -100 for token in row] for row in labels]
        model_inputs["labels"] = labels
        return model_inputs

    columns = raw["train"].column_names
    return raw.map(tokenize, batched=True, remove_columns=columns, desc="Tokenizing clinical QA corpus")


@torch.no_grad()
def evaluate(model, loader, tokenizer, device, max_target_length: int, num_beams: int) -> dict:
    model.eval()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {name: [] for name in ("rouge1", "rouge2", "rougeL")}
    for batch in loader:
        labels = batch.pop("labels")
        batch = {key: value.to(device) for key, value in batch.items()}
        generated = model.generate(**batch, max_length=max_target_length, num_beams=num_beams)
        references = labels.numpy()
        references = np.where(references != -100, references, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(generated.cpu(), skip_special_tokens=True)
        targets = tokenizer.batch_decode(references, skip_special_tokens=True)
        for prediction, target in zip(predictions, targets):
            scores = scorer.score(target.strip(), prediction.strip())
            for name in totals:
                totals[name].append(scores[name].fmeasure)
    return {name: float(np.mean(values)) for name, values in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"], use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name_or_path"]).to(device)
    model.resize_token_embeddings(len(tokenizer))
    datasets = prepare_datasets(config, tokenizer)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
    train_loader = DataLoader(
        datasets["train"], shuffle=True, collate_fn=collator, batch_size=int(config.get("batch_size", 64))
    )
    val_loader = DataLoader(
        datasets["validation"], shuffle=False, collate_fn=collator, batch_size=int(config.get("eval_batch_size", 64))
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-4)), weight_decay=float(config.get("weight_decay", 0.01)))
    epochs = int(config.get("epochs", 200))
    total_steps = epochs * len(train_loader)
    scheduler = get_scheduler(
        config.get("scheduler", "cosine"), optimizer=optimizer,
        num_warmup_steps=int(config.get("warmup_steps", 500)), num_training_steps=total_steps,
    )
    output_root = Path(config["checkpoint_path"])
    best_dir, final_dir = output_root / "best_model", output_root / "final_model"
    best_rouge1 = -1.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Generative pre-training {epoch + 1}/{epochs}"):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
        metrics = evaluate(
            model, val_loader, tokenizer, device,
            int(config.get("max_target_length", 128)), int(config.get("num_beams", 1)),
        )
        print(f"epoch={epoch + 1} loss={running_loss / len(train_loader):.6f} rouge1={metrics['rouge1']:.4f}")
        if metrics["rouge1"] > best_rouge1:
            best_rouge1 = metrics["rouge1"]
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            (best_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
