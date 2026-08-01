"""Fine-tune the semantically pre-trained T5 encoder for both four-level grading tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .dataset import TextGradingDataset
from .model import DualTaskTextClassifier


@torch.no_grad()
def validate(model, loader, device) -> tuple[float, float]:
    model.eval()
    nerve_true, nerve_pred, cell_true, cell_pred = [], [], [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        nerve_logits, cell_logits = model(input_ids, attention_mask)
        nerve_true.extend(batch["nerve_label"].numpy())
        cell_true.extend(batch["cell_label"].numpy())
        nerve_pred.extend(nerve_logits.argmax(1).cpu().numpy())
        cell_pred.extend(cell_logits.argmax(1).cpu().numpy())
    return accuracy_score(nerve_true, nerve_pred), accuracy_score(cell_true, cell_pred)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--val-json", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--generative-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/text_branch/grading_adaptation/best.pt"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer or args.base_model), use_fast=False)
    train_dataset = TextGradingDataset(args.train_json, tokenizer, args.max_length)
    val_dataset = TextGradingDataset(args.val_json, tokenizer, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    model = DualTaskTextClassifier(
        args.base_model, args.generative_checkpoint, feature_dim=args.feature_dim
    ).to(device)
    if args.freeze_encoder:
        model.encoder.requires_grad_(False)
    optimizer = torch.optim.AdamW(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=args.learning_rate)
    best_score = -1.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            nerve_label = batch["nerve_label"].to(device)
            cell_label = batch["cell_label"].to(device)
            nerve_logits, cell_logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(nerve_logits, nerve_label) + F.cross_entropy(cell_logits, cell_label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
        nerve_acc, cell_acc = validate(model, val_loader, device)
        mean_acc = (nerve_acc + cell_acc) / 2
        print(
            f"epoch={epoch + 1} loss={running_loss / len(train_loader):.6f} "
            f"nerve_acc={nerve_acc:.4f} cell_acc={cell_acc:.4f}"
        )
        if mean_acc > best_score:
            best_score = mean_acc
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "feature_dim": args.feature_dim, "best_mean_acc": best_score}, args.output)


if __name__ == "__main__":
    main()
