"""Fine-tune the topology-aware encoder for simultaneous four-level grading."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from vision_branch.grading_adaptation.dataset import VisionGradingDataset
from vision_branch.grading_adaptation.model import DualTaskVisionClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/vision_grader.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualTaskVisionClassifier.from_pretrained_autoencoder(args.pretrained).to(device)
    loader = DataLoader(
        VisionGradingDataset(args.manifest, args.root),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            nerve = batch["nerve_label"].to(device, non_blocking=True)
            cell = batch["cell_label"].to(device, non_blocking=True)
            nerve_logits, cell_logits = model(image)
            loss = F.cross_entropy(nerve_logits, nerve) + F.cross_entropy(cell_logits, cell)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"epoch={epoch + 1} loss={running / len(loader):.6f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "paper_config": vars(args)}, args.output)


if __name__ == "__main__":
    main()
