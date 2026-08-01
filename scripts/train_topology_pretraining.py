"""Train the topology-aware visual autoencoder."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from biominer.datasets import TopologyPretrainingDataset
from biominer.topology import TopologyCorruptionConfig, TopologyCorruptor, topology_reconstruction_loss
from biominer.vision import TopologyAutoencoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/topology_autoencoder.pt"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corruptor = TopologyCorruptor(TopologyCorruptionConfig(noise_std=args.noise_std), seed=args.seed)
    dataset = TopologyPretrainingDataset(args.manifest, args.root, corruptor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    model = TopologyAutoencoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in loader:
            batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
            prediction = model(batch["input"])
            loss, components = topology_reconstruction_loss(
                prediction,
                batch["image"],
                batch["labels"],
                batch["image_corruption_mask"],
                batch["label_corruption_mask"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(
            f"epoch={epoch + 1} loss={running / len(loader):.6f} "
            f"l_tplc={components['l_tplc'].item():.6f} l_tgic={components['l_tgic'].item():.6f}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "paper_config": vars(args)}, args.output)


if __name__ == "__main__":
    main()
