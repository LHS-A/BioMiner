"""Train BioMiner's fusion module while keeping both adapted encoders frozen."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from fusion_branch.dataset import FusionDataset
from fusion_branch.model import BioMinerFusion
from text_branch.grading_adaptation.encoder import TaskAdaptedTextEncoder
from vision_branch.grading_adaptation.model import DualTaskVisionClassifier, load_vision_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--vision-checkpoint", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--text-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/fusion.pt"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer or args.text_model))
    dataset = FusionDataset(args.manifest, args.image_root, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    vision_model = DualTaskVisionClassifier()
    load_vision_checkpoint(vision_model, args.vision_checkpoint)
    text_encoder = TaskAdaptedTextEncoder(args.text_model, args.text_checkpoint)
    model = BioMinerFusion(
        visual_encoder=vision_model.encoder,
        text_encoder=text_encoder,
        visual_dim=vision_model.encoder.output_dim,
        text_dim=text_encoder.output_dim,
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in loader:
            tensors = {
                name: batch[name].to(device, non_blocking=True)
                for name in ("image", "segmentation_mask", "input_ids", "attention_mask", "nerve_label", "cell_label")
            }
            nerve_logits, cell_logits = model(
                tensors["image"], tensors["segmentation_mask"], tensors["input_ids"], tensors["attention_mask"]
            )
            loss = F.cross_entropy(nerve_logits, tensors["nerve_label"])
            loss = loss + F.cross_entropy(cell_logits, tensors["cell_label"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"epoch={epoch + 1} loss={running / len(loader):.6f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "paper_config": vars(args)}, args.output)


if __name__ == "__main__":
    main()
