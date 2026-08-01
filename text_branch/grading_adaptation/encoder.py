"""Text-encoder adapter for the semantics-informed grading checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import T5EncoderModel


class TaskAdaptedTextEncoder(nn.Module):
    """Expose task-adapted T5 token representations to the fusion module."""

    def __init__(self, model_path: str | Path, checkpoint: str | Path | None = None) -> None:
        super().__init__()
        checkpoint_path = Path(checkpoint) if checkpoint is not None else None
        encoder_source = checkpoint_path if checkpoint_path is not None and checkpoint_path.is_dir() else model_path
        self.encoder = T5EncoderModel.from_pretrained(str(encoder_source))
        if checkpoint_path is not None and checkpoint_path.is_file():
            self.load_task_checkpoint(checkpoint_path)

    @property
    def output_dim(self) -> int:
        return int(self.encoder.config.d_model)

    def load_task_checkpoint(self, checkpoint: str | Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload)
        current = self.encoder.state_dict()
        selected = {}
        prefixes = ("encoder.encoder.", "model.t5.", "t5.", "encoder.")
        for key, value in state.items():
            candidates = [key]
            candidates.extend(key[len(prefix) :] for prefix in prefixes if key.startswith(prefix))
            for candidate in candidates:
                if candidate in current and current[candidate].shape == value.shape:
                    selected[candidate] = value
                    break
        if not selected:
            raise ValueError("No T5 encoder tensors matched the task-adapted checkpoint.")
        self.encoder.load_state_dict(selected, strict=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask)
