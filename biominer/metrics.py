"""Evaluation metrics reported by the BioMiner paper."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def grading_metrics(
    labels: Iterable[int], probabilities: np.ndarray, num_classes: int = 4
) -> Dict[str, object]:
    """Compute overall accuracy, within-level accuracy, and one-vs-rest AUC."""
    labels = np.asarray(list(labels), dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(labels), num_classes):
        raise ValueError(f"probabilities must have shape ({len(labels)}, {num_classes}).")
    predictions = probabilities.argmax(axis=1)
    level_accuracy = []
    level_auc = []
    for level in range(num_classes):
        present = labels == level
        level_accuracy.append(float((predictions[present] == level).mean()) if present.any() else float("nan"))
        binary = present.astype(np.int64)
        level_auc.append(
            float(roc_auc_score(binary, probabilities[:, level])) if np.unique(binary).size == 2 else float("nan")
        )
    return {
        "overall_accuracy": float(accuracy_score(labels, predictions)),
        "level_accuracy": level_accuracy,
        "level_auc_ovr": level_auc,
        "macro_auc_ovr": float(np.nanmean(level_auc)),
    }
