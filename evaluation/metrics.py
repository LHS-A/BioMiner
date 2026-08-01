"""Evaluation metrics reported by the BioMiner paper."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix


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


def get_metrix(predictions: Iterable[int], labels: Iterable[int], num_classes: int = 4):
    """Compatibility metrics used by the retained text-adaptation training loop."""
    predictions = np.asarray(list(predictions), dtype=np.int64)
    labels = np.asarray(list(labels), dtype=np.int64)
    matrix = confusion_matrix(labels, predictions, labels=np.arange(num_classes)).astype(np.float64)
    tp = np.diag(matrix)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    tn = matrix.sum() - tp - fp - fn
    support = matrix.sum(axis=1)
    weights = support / support.sum() if support.sum() else np.zeros(num_classes)

    def safe_divide(numerator, denominator):
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

    level_acc = safe_divide(tp + tn, tp + tn + fp + fn)
    level_sensitivity = safe_divide(tp, tp + fn)
    level_specificity = safe_divide(tn, tn + fp)
    weighted = lambda values: float(np.sum(weights * values))
    return (
        [weighted(level_acc), level_acc.tolist()],
        [weighted(level_sensitivity), level_sensitivity.tolist()],
        [weighted(level_specificity), level_specificity.tolist()],
    )
