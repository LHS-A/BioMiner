"""Evaluate saved dual-task probabilities with the paper-reported metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.metrics import grading_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    records = json.loads(args.predictions.read_text(encoding="utf-8"))
    result = {}
    for task in ("nerve", "cell"):
        labels = [record[f"{task}_label"] for record in records]
        probabilities = np.asarray([record[f"{task}_probabilities"] for record in records])
        result[task] = grading_metrics(labels, probabilities)
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
