"""Evaluate vocabulary composition, numerical integrity, coverage, and token efficiency."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


NUMBER = re.compile(r"(?<!\w)[+-]?(?:\d+\.\d+|\d+)(?!\w)")
MIXED = re.compile(r"(?=.*[A-Za-z])(?=.*\d)")


def evaluate_tokenizer(tokenizer_path: Path, corpus: Path, limit: int | None = None) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=False)
    lines = [line.strip() for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        lines = lines[:limit]
    lengths, exact, numerical_integrity = [], [], []
    for line in lines:
        token_ids = tokenizer.encode(line, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        lengths.append(len(token_ids))
        exact.append(" ".join(decoded.split()) == " ".join(line.split()))
        numerical_integrity.append(NUMBER.findall(decoded) == NUMBER.findall(line))
    vocabulary = tokenizer.get_vocab().keys()
    return {
        "vocab_size": len(tokenizer),
        "mixed_alphanumeric_tokens": sum(bool(MIXED.search(token)) for token in vocabulary),
        "coverage": float(np.mean(exact)),
        "numerical_integrity": float(np.mean(numerical_integrity)),
        "average_tokens_per_sentence": float(np.mean(lengths)),
        "num_sentences": len(lines),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = evaluate_tokenizer(args.tokenizer, args.corpus, args.limit)
    rendered = json.dumps(metrics, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
