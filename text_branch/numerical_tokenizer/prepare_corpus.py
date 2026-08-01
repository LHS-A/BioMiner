"""Convert clinical QA JSON files into one-line tokenizer training sentences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def _records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    if isinstance(payload, dict):
        for key in ("data", "samples", "records"):
            if isinstance(payload.get(key), list):
                return [record for record in payload[key] if isinstance(record, dict)]
    raise ValueError(f"Unsupported JSON structure in {path}.")


def _sentence(record: dict) -> str:
    observation = record.get("observation") or record.get("input")
    if not observation:
        context, question = record.get("context", ""), record.get("question", "")
        observation = f"{context} {question}".strip()
    answer = record.get("forecast") or record.get("answer") or record.get("output") or ""
    return " ".join(f"{observation} {answer}".split())


def prepare_corpus(inputs: Iterable[Path], output: Path) -> int:
    sentences = []
    for path in inputs:
        sentences.extend(filter(None, (_sentence(record) for record in _records(path))))
    if not sentences:
        raise ValueError("No tokenizer training sentences were found.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(sentences) + "\n", encoding="utf-8")
    print(f"Wrote {len(sentences)} sentences to {output}")
    return len(sentences)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/text_branch/tokenizer_corpus.txt"))
    args = parser.parse_args()
    prepare_corpus(args.inputs, args.output)


if __name__ == "__main__":
    main()
