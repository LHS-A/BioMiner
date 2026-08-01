"""Validate and render BioMiner text JSON without downloading a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from text_branch.clinical_narrative import load_text_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--output", type=Path, help="Optional normalized JSON output path.")
    parser.add_argument("--without-labels", action="store_true")
    args = parser.parse_args()
    samples = load_text_samples(args.json_path, require_labels=not args.without_labels)
    print(f"Validated {len(samples)} samples.")
    for sample in samples[:3]:
        print(f"[{sample.get('id', 'sample')}] {sample['input']}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Normalized JSON written to {args.output}.")


if __name__ == "__main__":
    main()
