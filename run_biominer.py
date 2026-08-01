"""Unified command-line entry point for the three BioMiner method branches."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


COMMANDS = {
    "vision-pretrain": "vision_branch.topology_aware_pretraining.train",
    "vision-adapt": "vision_branch.grading_adaptation.train",
    "tokenizer-prepare": "text_branch.numerical_tokenizer.prepare_corpus",
    "tokenizer-train": "text_branch.numerical_tokenizer.train",
    "tokenizer-evaluate": "text_branch.numerical_tokenizer.evaluate",
    "text-pretrain": "text_branch.generative_pretraining.train",
    "text-pretrain-evaluate": "text_branch.generative_pretraining.evaluate",
    "text-adapt": "text_branch.grading_adaptation.train",
    "text-evaluate": "text_branch.grading_adaptation.evaluate",
    "text-predict": "text_branch.grading_adaptation.predict",
    "fusion-train": "fusion_branch.train",
    "evaluate-predictions": "evaluation.evaluate_predictions",
    "validate-text": "text_branch.validate_input",
}


def run_stage(command: str, arguments: list[str]) -> None:
    module = COMMANDS[command]
    previous_argv = sys.argv
    try:
        sys.argv = [module, *arguments]
        runpy.run_module(module, run_name="__main__")
    finally:
        sys.argv = previous_argv


def run_pipeline(arguments: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Run several BioMiner stages sequentially from JSON.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(arguments)
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    stages = payload.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("A pipeline configuration requires a non-empty 'stages' list.")
    for index, stage in enumerate(stages, start=1):
        command = stage.get("command")
        stage_arguments = stage.get("arguments", [])
        if command not in COMMANDS or not isinstance(stage_arguments, list):
            raise ValueError(f"Invalid pipeline stage {index}: {stage}.")
        print(f"\n=== Stage {index}/{len(stages)}: {command} ===")
        run_stage(command, [str(value) for value in stage_arguments])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Run `python run_biominer.py COMMAND --help` for stage-specific options.",
    )
    parser.add_argument("command", choices=[*COMMANDS, "pipeline"])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "pipeline":
        run_pipeline(args.arguments)
    else:
        run_stage(args.command, args.arguments)


if __name__ == "__main__":
    main()
