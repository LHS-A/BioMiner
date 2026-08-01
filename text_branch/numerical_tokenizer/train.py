"""Train the numerical-aware SentencePiece BPE tokenizer described in Methods."""

from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm
from transformers import T5Tokenizer


def train_tokenizer(corpus: Path, output_dir: Path, vocab_size: int = 1224, model_type: str = "bpe") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"biominer_numerical_{model_type}"
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        unk_id=3,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        control_symbols="[PAD],[UNK],[CLS],[SEP],[MASK]",
        model_type=model_type,
        train_extremely_large_corpus=True,
        split_by_number=False,
        character_coverage=1.0,
    )
    model_file = prefix.with_suffix(".model")
    tokenizer = T5Tokenizer(vocab_file=str(model_file))
    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved to {output_dir}")
    return model_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="One training sentence per line.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/text_branch/numerical_tokenizer"))
    parser.add_argument("--vocab-size", type=int, default=1224)
    parser.add_argument("--model-type", choices=("bpe", "unigram", "char", "word"), default="bpe")
    args = parser.parse_args()
    train_tokenizer(args.corpus, args.output_dir, args.vocab_size, args.model_type)


if __name__ == "__main__":
    main()
