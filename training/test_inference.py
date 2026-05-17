"""Minimal inference smoke test for a fine-tuned UnBias-Plus model.

Usage:
python training/test_inference.py \
  --model-path vector-institute/Qwen3-4B-UnBias-Plus-SFT

 python training/test_inference.py \
  --model-path vector-institute/Qwen3-4B-UnBias-Plus-SFT \
  --load-in-4bit
  """

# ruff: noqa: E402, I001
from __future__ import annotations

import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from unbias_plus.pipeline import UnBiasPlus  # noqa: E402

SAMPLE_ARTICLE = """
The radical left's reckless agenda is destroying our great nation. These
out-of-touch elites refuse to listen to hardworking Americans, instead
pushing dangerous policies that threaten our way of life. Patriots must
stand up against this tyrannical overreach before it is too late.
""".strip()


def main() -> None:
    """Run a single inference call and pretty-print the JSON output."""
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--load-in-4bit", action="store_true")
    args = p.parse_args()

    print(f"Loading model: {args.model_path}")
    pipe = UnBiasPlus(
        model_name_or_path=str(args.model_path),
        load_in_4bit=args.load_in_4bit,
    )

    print("\n--- INPUT ---")
    print(SAMPLE_ARTICLE)

    print("\n--- OUTPUT ---")
    result = pipe.analyze(SAMPLE_ARTICLE)
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
