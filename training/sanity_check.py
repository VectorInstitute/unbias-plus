"""Standalone sanity check inference script for Qwen3 UnBias models.

Usage:
  python training/sanity_check.py \\
    --model-path /path/to/Qwen3_8B_UnBias_SFT_Instruct/merged_16bit \\
    --article-file my_article.txt
"""

# ruff: noqa: E402, I001 — unsloth must be imported before torch; ruff isort
# would otherwise reorder it after the stdlib block.
from __future__ import annotations

from unsloth import FastLanguageModel

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, cast

import torch

from prompts import SFT_SYSTEM_PROMPT as SYSTEM_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_MAX_SEQ_LENGTH = 8192
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the sanity-check inference run."""
    parser = argparse.ArgumentParser(
        description="Sanity check a Qwen3 UnBias model on a single article.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to a merged_16bit model directory (or any HuggingFace model ID).",
    )
    parser.add_argument(
        "--article-file",
        type=Path,
        required=True,
        help="Path to a .txt file containing the article to analyze.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help="Maximum sequence length for model load.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature (low = near-deterministic).",
    )
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable Qwen <think> block at inference. Default False to match "
            "the training-time setting and avoid train/inference drift."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def load_model(model_path: Path, max_seq_length: int) -> tuple[Any, Any]:
    """Load the merged-16bit model in bf16 on a single GPU."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        str(model_path),
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_response(
    model: Any,
    tokenizer: Any,
    article: str,
    max_new_tokens: int,
    temperature: float,
    enable_thinking: bool,
) -> str:
    """Run a single forward generation pass and return decoded new tokens."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Analyze the following article for bias and return the result "
                "in the required JSON format.\n\n"
                f"ARTICLE:\n{article}"
            ),
        },
    ]

    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_tensors="pt",
        return_dict=True,
    )

    input_ids = tokenized["input_ids"].to("cuda")
    attention_mask = tokenized["attention_mask"].to("cuda")

    logger.info("Input tokens : %d", input_ids.shape[1])
    logger.info("Generating...")

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    new_tokens = outputs[0][input_ids.shape[1] :]
    return cast(str, tokenizer.decode(new_tokens, skip_special_tokens=False))


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def extract_json(text: str) -> str:
    """Extract a top-level JSON object from model output.

    Handles all three completion shapes the model can produce:
      - <think>...</think> block closed cleanly, JSON follows
      - <think> block hit max_tokens and was never closed
      - Output is pure JSON with no thinking block

    Strategy: skip past any closed </think>, find the first ``{``, and
    walk forward counting brace depth to locate the matching ``}``.
    """
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()

    start = text.find("{")
    if start == -1:
        return ""

    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    # Generation truncated mid-JSON — return everything from { for diagnosis.
    return text[start:]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report(response: str, json_str: str) -> None:
    """Pretty-print the raw model output and the parsed JSON summary."""
    logger.info("=" * 60)
    logger.info("RAW OUTPUT:")
    logger.info("=" * 60)
    logger.info(response)

    logger.info("")
    logger.info("=" * 60)
    logger.info("JSON VALIDATION:")
    logger.info("=" * 60)

    if not json_str:
        logger.error("No JSON found in output")
        logger.error("  The model likely hit max_new_tokens before finishing.")
        logger.error("  Try increasing --max-new-tokens or shortening the article.")
        return

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON: %s", e)
        logger.error("Extracted text attempted to parse:")
        logger.error(json_str[:500])
        logger.error("Tip: the model likely truncated mid-JSON.")
        logger.error("     Try increasing --max-new-tokens or shortening the article.")
        return

    logger.info("Valid JSON")
    logger.info("  binary_label   : %s", parsed.get("binary_label"))
    logger.info("  severity       : %s", parsed.get("severity"))
    logger.info("  bias_found     : %s", parsed.get("bias_found"))
    logger.info("  segments found : %d", len(parsed.get("biased_segments", [])))
    logger.info("")

    for i, seg in enumerate(parsed.get("biased_segments", []), 1):
        logger.info('  Segment %d: "%s"', i, seg.get("original"))
        logger.info("    → %s", seg.get("replacement"))
        logger.info(
            "    type: %s | severity: %s",
            seg.get("bias_type"),
            seg.get("severity"),
        )
        logger.info("    reason: %s", seg.get("reasoning"))
        logger.info("")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the sanity-check inference pipeline end to end."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    args = parse_args()
    article = args.article_file.read_text(encoding="utf-8").strip()

    logger.info("Loading model from: %s", args.model_path)
    model, tokenizer = load_model(args.model_path, args.max_seq_length)
    logger.info("Model loaded.")

    response = generate_response(
        model,
        tokenizer,
        article,
        args.max_new_tokens,
        args.temperature,
        args.thinking,
    )

    json_str = extract_json(response)
    report(response, json_str)


if __name__ == "__main__":
    main()
