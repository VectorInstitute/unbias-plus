# ruff: noqa: E402, I001 — heavy ML imports are grouped after the docstring and
# stdlib imports to keep model loading lazy and the import order readable.
"""Standalone evaluation CLI for the SFT recipes (train_4 held-out split).

Deliberately independent of the installable ``unbias_plus`` package: that
toolkit is wired to the deployed schema, whereas these recipes speak the raw
``train_4`` schema (severity 0-10, Low/Medium/High segments, snake_case bias
types). This script embeds the recipe's own prompt and parses the model's JSON
directly.

Run from the ``training/`` directory::

    python -m recipes.eval --recipe bias_weighted --text "The stakes are high."
    python -m recipes.eval --recipe bias_weighted --file article.txt
    python -m recipes.eval --recipe bias_weighted --jsonl HELDOUT.jsonl --limit 20
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from recipes.config import load_recipe
from recipes.data import SOFT_DEBIAS_AUDIT_PATTERNS, load_jsonl
from recipes.prompts import build_messages


try:
    from json_repair import repair_json
except ImportError:  # pragma: no cover - optional dependency
    repair_json = None

_LAUNDERING_REGEXES = [re.compile(p, re.IGNORECASE) for p in SOFT_DEBIAS_AUDIT_PATTERNS]


class Model:
    """Thin wrapper around a merged HF causal-LM for structured bias analysis."""

    def __init__(
        self,
        model_path: str,
        prompt_id: str,
        max_seq_length: int = 8192,
        max_new_tokens: int = 4096,
        load_in_4bit: bool = False,
    ) -> None:
        """Load the tokenizer and model, optionally in 4-bit."""
        self.prompt_id = prompt_id
        self.max_seq_length = max_seq_length
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.model.eval()

    def generate(self, article: str) -> str:
        """Run the model on one article and return the raw generated string."""
        messages = build_messages(self.prompt_id, article)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def parse_json(raw: str) -> dict[str, Any] | None:
    """Parse the model output into a dict, repairing malformed JSON if possible."""
    text = raw.strip()
    start = text.find("{")
    if start > 0:
        text = text[start:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if repair_json is not None:
        try:
            repaired = repair_json(text)
            parsed = json.loads(repaired)
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def count_laundering(result: dict[str, Any]) -> int:
    """Count predicted replacements / rewrite spans that look like laundering."""
    hits = 0
    for seg in result.get("biased_segments", []) or []:
        replacement = str(seg.get("replacement", "") or "")
        if any(rx.search(replacement) for rx in _LAUNDERING_REGEXES):
            hits += 1
    unbiased = str(result.get("unbiased_text", "") or "")
    if any(rx.search(unbiased) for rx in _LAUNDERING_REGEXES):
        hits += 1
    return hits


def print_result(result: dict[str, Any], gold: dict[str, Any] | None) -> None:
    """Pretty-print a single predicted result, with gold comparison if present."""
    severity = result.get("severity", "?")
    segments = result.get("biased_segments", []) or []
    header = f"severity={severity} | segments={len(segments)}"
    if gold is not None:
        header += (
            f"  (gold severity={gold.get('severity', '?')} | "
            f"segments={len(gold.get('biased_segments', []) or [])})"
        )
    print(header)
    for i, seg in enumerate(segments, 1):
        print(
            f"  [{i}] ({seg.get('bias_type', '?')}/{seg.get('severity', '?')}) "
            f"{seg.get('original', '')!r} -> {seg.get('replacement', '')!r}"
        )
    launder = count_laundering(result)
    if launder:
        print(f"  ! laundering-pattern hits: {launder}")


def evaluate_dataset(model: Model, records: list[dict[str, Any]]) -> None:
    """Run the model over gold records and print per-row + aggregate metrics."""
    severity_abs_errors: list[float] = []
    segment_deltas: list[int] = []
    laundering_total = 0
    parse_failures = 0

    for idx, record in enumerate(records):
        article = record.get("article_text", "")
        raw = model.generate(article)
        result = parse_json(raw)
        print(f"\n--- row {idx} ---")
        if result is None:
            parse_failures += 1
            print("  <parse failure>")
            continue

        print_result(result, record)
        gold_severity = record.get("severity")
        pred_severity = result.get("severity")
        if isinstance(gold_severity, int) and isinstance(pred_severity, (int, float)):
            severity_abs_errors.append(abs(float(pred_severity) - gold_severity))
        gold_segments = record.get("biased_segments", []) or []
        pred_segments = result.get("biased_segments", []) or []
        segment_deltas.append(abs(len(pred_segments) - len(gold_segments)))
        laundering_total += count_laundering(result)

    print("\n=== Aggregate metrics ===")
    print(f"  Rows evaluated       : {len(records)}")
    print(f"  Parse failures       : {parse_failures}")
    if severity_abs_errors:
        mae = sum(severity_abs_errors) / len(severity_abs_errors)
        print(f"  Severity MAE         : {mae:.2f}")
    if segment_deltas:
        avg_delta = sum(segment_deltas) / len(segment_deltas)
        print(f"  Mean |segment delta| : {avg_delta:.2f}")
    print(f"  Laundering hits (sum): {laundering_total}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate an UnBias-Plus recipe model."
    )
    parser.add_argument(
        "--recipe",
        required=True,
        help="Recipe name/path; selects the prompt and the default model dir.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Merged model dir (default: outputs/<recipe>/merged_16bit).",
    )
    parser.add_argument("--text", default=None, help="Analyze a single raw string.")
    parser.add_argument(
        "--file", default=None, help="Analyze a .txt article or a .jsonl dataset."
    )
    parser.add_argument("--jsonl", default=None, help="Analyze a JSONL dataset (gold).")
    parser.add_argument(
        "--index", type=int, default=None, help="Single JSONL row index."
    )
    parser.add_argument("--limit", type=int, default=10, help="Max JSONL rows to run.")
    parser.add_argument(
        "--load-in-4bit", action="store_true", help="Load model in 4-bit."
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--raw", action="store_true", help="Print raw model JSON.")
    return parser.parse_args()


def resolve_jsonl_source(args: argparse.Namespace) -> str | None:
    """Return the JSONL path implied by ``--jsonl`` or a ``.jsonl`` ``--file``."""
    if args.jsonl:
        return args.jsonl
    if args.file and args.file.endswith(".jsonl"):
        return args.file
    return None


def main() -> None:
    """Dispatch to single-text, single-file, or dataset evaluation."""
    args = parse_args()
    cfg = load_recipe(args.recipe)
    model_path = args.model_path or f"outputs/{cfg.name}/merged_16bit"

    model = Model(
        model_path,
        prompt_id=cfg.prompt,
        max_seq_length=cfg.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        load_in_4bit=args.load_in_4bit,
    )

    jsonl_source = resolve_jsonl_source(args)
    if jsonl_source is not None:
        records = load_jsonl(jsonl_source)
        if args.index is not None:
            records = records[args.index : args.index + 1]
        else:
            records = records[: args.limit]
        evaluate_dataset(model, records)
        return

    if args.text is not None:
        article = args.text
    elif args.file is not None:
        article = Path(args.file).read_text(encoding="utf-8")
    else:
        raise SystemExit("Provide one of --text, --file, or --jsonl.")

    raw = model.generate(article)
    if args.raw:
        print(raw)
        return
    result = parse_json(raw)
    if result is None:
        print("<parse failure>\n")
        print(raw)
        return
    print_result(result, None)


if __name__ == "__main__":
    main()
