"""Run inference on a test set using a fine-tuned UnBias-Plus model.

Saves raw outputs and parsed predictions to a JSONL file for downstream
judge evaluation. No OpenAI API calls here — GPU job only.

Supports both BABE golden schema (text, neutral_rewrite) and
VLDBench schema (article_text, unbiased_text) — field mapping is handled
automatically at load time.

Sampling is stratified by binary_label to preserve the biased/unbiased
ratio of the source dataset.

Usage:
  python train/run_inference.py \
      --model-path train/Models/qwen3_8b/merged_16bit \
      --test-path  evaluation/babe_golden_500.json \
      --output-dir train/inference_results \
      --max-samples 100

# ---------------------------------------------------------------------------
# Post-processing notes per model (applied uniformly to all models)
# ---------------------------------------------------------------------------
#
# qwen3_8b (local, new training):
#   - Output format: clean JSON, no fences
#   - Issues: occasional duplicate biased_segments (same "original" repeated)
#   - Issues: occasional spurious extra fields (e.g. "unbiased_text_full")
#   - Fix applied: deduplicate segments, strip non-schema fields
#
# qwen35_4b (local, new training):
#   - Output format: clean JSON, no fences
#   - Issues: occasional very large single segment covering the entire article
#   - Fix applied: deduplicate segments, strip non-schema fields
#   - Note: ~84s/sample (slower due to VLM architecture)
#
# vector-institute/Qwen3-8B-UnBias-Plus-SFT-Instruct (HuggingFace, old training):
#   - Output format: valid JSON followed by extra text after closing brace
#   - Output format: thinking tokens (<think>/</think>) and double JSON blocks
#   - Fix applied: balanced bracket extractor ignores everything after first
#                  complete JSON object, handles thinking tokens and extra text
#   - Note: old model trained without format_sample fix (open-ended sequence)
#
# Future demo integration note:
#   All post-processing steps (fence stripping, bracket extraction,
#   deduplication, field filtering, hallucinated segment removal) are applied
#   uniformly. Any new model added to the demo should go through the same
#   parse_output pipeline without special-casing.
#
# Known model limitation — hallucinated segments (all models):
#   Occasionally a model outputs a biased_segment where the "original" field
#   is not a substring of the input article (e.g. it uses the replacement text
#   or a paraphrased version as the "original"). These segments are silently
#   removed in clean_parsed() since they cannot be grounded in the source text
#   and would break any highlight/replacement logic in the demo.
# ---------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch


logger = logging.getLogger(__name__)

MAX_ARTICLE_CHARS = 50_000

SCHEMA_FIELDS = {
    "binary_label",
    "severity",
    "bias_found",
    "biased_segments",
    "unbiased_text",
}

# ---------------------------------------------------------------------------
# System prompt — must match train_sft.py exactly for inference/training parity
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert linguist and bias detection specialist.
Your task is to carefully read a news article, detect ALL biased language,
and return a structured JSON response.

## BIAS TYPES
- loaded language             : words with strong emotional connotations
- dehumanizing framing        : language that strips dignity from groups
- false generalizations       : sweeping statements ("they always", "all of them")
- framing bias                : selective wording that implies a viewpoint
- euphemism/dysphemism        : softening or hardening language to manipulate perception
- politically charged terminology : labels used to provoke rather than describe
- sensationalism              : exaggerated language to evoke emotional responses

## SEGMENT RULES
- A segment is a consecutive sequence of words forming ONE biased idea.
- Prefer fewer, longer segments over many short overlapping ones.
- If two biased words are adjacent and part of the same biased idea → ONE segment.
- If biased words are separated by neutral words → SEPARATE segments.
- "original" MUST be the EXACT substring as it appears in the input (case-sensitive).
- Only modify phrases listed in biased_segments; preserve all factual content.
- Replacements must be similar in length to the original phrase. Do not use a long phrase to replace a short one.

## SEVERITY (per segment — string value)
- high   : dehumanizing, hateful, or strongly prejudiced language
- medium : framing bias, loaded terms, misleading generalizations
- low    : subtle word choice bias, mild framing issues

## GLOBAL SEVERITY (article-level — integer value)
- 0 : neutral / no bias
- 2 : recurring biased framing
- 3 : strong persuasive tone
- 4 : inflammatory rhetoric

## OUTPUT SCHEMA
Return ONLY a raw JSON object. No markdown, no code fences, no backticks.
The response must start with { and end with }.
{
  "binary_label": "biased" | "unbiased",
  "severity": 0 | 2 | 3 | 4,              // GLOBAL article-level integer
  "bias_found": true | false,
  "biased_segments": [
    {
      "original": "exact substring from input",
      "replacement": "neutral alternative phrase in the same language as original",
      "severity": "high" | "medium" | "low",   // SEGMENT-level string
      "bias_type": "loaded language | dehumanizing framing | false generalizations | framing bias | euphemism/dysphemism | politically charged terminology | sensationalism",
      "reasoning": "1-2 sentence explanation of why this is biased"
    }
  ],
  "unbiased_text": "Full rewritten neutral article in the same language as the input"
}

## REWRITE RULES
- Build unbiased_text by replacing each biased phrase with its neutral replacement from biased_segments.
- Only modify phrases listed in biased_segments — leave everything else unchanged.
- Preserve the original article's facts, structure, and length. The rewritten text must be as close in length as possible to the original. Do not add sentences, expand phrases, or elaborate. Only swap biased phrases with neutral alternatives of similar length.
- Do not add new information, opinions, or commentary.
- If the article is unbiased, return the original text exactly as-is.

## LANGUAGE HANDLING
- Always respond in the same language as the input article.
- All text fields (original, replacement, unbiased_text) must be in the article's original language.
- JSON keys must always remain in English.
- If the article's language is not well-supported, return unbiased_text in English and note the limitation in the reasoning field.

Rules:
- If no bias: severity=0, bias_found=false, biased_segments=[], unbiased_text=<original text unchanged>
- If biased: severity must be 2, 3, or 4 — never 0
- Return ONLY the JSON object. No preamble, no markdown fences.
""".strip()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--test-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--load-4bit", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading with schema normalization + stratified sampling
# ---------------------------------------------------------------------------


def normalize_sample(s: dict) -> dict:
    """Normalize BABE golden and VLDBench schemas to a common format.

    BABE golden : text, neutral_rewrite, uuid
    VLDBench    : article_text, unbiased_text, index
    """
    if "text" in s and "article_text" not in s:
        s["article_text"] = s["text"]
    return s


def filter_samples(samples: list[dict]) -> list[dict]:
    """Remove biased samples that have no biased_words annotations.

    BABE has 83 biased samples without biased_words (annotator gap).
    These are excluded to ensure all 100 evaluation samples are fully
    annotated and consistent across model runs.
    """
    filtered = []
    excluded = 0
    for s in samples:
        if s.get("binary_label") == "biased" and not s.get("biased_words"):
            excluded += 1
            continue
        filtered.append(s)
    if excluded:
        logger.info("  Excluded %d biased samples without biased_words", excluded)
    return filtered


def stratified_sample(samples: list[dict], n: int, seed: int) -> list[dict]:
    """Sample n items preserving the biased/unbiased ratio of the dataset."""
    random.seed(seed)
    biased = [s for s in samples if s.get("binary_label") == "biased"]
    unbiased = [s for s in samples if s.get("binary_label") == "unbiased"]

    total = len(samples)
    n_biased = round(n * len(biased) / total)
    n_unbiased = n - n_biased

    sampled_biased = random.sample(biased, min(n_biased, len(biased)))
    sampled_unbiased = random.sample(unbiased, min(n_unbiased, len(unbiased)))
    result = sampled_biased + sampled_unbiased
    random.shuffle(result)

    logger.info(
        "  Stratified sample: %d biased + %d unbiased = %d total",
        len(sampled_biased),
        len(sampled_unbiased),
        len(result),
    )
    return result


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(model_path: Path, load_4bit: bool) -> tuple[Any, Any]:
    """Load model with Unsloth first for optimized inference.

    Falls back to AutoModelForCausalLM if Unsloth doesn't support the architecture.
    """
    try:
        from unsloth import FastLanguageModel  # noqa: PLC0415

        model, tokenizer = FastLanguageModel.from_pretrained(
            str(model_path),
            load_in_4bit=load_4bit,
            dtype=torch.bfloat16,
            max_seq_length=4096,
        )
        FastLanguageModel.for_inference(model)
        logger.info("  Loaded with Unsloth")
        return model, tokenizer
    except Exception:
        logger.info("  Unsloth failed — falling back to AutoModelForCausalLM...")
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        kwargs: dict[str, Any] = {"device_map": "auto"}
        if load_4bit:
            from transformers import BitsAndBytesConfig  # noqa: PLC0415

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)
        model.eval()
        logger.info("  Loaded with AutoModelForCausalLM")
        return model, tokenizer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def build_prompt(article: str, tokenizer: Any) -> str:
    """Build inference prompt matching training format exactly."""
    article = article[:MAX_ARTICLE_CHARS]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Analyze the following article for bias and return the result "
                f"in the required JSON format.\n\nARTICLE:\n{article}"
            ),
        },
    ]
    return tokenizer.apply_chat_template(  # type: ignore
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate(
    model: Any, tokenizer: Any, prompt: str, max_new_tokens: int
) -> tuple[str, float]:
    """Run one generation. Returns (decoded_text, latency_seconds)."""
    # VLM processor unwrapping for Ministral/Gemma4/Qwen3.5
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0
    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True), elapsed


def extract_first_json(text: str) -> str | None:
    """Extract the first complete JSON object using balanced bracket counting.

    More robust than regex for:
    - Extra text after the closing brace (old HF model behavior)
    - Thinking tokens and double JSON blocks (old HF model with thinking enabled)
    - Missing closing brace (smoke-test model behavior)
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i, c in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if c == "\\" and in_string:
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
        if not in_string:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    # No closing brace found — return everything from start (incomplete JSON)
    if depth > 0:
        logger.debug(
            "DEBUG: model did not close JSON — bracket extractor appending brace"
        )
        return text[start:] + "\n}"
    return None


def clean_parsed(data: dict, article_text: str = "") -> dict:
    """Apply uniform post-processing to parsed model output.

    1. Strip non-schema fields (e.g. spurious 'unbiased_text_full')
    2. Deduplicate biased_segments by 'original' field
       (qwen3_8b and llama31_8b occasionally repeat the same segment)
    3. Remove hallucinated segments where 'original' is not a substring
       of the input article — these cannot be grounded in the source text
       and would break highlight/replacement logic in the demo
    """
    # Keep only schema fields
    data = {k: v for k, v in data.items() if k in SCHEMA_FIELDS}

    # Deduplicate and validate segments
    segments = data.get("biased_segments", [])
    if segments:
        seen = set()
        cleaned = []
        for seg in segments:
            key = seg.get("original", "")
            # Skip duplicates
            if key in seen:
                continue
            seen.add(key)
            # Skip hallucinated segments — original must exist in article
            if article_text and key and key not in article_text:
                logger.debug(
                    "DEBUG: removed hallucinated segment (original not in article): %s",
                    key[:60],
                )
                continue
            cleaned.append(seg)
        if len(cleaned) < len(segments):
            logger.debug(
                "DEBUG: removed %d segments (duplicates or hallucinated)",
                len(segments) - len(cleaned),
            )
        data["biased_segments"] = cleaned

    return data


def parse_output(raw: str, article_text: str = "") -> dict[str, Any] | None:
    """Extract and clean the first JSON object from raw model output.

    Handles all known model output quirks uniformly:
    - Markdown code fences (Ministral wraps output in ```json ... ```)
    - Extra text after closing brace (old HF model)
    - Thinking tokens and double JSON blocks (old HF model with thinking)
    - Missing closing brace (smoke-test / truncated outputs)
    - Duplicate segments and spurious fields (qwen3_8b, llama31_8b)
    - Hallucinated segments where original not in article (all models)
    """
    raw = raw.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
        logger.debug("DEBUG: stripped markdown fences from output")

    json_str = extract_first_json(raw)
    if json_str is None:
        return None

    try:
        data = json.loads(json_str)
        return clean_parsed(data, article_text)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the main inference loop and save results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[1/2] Loading test data + model...")
    with args.test_path.open(encoding="utf-8") as f:
        all_samples = [normalize_sample(s) for s in json.load(f)]

    all_samples = filter_samples(all_samples)
    samples = stratified_sample(all_samples, args.max_samples, args.seed)
    logger.info("  %d test samples", len(samples))

    model, tokenizer = load_model(args.model_path, args.load_4bit)

    logger.info("[2/2] Running inference...")
    rows = []
    latencies = []

    for i, s in enumerate(samples, 1):
        prompt = build_prompt(s["article_text"], tokenizer)
        raw, latency = generate(model, tokenizer, prompt, args.max_new_tokens)
        parsed = parse_output(raw, s["article_text"])
        latencies.append(latency)
        rows.append(
            {
                "idx": i - 1,
                "gold": s,
                "raw_output": raw,
                "pred": parsed,
                "latency": latency,
                "parsed_ok": parsed is not None,
            }
        )
        if i % 10 == 0:
            logger.info(
                "  Inference %d/%d | latency %.2fs | avg %.2fs",
                i,
                len(samples),
                latency,
                sum(latencies) / len(latencies),
            )

    # Runtime summary
    parse_rate = sum(1 for r in rows if r["parsed_ok"]) / len(rows) if rows else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    logger.info("\n=== INFERENCE SUMMARY ===")
    logger.info("Samples           : %d", len(rows))
    logger.info("Parse rate        : %.1f%%", parse_rate * 100)
    logger.info("Avg latency       : %.2fs", avg_latency)
    logger.info("Total time        : %.1fmin", sum(latencies) / 60)

    # Output filename — use model key for local models, model name for HF models
    model_path_obj = Path(str(args.model_path))
    if model_path_obj.name == "merged_16bit":
        model_name = model_path_obj.parent.name  # e.g. qwen3_8b, qwen35_4b
    else:
        model_name = model_path_obj.name  # e.g. Qwen3-8B-UnBias-Plus-SFT-Instruct
    out_file = args.output_dir / f"inference_{model_name}.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("\nSaved %d records to %s", len(rows), out_file)
    logger.info("Run run_judge.py --inference-file %s to evaluate.", out_file)


if __name__ == "__main__":
    main()
