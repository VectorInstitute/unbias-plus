"""Quick inference check to verify the merged model works correctly.

Tests:
  - English biased article
  - English neutral article
  - French biased article   (language handling fix)
  - Arabic biased article   (language handling fix)

Usage:
    python train/quick_test.py
    python train/quick_test.py --model-path /path/to/merged_16bit
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch


# ---------------------------------------------------------------------------
# System prompt — must match train_sft.py exactly
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

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Models", "qwen3_8b", "merged_16bit"
)

# ---------------------------------------------------------------------------
# Test articles
# ---------------------------------------------------------------------------

ARTICLES = {
    "english_biased": (
        "The radical extremist senator pushed his dangerous agenda through "
        "the corrupt establishment, ignoring the will of real Americans."
    ),
    "english_neutral": (
        "The senator introduced new legislation on healthcare reform this week. "
        "The bill passed committee review with support from members of both parties "
        "and is scheduled for a floor vote next month."
    ),
    "french_biased": (
        "Le sénateur extrémiste radical a imposé son agenda dangereux à travers "
        "l'establishment corrompu, ignorant la volonté des vrais Français."
    ),
    "arabic_biased": (
        "دفع السناتور المتطرف الراديكالي أجندته الخطيرة عبر المؤسسة الفاسدة، "
        "متجاهلاً إرادة المواطنين الحقيقيين."
    ),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Quick inference check for merged model."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to merged_16bit model directory.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(model_path: str) -> tuple[Any, Any]:
    """Load model with AutoModelForCausalLM.

    Falls back to Unsloth for architectures not supported by standard
    transformers (e.g. Ministral).
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        tok = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()
        print("  Loaded with AutoModelForCausalLM")
        return model, tok
    except (ValueError, OSError):
        print("  AutoModelForCausalLM failed — falling back to Unsloth...")
        from unsloth import FastLanguageModel  # noqa: PLC0415

        model, tok = FastLanguageModel.from_pretrained(
            model_path, load_in_4bit=False, dtype=torch.bfloat16
        )
        FastLanguageModel.for_inference(model)
        print("  Loaded with Unsloth")
        return model, tok


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def build_prompt(article: str, tokenizer: Any) -> Any:
    """Build inference prompt matching the training format exactly."""
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
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def run(label: str, article: str, model: Any, tokenizer: Any) -> None:
    """Run inference for a single article and print the parsed output."""
    prompt = build_prompt(article, tokenizer)

    # VLM processor unwrapping for Ministral/Gemma4/Qwen3.5
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = out[0][inputs.input_ids.shape[1] :]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=False)
    clean = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"\n{'=' * 60}")
    print(f"  TEST: {label.upper()}")
    print(f"{'=' * 60}")
    print(f"\n  Input:\n  {article}")
    print(f"\n  Raw output:\n  {raw}")
    print(f"\n  Clean output:\n  {clean}")

    try:
        clean = clean.strip()

        # Strip markdown fences if present
        if clean.startswith("```"):
            print("  DEBUG: model wrapped output in markdown fences — stripping")
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
            clean = clean.strip()

        # Append closing brace if missing
        if not clean.endswith("}"):
            print("  DEBUG: model did not close JSON — appending closing brace")
            clean = clean + "\n}"

        parsed = json.loads(clean)
        print(f"\n  binary_label : {parsed.get('binary_label')}")
        print(f"  severity     : {parsed.get('severity')}")
        print(f"  bias_found   : {parsed.get('bias_found')}")
        lang_check = parsed.get("unbiased_text", "")[:80]
        print(f"  unbiased_text (first 80 chars): {lang_check}")
    except json.JSONDecodeError:
        print("\n  WARNING: output is not valid JSON")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Execute the quick test suite against the specified model."""
    args = parse_args()

    print(f"Loading model from: {args.model_path}")
    model, tok = load_model(args.model_path)
    print("Model loaded.\n")

    for label, article in ARTICLES.items():
        run(label, article, model, tok)

    print(f"\n{'=' * 60}")
    print("  All tests complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
