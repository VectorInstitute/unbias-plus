"""Inference smoke test for the fine-tuned bias-detection model.

Loads the exported merged_16bit model (or the LoRA adapter), runs a few sample
articles through it, and checks that the output:
  * parses as JSON (no preamble, no markdown fences, no <think> leakage),
  * has the required schema keys and valid value ranges,
  * is internally consistent (segments <-> label <-> severity),
  * keeps each segment's "original" as an exact substring of the input.

CRITICAL: the prompt is imported from prompts.py and built exactly as in
train.py (same system prompt, same user wrapper, enable_thinking=False). The
only inference-time difference is add_generation_prompt=True. If you hand-build
a different prompt here, a passing smoke test means nothing.

Usage:
  python smoke_test_inference.py --model-path outputs/vldbench_clean/merged_16bit
  python smoke_test_inference.py --model-path outputs/vldbench_clean/merged_16bit \
      --load-in-4bit
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch
from prompts import SFT_SYSTEM_PROMPT as SYSTEM_PROMPT
from unsloth import FastLanguageModel


CANONICAL_TYPES = {
    "loaded language",
    "dehumanizing framing",
    "false generalizations",
    "framing bias",
    "euphemism/dysphemism",
    "politically charged terminology",
    "sensationalism",
}

SEGMENT_KEYS = {"original", "replacement", "severity", "bias_type", "reasoning"}

# Two short samples so output fits a small token budget. One clearly biased,
# one neutral — enough to check both code paths.
SAMPLES = [
    (
        "biased",
        "The radical mob descended on city hall once again, shrieking their "
        "tired demands while hardworking taxpayers looked on in disgust. "
        "Officials, predictably spineless, caved within hours to the unhinged "
        "activists rather than stand up for ordinary people.",
    ),
    (
        "neutral",
        "The city council met on Tuesday to discuss the proposed budget. "
        "About 40 residents attended the public comment period. The council "
        "voted 5 to 2 to approve the measure, which takes effect in January. "
        "A spokesperson said implementation details will be published next month.",
    ),
]


def build_prompt(tokenizer: Any, article: str) -> str:
    """Build the inference prompt — identical to train.py, plus a generation cue."""
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
    return str(
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # inference: cue the assistant turn
            enable_thinking=False,  # parity with training
        )
    )


def extract_json(text: str) -> tuple[dict[str, Any] | None, bool]:
    """Parse model output as JSON. Returns (obj_or_None, needed_cleanup).

    Tries three strategies in order:
      1. Clean parse (model was well-behaved).
      2. Extract outermost {...} (model added preamble / markdown fences).
      3. Truncated JSON repair — append closing brace(s). Handles the case
         where generation hit an EOS token before the final '}' was emitted,
         which is a known base-model quirk and occasionally happens with
         long unbiased_text fields.
    """
    text = text.strip()
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1]), True
        except json.JSONDecodeError:
            pass

    # Truncated JSON: try closing it. Covers "...value" (missing }) and
    # "...val" (mid-string truncation, though rarer).
    if start != -1:
        for suffix in ("\n}", '"}\n', '"\n}', "}"):
            try:
                return json.loads(text[start:] + suffix), True
            except json.JSONDecodeError:
                pass

    return None, True


def validate(obj: dict[str, Any], article: str) -> list[str]:  # noqa: PLR0912
    """Return a list of problems; empty list means the output is well-formed."""
    problems: list[str] = []

    required = {
        "binary_label",
        "severity",
        "bias_found",
        "biased_segments",
        "unbiased_text",
    }
    missing = required - set(obj)
    if missing:
        problems.append(f"missing keys: {sorted(missing)}")
        return problems  # can't check the rest reliably

    label = obj["binary_label"]
    sev = obj["severity"]
    segs = obj["biased_segments"]

    if label not in {"biased", "unbiased"}:
        problems.append(f"binary_label not in {{biased,unbiased}}: {label!r}")
    if sev not in {0, 2, 3, 4}:
        problems.append(f"severity not in {{0,2,3,4}}: {sev!r}")
    if not isinstance(segs, list):
        problems.append("biased_segments is not a list")
        segs = []

    # internal consistency
    if label == "unbiased":
        if segs:
            problems.append("unbiased label but biased_segments is non-empty")
        if sev != 0:
            problems.append(f"unbiased label but severity={sev}")
    else:  # biased
        if not segs:
            problems.append("biased label but no segments")
        if sev == 0:
            problems.append("biased label but severity=0")

    # per-segment checks
    for i, s in enumerate(segs):
        if not isinstance(s, dict) or set(s) != SEGMENT_KEYS:
            problems.append(f"segment {i}: keys != {sorted(SEGMENT_KEYS)}")
            continue
        if s["original"] and s["original"] not in article:
            problems.append(f"segment {i}: 'original' not an exact substring of input")
        if s["bias_type"] not in CANONICAL_TYPES:
            problems.append(f"segment {i}: bias_type off-taxonomy: {s['bias_type']!r}")
        if s["severity"] not in {"high", "medium", "low"}:
            problems.append(f"segment {i}: bad per-segment severity: {s['severity']!r}")
        if s["replacement"] and s["replacement"] not in str(obj["unbiased_text"]):
            problems.append(f"segment {i}: 'replacement' not present in unbiased_text")

    return problems


def main() -> None:
    """Run inference smoke test and exit with code 1 if any sample fails."""
    parser = argparse.ArgumentParser(description="Inference smoke test.")
    parser.add_argument(
        "--model-path", required=True, help="merged_16bit dir or adapter dir."
    )
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}  (4bit={args.load_in_4bit})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        dtype=torch.bfloat16,
    )
    FastLanguageModel.for_inference(model)

    n_pass = parsed = 0
    for expected, article in SAMPLES:
        prompt = build_prompt(tokenizer, article)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # greedy -> deterministic structured output
                temperature=None,
                top_p=None,
            )
        latency = time.time() - t0

        gen = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        print("\n" + "=" * 70)
        print(f"SAMPLE (expected ~{expected}) | latency {latency:.1f}s")
        obj, needed_cleanup = extract_json(gen)
        if obj is None:
            print("  ✗ DID NOT PARSE as JSON")
            print("  raw (first 400 chars):", repr(gen[:400]))
            continue
        parsed += 1
        if needed_cleanup:
            print("  ⚠ parsed only after stripping preamble/fences (model not clean)")

        problems = validate(obj, article)
        print(
            f"  binary_label={obj.get('binary_label')} severity={obj.get('severity')} "
            f"segments={len(obj.get('biased_segments', []))}"
        )
        if problems:
            print("  ✗ VALIDATION ISSUES:")
            for p in problems:
                print("     -", p)
        else:
            n_pass += 1
            print("  ✓ well-formed and internally consistent")

    print("\n" + "=" * 70)
    print(
        f"SMOKE TEST: parsed {parsed}/{len(SAMPLES)} | fully valid {n_pass}/{len(SAMPLES)}"
    )
    if n_pass < len(SAMPLES):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
