"""Evaluate a fine-tuned UnBias-Plus model and produce an aggregate report.

Judge design — unpaired original bias:
  original_bias is computed in a separate unpaired call per article,
  independent of the rewrite. This prevents the judge from inflating or
  deflating the original score based on how much the rewrite changed.

  Biased samples:
    Call 1 (unpaired): original article only  -> original_bias
    Call 2 (paired):   original + rewrite     -> rewrite_bias, contextual_relevance

  Unbiased samples:
    Call 1 (paired): original + predicted label + rewrite
                  -> correct_identification, unnecessary_rewrite

All scores are on a 0-5 scale.
Aggregate (mean / median / min / max) written to metrics.json.
Per-sample scores written to predictions.jsonl.

Setup:
  uv pip install python-dotenv openai
  Put OPENAI_API_KEY=sk-... in a .env file (or pass --env-path).

Usage:
  python eval_judge.py \
      --model-path /path/to/merged_16bit \
      --test-path  data/VLDBench_test.json \
      --output-dir ./eval_results \
      --env-path   /path/to/.env
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, cast

import torch
from dotenv import load_dotenv
from openai import OpenAI
from prompts import SFT_SYSTEM_PROMPT as SYSTEM_PROMPT
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)
MAX_ARTICLE_CHARS = 5000
DEFAULT_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
DEFAULT_JUDGE_BASE_URL = os.environ.get("OPENAI_BASE_URL")

# ---------------------------------------------------------------------------
# Judge prompts — unpaired design, 0-5 scale
# ---------------------------------------------------------------------------

ORIGINAL_BIAS_PROMPT = """Rate the bias level of the following news article.

ARTICLE:
{original}

original_bias (0-5):
0 = Fully neutral or factual
1 = Mostly neutral with minor word choice issues
2 = Some loaded language or mild framing
3 = Noticeable framing bias or loaded language
4 = Strong partisan or emotional tone
5 = Highly inflammatory or dehumanizing language

Respond with JSON only:
{{"original_bias": <0-5>}}"""

REWRITE_QUALITY_PROMPT = """Compare two versions of the same news article.

ORIGINAL:
{original}

REWRITE:
{rewrite}

Rate the following on a 0-5 scale.

rewrite_bias (0-5):
0 = Fully neutral or factual
1 = Mostly neutral with minor word choice issues
2 = Some loaded language or mild framing
3 = Noticeable framing bias or loaded language
4 = Strong partisan or emotional tone
5 = Highly inflammatory or dehumanizing language

contextual_relevance (0-5):
0 = Completely changed or unrelated
1 = Key facts significantly altered
2 = Most facts preserved but meaning shifted
3 = Facts mostly preserved with minor losses
4 = All facts preserved, mostly neutral tone
5 = All facts preserved, only biased language replaced

Respond with JSON only:
{{"rewrite_bias": <0-5>, "contextual_relevance": <0-5>}}"""

UNBIASED_JUDGE_PROMPT = """You are evaluating a bias detection model's output \
on a news article that is known to be unbiased.

ORIGINAL ARTICLE:
{original}

MODEL OUTPUT:
- Label predicted: {predicted_label}
- Rewritten text: {rewrite}

Rate the following on a 0-5 scale. Be lenient — minor paraphrasing or small \
word changes should not be heavily penalized.

correct_identification (0-5):
0 = Labeled as biased with many fabricated segments
2 = Labeled as biased but with only minor unjustified changes
3 = Correctly labeled as unbiased but rewrote parts of the text
5 = Correctly labeled as unbiased and preserved the original text

unnecessary_rewrite (0-5):
0 = Text heavily rewritten despite being unbiased
2 = Several sentences changed without clear justification
3 = Minor rephrasing of neutral text
5 = Original text preserved with at most trivial differences

Respond with JSON only:
{{"correct_identification": <0-5>, "unnecessary_rewrite": <0-5>}}"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--test-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--env-path",
        type=Path,
        default=None,
        help="Optional .env file path. Defaults to .env in CWD.",
    )
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument(
        "--judge-base-url",
        default=DEFAULT_JUDGE_BASE_URL,
        help="Optional OpenAI-compatible base URL, e.g. Vector proxy.",
    )
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--load-4bit", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model + inference
# ---------------------------------------------------------------------------


def load_model(model_path: Path, load_4bit: bool) -> tuple[Any, Any]:
    """Load the merged HF model."""
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    kwargs: dict[str, Any] = {"device_map": "auto"}
    if load_4bit:
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    else:
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)
    model.eval()
    return model, tokenizer


def build_prompt(article: str, tokenizer: Any) -> str:
    """Mirror the training prompt format exactly."""
    article = article[:MAX_ARTICLE_CHARS]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Analyze the following article for bias and return the result "
            f"in the required JSON format.\n\nARTICLE:\n{article}",
        },
    ]
    return str(
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def generate(
    model: Any, tokenizer: Any, prompt: str, max_new_tokens: int
) -> tuple[str, float]:
    """Run one generation. Returns (decoded_text, latency_seconds)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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


def parse_output(raw: str) -> dict[str, Any] | None:
    """Extract the first complete JSON object from raw model output."""
    start = raw.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(raw[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return cast(dict[str, Any], json.loads(raw[start : i + 1]))
                except json.JSONDecodeError:
                    return None

    return None


# ---------------------------------------------------------------------------
# Judge calls
# ---------------------------------------------------------------------------


def _clamp(v: Any, lo: float, hi: float) -> float | None:
    """Clamp and validate a numeric score within [lo, hi]."""
    if isinstance(v, (int, float)) and lo <= v <= hi:
        return float(v)
    return None


def _call(client: OpenAI, prompt: str, model: str) -> dict[str, Any]:
    """Single judge call with retry. Returns parsed response data."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return cast(
                dict[str, Any], json.loads(resp.choices[0].message.content or "{}")
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Judge attempt %d/%d failed: %s", attempt + 1, max_retries, exc
            )
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
    return {}


def judge_original(client: OpenAI, original: str, model: str) -> float | None:
    """Unpaired call: score original article bias only (0-5)."""
    data = _call(client, ORIGINAL_BIAS_PROMPT.format(original=original[:6000]), model)
    return _clamp(data.get("original_bias"), 0, 5)


def judge_rewrite(
    client: OpenAI, original: str, rewrite: str, model: str
) -> dict[str, float | None]:
    """Paired call: score rewrite bias and contextual relevance (0-5)."""
    data = _call(
        client,
        REWRITE_QUALITY_PROMPT.format(original=original[:6000], rewrite=rewrite[:6000]),
        model,
    )
    return {
        "rewrite_bias": _clamp(data.get("rewrite_bias"), 0, 5),
        "contextual_relevance": _clamp(data.get("contextual_relevance"), 0, 5),
    }


def judge_unbiased(
    client: OpenAI, original: str, rewrite: str, predicted_label: str, model: str
) -> dict[str, float | None]:
    """Paired call: score unbiased sample handling (0-5)."""
    data = _call(
        client,
        UNBIASED_JUDGE_PROMPT.format(
            original=original[:6000],
            rewrite=rewrite[:6000],
            predicted_label=predicted_label,
        ),
        model,
    )
    return {
        "correct_identification": _clamp(data.get("correct_identification"), 0, 5),
        "unnecessary_rewrite": _clamp(data.get("unnecessary_rewrite"), 0, 5),
    }


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------


def stats(scores: list[float]) -> dict[str, float | int]:
    """Mean / median / min / max / n for a list of scores."""
    if not scores:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": round(statistics.mean(scores), 3),
        "median": round(statistics.median(scores), 3),
        "min": round(min(scores), 3),
        "max": round(max(scores), 3),
        "n": len(scores),
    }


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------


def _run_inference(
    samples: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Run inference on all samples and return rows with raw + parsed output."""
    rows = []
    for i, s in enumerate(samples, 1):
        prompt = build_prompt(s["article_text"], tokenizer)
        raw, latency = generate(model, tokenizer, prompt, max_new_tokens)
        parsed = parse_output(raw)
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
                "  Inference %d/%d (last latency %.2fs)", i, len(samples), latency
            )
    return rows


# ---------------------------------------------------------------------------
# Judge loop
# ---------------------------------------------------------------------------


def _run_judge(  # noqa: PLR0912, PLR0915
    rows: list[dict[str, Any]],
    client: OpenAI,
    judge_model: str,
) -> dict[str, Any]:
    """Run judge on all parsed rows. Returns score lists and counts."""
    orig_scores: list[float] = []
    rew_scores: list[float] = []
    cr_scores: list[float] = []
    reduction: list[float] = []
    reduction_pct: list[float] = []
    ci_scores: list[float] = []
    ur_scores: list[float] = []
    length_ratios: list[float] = []
    judged_biased = 0
    judged_unbiased = 0

    for r in rows:
        r.update(
            {
                "original_bias": None,
                "rewrite_bias": None,
                "contextual_relevance": None,
                "bias_reduction": None,
                "bias_reduction_pct": None,
                "correct_identification": None,
                "unnecessary_rewrite": None,
                "rewrite_length_ratio": None,
            }
        )

        if not r["parsed_ok"]:
            continue

        gold_label = r["gold"].get("binary_label")
        pred_label = r["pred"].get("binary_label", "")
        article = r["gold"]["article_text"]
        rewrite = r["pred"].get("unbiased_text", "") or article

        ratio = round(len(rewrite) / len(article), 3) if article else None
        r["rewrite_length_ratio"] = ratio
        if ratio is not None:
            length_ratios.append(ratio)

        if gold_label == "biased":
            ob = judge_original(client, article, judge_model)
            rw = judge_rewrite(client, article, rewrite, judge_model)
            rb = rw["rewrite_bias"]
            cr = rw["contextual_relevance"]

            r["original_bias"] = ob
            r["rewrite_bias"] = rb
            r["contextual_relevance"] = cr

            if ob is not None:
                orig_scores.append(ob)
            if rb is not None:
                rew_scores.append(rb)
            if cr is not None:
                cr_scores.append(cr)
            if ob is not None and rb is not None:
                diff = ob - rb
                r["bias_reduction"] = round(diff, 3)
                reduction.append(diff)
                if ob > 0:
                    pct = diff / ob
                    r["bias_reduction_pct"] = round(pct, 3)
                    reduction_pct.append(pct)
            judged_biased += 1

        elif gold_label == "unbiased":
            scores = judge_unbiased(client, article, rewrite, pred_label, judge_model)
            ci = scores["correct_identification"]
            ur = scores["unnecessary_rewrite"]
            r["correct_identification"] = ci
            r["unnecessary_rewrite"] = ur
            if ci is not None:
                ci_scores.append(ci)
            if ur is not None:
                ur_scores.append(ur)
            judged_unbiased += 1

        if (judged_biased + judged_unbiased) % 20 == 0:
            logger.info(
                "  Judged %d biased + %d unbiased so far",
                judged_biased,
                judged_unbiased,
            )

    return {
        "orig_scores": orig_scores,
        "rew_scores": rew_scores,
        "cr_scores": cr_scores,
        "reduction": reduction,
        "reduction_pct": reduction_pct,
        "ci_scores": ci_scores,
        "ur_scores": ur_scores,
        "length_ratios": length_ratios,
        "judged_biased": judged_biased,
        "judged_unbiased": judged_unbiased,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run evaluation pipeline + write aggregate report."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.env_path:
        load_dotenv(str(args.env_path))
    else:
        load_dotenv()

    logger.info("[1/3] Loading test data + model...")
    with args.test_path.open(encoding="utf-8") as f:
        samples = json.load(f)[: args.max_samples]
    logger.info("  %d test samples", len(samples))
    model, tokenizer = load_model(args.model_path, args.load_4bit)

    logger.info("[2/3] Running inference...")
    rows = _run_inference(samples, model, tokenizer, args.max_new_tokens)

    logger.info(
        "[3/3] Judging with %s%s...",
        args.judge_model,
        f" @ {args.judge_base_url}" if args.judge_base_url else "",
    )
    client_kwargs: dict[str, Any] = {}
    if args.judge_base_url:
        client_kwargs["base_url"] = args.judge_base_url
    client = OpenAI(**client_kwargs)

    judge_results = _run_judge(rows, client, args.judge_model)

    metrics: dict[str, Any] = {
        "n_samples": len(rows),
        "n_judged_biased": judge_results["judged_biased"],
        "n_judged_unbiased": judge_results["judged_unbiased"],
        "parse_rate": round(
            sum(r["parsed_ok"] for r in rows) / len(rows) if rows else 0.0, 3
        ),
        "scale_note": "All scores are 0-5",
        "judge_design": "unpaired_original_bias",
        "original_bias": stats(judge_results["orig_scores"]),
        "rewrite_bias": stats(judge_results["rew_scores"]),
        "bias_reduction": stats(judge_results["reduction"]),
        "bias_reduction_pct": stats(judge_results["reduction_pct"]),
        "contextual_relevance": stats(judge_results["cr_scores"]),
        "correct_identification": stats(judge_results["ci_scores"]),
        "unnecessary_rewrite": stats(judge_results["ur_scores"]),
        "rewrite_length_ratio": stats(judge_results["length_ratios"]),
    }

    metrics_path = args.output_dir / "metrics.json"
    preds_path = args.output_dir / "predictions.jsonl"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    with preds_path.open("w") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {
                        "idx": r["idx"],
                        "parsed_ok": r["parsed_ok"],
                        "latency": r["latency"],
                        "gold": r["gold"],
                        "pred": r["pred"],
                        "raw_output": r["raw_output"],
                        "original_bias": r["original_bias"],
                        "rewrite_bias": r["rewrite_bias"],
                        "bias_reduction": r["bias_reduction"],
                        "bias_reduction_pct": r["bias_reduction_pct"],
                        "contextual_relevance": r["contextual_relevance"],
                        "correct_identification": r["correct_identification"],
                        "unnecessary_rewrite": r["unnecessary_rewrite"],
                        "rewrite_length_ratio": r["rewrite_length_ratio"],
                    }
                )
                + "\n"
            )

    logger.info("\n=== REPORT ===")
    logger.info("Samples processed        : %d", metrics["n_samples"])
    logger.info("Biased judged            : %d", metrics["n_judged_biased"])
    logger.info("Unbiased judged          : %d", metrics["n_judged_unbiased"])
    logger.info("Parse rate               : %.1f%%", metrics["parse_rate"] * 100)
    logger.info("Note: all scores are 0-5 scale")
    logger.info("Note: original_bias computed via unpaired call")
    logger.info("")
    logger.info("--- Biased samples ---")
    logger.info(
        "Original bias        (mean/median): %s / %s",
        metrics["original_bias"]["mean"],
        metrics["original_bias"]["median"],
    )
    logger.info(
        "Rewrite bias         (mean/median): %s / %s",
        metrics["rewrite_bias"]["mean"],
        metrics["rewrite_bias"]["median"],
    )
    logger.info(
        "Bias reduction       (mean/median): %s / %s",
        metrics["bias_reduction"]["mean"],
        metrics["bias_reduction"]["median"],
    )
    logger.info(
        "Bias reduction %%     (mean/median): %s / %s",
        metrics["bias_reduction_pct"]["mean"],
        metrics["bias_reduction_pct"]["median"],
    )
    logger.info(
        "Contextual relevance (mean/median): %s / %s",
        metrics["contextual_relevance"]["mean"],
        metrics["contextual_relevance"]["median"],
    )
    logger.info("")
    logger.info("--- Unbiased samples ---")
    logger.info(
        "Correct identification (mean/median): %s / %s",
        metrics["correct_identification"]["mean"],
        metrics["correct_identification"]["median"],
    )
    logger.info(
        "Unnecessary rewrite    (mean/median): %s / %s",
        metrics["unnecessary_rewrite"]["mean"],
        metrics["unnecessary_rewrite"]["median"],
    )
    logger.info(
        "Rewrite length ratio   (mean/median): %s / %s",
        metrics["rewrite_length_ratio"]["mean"],
        metrics["rewrite_length_ratio"]["median"],
    )
    logger.info("")
    logger.info("Wrote %s", metrics_path)
    logger.info("Wrote %s", preds_path)


if __name__ == "__main__":
    main()
