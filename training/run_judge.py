"""Evaluate model inference outputs using GPT-4o-mini as a judge.

Takes the JSONL output from run_inference.py and scores each sample.

DESIGN — Unpaired original bias:
  original_bias is computed in a separate unpaired call per article,
  independent of the rewrite. This prevents the judge from inflating or
  deflating the original score based on how much the rewrite changed.
  The unpaired score is computed once and shared across all models when
  the same inference JSONL is re-evaluated.

  Biased samples:
    Call 1 (unpaired): original article only  -> original_bias
    Call 2 (paired):   original + rewrite     -> rewrite_bias, contextual_relevance

  Unbiased samples:
    Call 1 (paired): original + predicted label + rewrite
                  -> correct_identification, unnecessary_rewrite



Usage:
  python train/run_judge.py \
      --inference-file train/inference_results/inference_qwen3_8b.jsonl \
      --output-dir     train/inference_results
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
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
VECTOR_PROXY_URL = "https://proxy.vectorinstitute.ai/v1"
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Cost estimates per 1M tokens (USD)
# ---------------------------------------------------------------------------
MODEL_COSTS = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

# ---------------------------------------------------------------------------
# Judge prompts
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


UNBIASED_JUDGE_PROMPT = """You are evaluating a bias detection model's output on a news article that is known to be unbiased.

ORIGINAL ARTICLE:
{original}

MODEL OUTPUT:
- Label predicted: {predicted_label}
- Rewritten text: {rewrite}

Rate the following on a 0-5 scale. Be lenient — minor paraphrasing or small word changes should not be heavily penalized.

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
    """Parse command line arguments."""
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--inference-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--base-url", default=VECTOR_PROXY_URL)
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples (for testing)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Judge calls
# ---------------------------------------------------------------------------


def _clamp(v: Any, lo: float, hi: float) -> float | None:
    if isinstance(v, (int, float)) and lo <= v <= hi:
        return float(v)
    return None


def _call(client: OpenAI, prompt: str, model: str) -> tuple[dict, dict]:
    """Single judge call with retry. Returns (parsed_data, usage)."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            usage = (
                {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens,
                }
                if resp.usage
                else {}
            )
            return data, usage
        except Exception as exc:
            logger.warning(
                "Judge attempt %d/%d failed: %s", attempt + 1, MAX_RETRIES, exc
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(2**attempt)
    return {}, {}


def judge_original(
    client: OpenAI, original: str, model: str
) -> tuple[float | None, dict]:
    """Unpaired call: score original article bias only."""
    data, usage = _call(
        client, ORIGINAL_BIAS_PROMPT.format(original=original[:6000]), model
    )
    return _clamp(data.get("original_bias"), 0, 5), usage


def judge_rewrite(
    client: OpenAI, original: str, rewrite: str, model: str
) -> tuple[dict[str, float | None], dict]:
    """Paired call: score rewrite bias and contextual relevance."""
    data, usage = _call(
        client,
        REWRITE_QUALITY_PROMPT.format(original=original[:6000], rewrite=rewrite[:6000]),
        model,
    )
    return {
        "rewrite_bias": _clamp(data.get("rewrite_bias"), 0, 5),
        "contextual_relevance": _clamp(data.get("contextual_relevance"), 0, 5),
    }, usage


def judge_unbiased(
    client: OpenAI, original: str, rewrite: str, predicted_label: str, model: str
) -> tuple[dict[str, float | None], dict]:
    """Paired call: score unbiased sample handling."""
    data, usage = _call(
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
    }, usage


# ---------------------------------------------------------------------------
# Stats + cost
# ---------------------------------------------------------------------------


def stats(scores: list[float]) -> dict[str, float | int]:
    """Calculate basic statistics for a list of scores."""
    if not scores:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": round(statistics.mean(scores), 3),
        "median": round(statistics.median(scores), 3),
        "min": round(min(scores), 3),
        "max": round(max(scores), 3),
        "n": len(scores),
    }


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate the USD cost of an OpenAI API call."""
    if model not in MODEL_COSTS:
        return 0.0
    c = MODEL_COSTS[model]
    return round(
        (prompt_tokens / 1_000_000) * c["input"]
        + (completion_tokens / 1_000_000) * c["output"],
        4,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: PLR0912, PLR0915
    """Run the evaluation script to score outputs using a judge model."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(str(project_root / ".env"))

    api_key = os.environ.get("VECTOR_API_KEY")
    if not api_key:
        raise ValueError("No API key found. Set VECTOR_API_KEY in your .env file.")

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    logger.info("[1/2] Loading inference results...")
    rows = []
    with args.inference_file.open(encoding="utf-8") as f:
        for line in f:
            clean_line = line.strip()
            if clean_line:
                rows.append(json.loads(clean_line))

    if args.max_samples:
        rows = rows[: args.max_samples]
        logger.info("  Limiting to %d samples for testing", len(rows))

    logger.info("  %d records loaded", len(rows))
    parse_rate = sum(r["parsed_ok"] for r in rows) / len(rows) if rows else 0.0
    logger.info("  Parse rate: %.1f%%", parse_rate * 100)

    logger.info("[2/2] Judging with %s via %s...", args.judge_model, args.base_url)

    # Score accumulators
    orig_scores, rew_scores, cr_scores = [], [], []
    reduction, reduction_pct = [], []
    ci_scores, ur_scores = [], []

    # Usage tracking
    usage_log = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_calls = 0
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
            }
        )

        if not r["parsed_ok"]:
            continue

        gold_label = r["gold"].get("binary_label")
        pred_label = r["pred"].get("binary_label", "")
        article = r["gold"]["article_text"]
        rewrite = r["pred"].get("unbiased_text", "") or article

        def _log_usage(usage: dict, sample_idx: int, call_type: str) -> None:
            nonlocal total_prompt_tokens, total_completion_tokens, total_calls
            pt = usage.get("prompt_tokens", 0)
            ct = usage.get("completion_tokens", 0)
            total_prompt_tokens += pt
            total_completion_tokens += ct
            total_calls += 1
            usage_log.append(
                {
                    "call_num": total_calls,
                    "sample_idx": sample_idx,
                    "call_type": call_type,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "total_tokens": pt + ct,
                    "estimated_cost_usd": estimate_cost(args.judge_model, pt, ct),
                }
            )

        if gold_label == "biased":
            # Call 1: unpaired — original bias only
            ob, usage1 = judge_original(client, article, args.judge_model)
            _log_usage(usage1, r["idx"], "original_bias_unpaired")

            # Call 2: paired — rewrite quality
            rw_scores, usage2 = judge_rewrite(
                client, article, rewrite, args.judge_model
            )
            _log_usage(usage2, r["idx"], "rewrite_quality_paired")

            rb = rw_scores["rewrite_bias"]
            cr = rw_scores["contextual_relevance"]

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
            scores, usage = judge_unbiased(
                client, article, rewrite, pred_label, args.judge_model
            )
            _log_usage(usage, r["idx"], "unbiased_paired")

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

    latencies = [r["latency"] for r in rows if "latency" in r]
    est_cost = estimate_cost(
        args.judge_model, total_prompt_tokens, total_completion_tokens
    )

    metrics: dict[str, Any] = {
        "inference_file": str(args.inference_file),
        "judge_model": args.judge_model,
        "judge_design": "unpaired_original_bias",
        "base_url": args.base_url,
        "n_samples": len(rows),
        "n_judged_biased": judged_biased,
        "n_judged_unbiased": judged_unbiased,
        "parse_rate": round(parse_rate, 3),
        "scale_note": "All scores are 0-5",
        "latency_seconds": stats(latencies),
        "original_bias": stats(orig_scores),
        "rewrite_bias": stats(rew_scores),
        "bias_reduction": stats(reduction),
        "bias_reduction_pct": stats(reduction_pct),
        "contextual_relevance": stats(cr_scores),
        "correct_identification": stats(ci_scores),
        "unnecessary_rewrite": stats(ur_scores),
        "token_usage": {
            "total_calls": total_calls,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "estimated_cost_usd": est_cost,
            "avg_prompt_tokens_per_call": round(total_prompt_tokens / total_calls, 1)
            if total_calls
            else 0,
            "avg_completion_tokens_per_call": round(
                total_completion_tokens / total_calls, 1
            )
            if total_calls
            else 0,
        },
        "usage_log": usage_log,
    }

    stem = args.inference_file.stem.replace("inference_", "")
    metrics_path = args.output_dir / f"metrics_{stem}.json"
    preds_path = args.output_dir / f"predictions_{stem}.jsonl"
    usage_path = args.output_dir / f"usage_{stem}.json"

    metrics_path.write_text(json.dumps(metrics, indent=2))
    with preds_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {
                        "idx": r["idx"],
                        "parsed_ok": r["parsed_ok"],
                        "latency": r.get("latency"),
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
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    usage_path.write_text(
        json.dumps(
            {
                "inference_file": str(args.inference_file),
                "judge_model": args.judge_model,
                "judge_design": "unpaired_original_bias",
                "total_calls": total_calls,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
                "estimated_cost_usd": est_cost,
                "per_call": usage_log,
            },
            indent=2,
        )
    )

    logger.info("\n=== REPORT ===")
    logger.info("Samples processed        : %d", metrics["n_samples"])
    logger.info("Biased judged            : %d", metrics["n_judged_biased"])
    logger.info("Unbiased judged          : %d", metrics["n_judged_unbiased"])
    logger.info(
        "Total judge calls        : %d (2 per biased, 1 per unbiased)", total_calls
    )
    logger.info("Parse rate               : %.1f%%", parse_rate * 100)
    logger.info("")
    logger.info(
        "Latency (mean/median)    : %.2fs / %.2fs",
        metrics["latency_seconds"]["mean"],
        metrics["latency_seconds"]["median"],
    )
    logger.info("")
    logger.info("Note: all scores are 0-5 scale")
    logger.info(
        "Note: original_bias computed via unpaired call (independent of rewrite)"
    )
    logger.info("")
    logger.info("--- Biased samples ---")
    logger.info(
        "Original bias  (mean/median): %s / %s",
        metrics["original_bias"]["mean"],
        metrics["original_bias"]["median"],
    )
    logger.info(
        "Rewrite bias   (mean/median): %s / %s",
        metrics["rewrite_bias"]["mean"],
        metrics["rewrite_bias"]["median"],
    )
    logger.info(
        "Bias reduction (mean/median): %s / %s",
        metrics["bias_reduction"]["mean"],
        metrics["bias_reduction"]["median"],
    )
    logger.info(
        "Bias reduction %% (mean/median): %s / %s",
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
    logger.info("")
    logger.info("--- Token usage ---")
    logger.info("Total calls       : %d", total_calls)
    logger.info("Prompt tokens     : %d", total_prompt_tokens)
    logger.info("Completion tokens : %d", total_completion_tokens)
    logger.info("Total tokens      : %d", total_prompt_tokens + total_completion_tokens)
    logger.info("Estimated cost    : $%.4f USD", est_cost)
    logger.info("")
    logger.info("Wrote %s", metrics_path)
    logger.info("Wrote %s", preds_path)
    logger.info("Wrote %s", usage_path)


if __name__ == "__main__":
    main()
