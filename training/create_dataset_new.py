"""Data generation pipeline for unbias-plus.

Reads VLDBench articles, preprocesses them, and annotates each with
GPT-5.4 via Vector Institute proxy. Results are written to a JSONL
checkpoint file as they arrive — safe to interrupt and resume.

Usage
-----
# Full run
python create_dataset_new.py --input vldb.csv --output annotations.jsonl

# Quick smoke test (10 samples)
python create_dataset_new.py --input vldb.csv --output annotations.jsonl --num-samples 10

# Resume interrupted run (skips already-completed indices)
python create_dataset_new.py --input vldb.csv --output annotations.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "gpt-5.4"
MAX_CHARS = 4500
MAX_RETRIES = 4
RETRY_BASE_DELAY = 2.0

# ---------------------------------------------------------------------------
# Structured output schema — matches SFT training format exactly
# ---------------------------------------------------------------------------

class BiasedSegment(BaseModel):
    original: str
    replacement: str
    severity: Literal["low", "medium", "high"]
    bias_type: str
    reasoning: str


class BiasAnalysis(BaseModel):
    binary_label: Literal["biased", "unbiased"]
    severity: Literal[0, 1, 2, 3, 4]
    bias_found: bool
    biased_segments: list[BiasedSegment]
    unbiased_text: str


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert linguist and bias annotation specialist producing training data for a debiasing model.

# Goal
Analyze the article for biased language. Identify every biased segment, classify it, and produce a fully neutralized rewrite.

# Success criteria
- Every segment's "original" field is an exact verbatim substring of the input article
- Replacements eliminate the source of bias — not just soften the wording
- Unsupported generalizations about demographic groups (gender, race, religion, nationality, age) are NOT facts to preserve — they are the bias itself. Remove or restructure them entirely; do not paraphrase them into milder form
- The rewritten article passes this test: no sentence still negatively characterizes a group without explicit attribution to a named source
- All factual events, named entities, dates, statistics, and attributed direct quotes are preserved

# Bias types
loaded language | dehumanizing framing | false generalizations | framing bias | euphemism/dysphemism | politically charged terminology | sensationalism

# Severity scale (article-level)
0 = Fully neutral, no bias detected
1 = Minor isolated loaded words
2 = Recurring biased framing
3 = Strong persuasive or partisan tone
4 = Inflammatory rhetoric or sustained attack language

# Segment severity
high   = dehumanizing, hateful, strongly prejudiced language
medium = framing bias, loaded terms, misleading generalizations
low    = subtle word choice bias, mild framing issues

# Rewriting constraints
- Preserve the approximate length of the original article, do not summarize or compress.
- If severity is 0: return bias_found=false, empty segments list, unbiased_text unchanged
- Do not flag neutral factual statements as biased
- Prefer fewer longer segments over many short overlapping ones

# Output schema — return ONLY valid JSON, no preamble, no markdown fences
{
  "binary_label": "biased" | "unbiased",
  "severity": 0 | 1 | 2 | 3 | 4,
  "bias_found": true | false,
  "biased_segments": [
    {
      "original": "exact verbatim substring from article",
      "replacement": "neutral alternative",
      "severity": "low" | "medium" | "high",
      "bias_type": "loaded language" | "dehumanizing framing" | "false generalizations" | "framing bias" | "euphemism/dysphemism" | "politically charged terminology" | "sensationalism",
      "reasoning": "1-2 sentence explanation"
    }
  ],
  "unbiased_text": "full rewritten article"
}"""

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

_ARTIFACT_LINES: frozenset[str] = frozenset({
    "Advertisement",
    "Story continues below advertisement",
    "Story continues below",
    "This advertisement has not loaded yet, but your article continues below.",
    "Article content",
    "Share",
    "Trending Now",
    "End of carousel",
    "Skip to end of carousel",
    "Subscribe for unlimited access to The Post",
    "Save up to 70% for a limited time.",
    "Get your first year for CA$2",
    "every four weeks",
    "Daily puzzles including the New York Times Crossword.",
    "Enjoy the latest local, national and international news.",
    "Support local journalism.",
    "Reviews",
    "Why We Picked It",
    "Why Trust Post Wanted by the New York Post",
    "CLICK HERE TO GET THE FOX NEWS APP",
    "Sign-up for Your Vote: Text with the USA TODAY elections team.",
    "Frequently asked questions (FAQs)",
    "Pros",
    "Cons",
    "___",
})

_ARTIFACT_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\d+\s+weeks?\s+ago.*$", re.IGNORECASE),
    re.compile(r"^Jul(?:y)?\s+\d{4}.*$", re.IGNORECASE),
    re.compile(r"^–\s+.+$"),
    re.compile(r"^Exclusive articles by .+$", re.IGNORECASE),
    re.compile(r"^Unlimited online access to .+$", re.IGNORECASE),
    re.compile(r"^National Post ePaper.+$", re.IGNORECASE),
    re.compile(r"^We rank local service providers.+$", re.IGNORECASE),
    re.compile(r"^Services offered.*$", re.IGNORECASE),
    re.compile(r"^More From .+:.+$", re.IGNORECASE),
]

_SENTENCE_END = re.compile(r"[.!?]")


def _remove_artifacts(text: str) -> str:
    """Remove known scraping artifacts from VLDBench articles."""
    text = re.sub(r"Trending Now\n(.+\n){1,5}", "", text)
    clean_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped in _ARTIFACT_LINES:
            continue
        if any(p.match(stripped) for p in _ARTIFACT_PATTERNS):
            continue
        clean_lines.append(line)
    cleaned = "\n".join(clean_lines)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _densify(text: str) -> str:
    paragraphs = re.split(r"\n{2,}", text)
    dense = []
    for p in paragraphs:
        # Replace newlines/tabs with space, then collapse multiple spaces
        collapsed = re.sub(r"[\n\t]+", " ", p)
        collapsed = re.sub(r" {2,}", " ", collapsed)
        dense.append(collapsed.strip())
    return "\n\n".join(p for p in dense if p)

def _truncate_at_sentence_boundary(text: str, max_chars: int = MAX_CHARS) -> str:
    """Truncate to max_chars at the nearest sentence boundary. Never mid-sentence."""
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    matches = list(_SENTENCE_END.finditer(window))
    if not matches:
        return window.rstrip()
    return text[:matches[-1].end()].strip()


def preprocess(text: str) -> str:
    text = _remove_artifacts(text)
    text = _densify(text)
    text = _remove_artifacts(text)   # second pass catches anything densify exposed
    text = _truncate_at_sentence_boundary(text)
    return text


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_segments(analysis: BiasAnalysis, article: str) -> BiasAnalysis:
    """Drop any segment whose 'original' is not a verbatim substring of the article."""
    valid = []
    for seg in analysis.biased_segments:
        if seg.original in article:
            valid.append(seg)
        else:
            logger.warning(
                "Dropping segment — 'original' not found verbatim in article: %r",
                seg.original[:80],
            )
    if len(valid) != len(analysis.biased_segments):
        return BiasAnalysis(
            binary_label=analysis.binary_label,
            severity=analysis.severity,
            bias_found=bool(valid),
            biased_segments=valid,
            unbiased_text=analysis.unbiased_text,
        )
    return analysis


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def load_checkpoint(output_path: Path) -> set[int]:
    """Return set of article indices already written to the checkpoint file."""
    if not output_path.exists():
        return set()
    completed: set[int] = set()
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                completed.add(record["index"])
            except (json.JSONDecodeError, KeyError):
                continue
    logger.info("Checkpoint: %d samples already completed", len(completed))
    return completed


def append_result(output_path: Path, record: dict) -> None:
    """Append a single result to the JSONL checkpoint file immediately."""
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def annotate_article(
    client: OpenAI,
    article: str,
    index: int,
) -> BiasAnalysis | None:
    """Annotate one article via chat completions with JSON mode + Pydantic validation."""
    delay = RETRY_BASE_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"ARTICLE:\n{article}"},
                ],
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content
            if not raw:
                raise ValueError("Empty response content")

            # Strip markdown fences if model adds them despite instructions
            raw = re.sub(r"^```json\s*", "", raw.strip())
            raw = re.sub(r"```$", "", raw.strip())

            result = BiasAnalysis.model_validate_json(raw)
            return validate_segments(result, article)

        except Exception as e:
            logger.warning(
                "Index %d | attempt %d/%d failed: %s",
                index, attempt, MAX_RETRIES, e,
            )
            if attempt < MAX_RETRIES:
                jitter = delay * 0.2
                sleep_for = delay + (jitter * (2 * (hash(str(e)) % 2) - 1))
                time.sleep(max(sleep_for, 1.0))
                delay *= 2
            else:
                logger.error("Index %d | all retries exhausted — skipping", index)
                return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate bias annotations for VLDBench articles using GPT-5.4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to VLDBench CSV file.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Path to JSONL checkpoint/output file. Safe to resume from.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Number of samples to process. Omit for full dataset. Set to 10 for smoke test.",
    )
    parser.add_argument(
        "--env", type=Path, default=None,
        help="Path to .env file containing OPENAI_API_KEY.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    args = parse_args()

    if args.env:
        load_dotenv(args.env)
    else:
        load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Pass --env or set in environment.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://proxy.vectorinstitute.ai/v1",
    )

    logger.info("Loading dataset from %s", args.input)
    df = pd.read_csv(args.input)
    logger.info("Total rows in dataset: %d", len(df))

    if args.num_samples is not None:
        df = df.head(args.num_samples)
        logger.info("Limiting to %d samples (--num-samples)", args.num_samples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(args.output)
    remaining = df[~df.index.isin(completed)]
    logger.info("Remaining to process: %d / %d", len(remaining), len(df))

    if remaining.empty:
        logger.info("All samples already completed. Nothing to do.")
        return

    success = skipped = 0

    for idx, row in remaining.iterrows():
        raw_article: str = str(row.get("article_text", ""))
        vldb_label: str = str(row.get("classification_label", "unknown"))

        if not raw_article.strip():
            logger.warning("Index %d | empty article — skipping", idx)
            skipped += 1
            continue

        clean_article = preprocess(raw_article)

        if len(clean_article) < 100:
            logger.warning(
                "Index %d | article too short after preprocessing (%d chars) — skipping",
                idx, len(clean_article),
            )
            skipped += 1
            continue

        logger.info("Index %d | processing (%d chars)...", idx, len(clean_article))

        analysis = annotate_article(client, clean_article, idx)

        if analysis is None:
            skipped += 1
            continue

        record = {
            "index": int(idx),
            "binary_label": vldb_label,
            "article_text": clean_article,
            "severity": analysis.severity,
            "bias_found": analysis.bias_found,
            "biased_segments": [
                {
                    "original": seg.original,
                    "replacement": seg.replacement,
                    "severity": seg.severity,
                    "bias_type": seg.bias_type,
                    "reasoning": seg.reasoning,
                }
                for seg in analysis.biased_segments
            ],
            "unbiased_text": analysis.unbiased_text,
        }

        append_result(args.output, record)
        success += 1

        logger.info(
            "Index %d | done — severity=%d, segments=%d",
            idx, analysis.severity, len(analysis.biased_segments),
        )

    logger.info(
        "\nCompleted: %d success, %d skipped. Output: %s",
        success, skipped, args.output,
    )


if __name__ == "__main__":
    main()