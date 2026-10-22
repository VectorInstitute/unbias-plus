"""Normalize and repair the Unbias-plus SFT dataset before fine-tuning.

Two fixes, both reported in detail so changes can be reviewed before retraining:

  1. bias_type normalization — collapses the observed bias_type values down to
     the 7-type taxonomy used in prompts.py (casing drift, compound types,
     out-of-vocab labels, and the leaked global 'inflammatory rhetoric'
     severity label that ended up in per-segment bias_type).

  2. rewrite repair — for biased rows whose unbiased_text has diverged from a
     faithful application of their own biased_segments (original -> replacement),
     regenerate unbiased_text as that surgical substitution so the rewrite and
     the segments agree. Rows already consistent are left untouched. Unbiased
     rows (verbatim copies of the article) are never modified.

Input is never modified in place; a normalized copy + a text report are written.

Usage:
  python clean_dataset.py \
    --input  data/Unbias-plus-dataset.json \
    --output data/Unbias-plus-clean.json \
    --report data/clean_report.txt
"""

from __future__ import annotations

import argparse
import difflib
import json
from collections import Counter
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# 7-type taxonomy — MUST match prompts.py SFT_SYSTEM_PROMPT
# ---------------------------------------------------------------------------

CANONICAL = {
    "loaded language",
    "dehumanizing framing",
    "false generalizations",
    "framing bias",
    "euphemism/dysphemism",
    "politically charged terminology",
    "sensationalism",
}

# Explicit single-label aliases -> canonical. Compound types ("a / b") are
# resolved to their first-listed type BEFORE this map is consulted, so only
# single labels appear here. Judgment calls are commented — override freely.
ALIAS = {
    "false generalization": "false generalizations",
    "generalization": "false generalizations",
    "stereotyping": "false generalizations",  # sweeping claim about a group
    "euphemism": "euphemism/dysphemism",
    "dysphemism": "euphemism/dysphemism",
    "euphemisms": "euphemism/dysphemism",
    "euphemisms or dysphemisms": "euphemism/dysphemism",
    "editorializing": "framing bias",  # inserting opinion as fact
    "subjective language": "framing bias",
    "subjective claim": "framing bias",
    "interpretation bias": "framing bias",
    "promotional framing": "framing bias",
    "promotional content": "framing bias",
    "prejudiced framing": "framing bias",
    "speculation": "framing bias",
    "factual error": "framing bias",  # NOT a bias type — least-bad bucket (flagged)
    "labeling": "politically charged terminology",
    "stigmatizing terminology": "dehumanizing framing",  # demeaning a group
    "inflammatory rhetoric": "loaded language",  # global severity-4 label leaked in
    "metaphor": "loaded language",
}

# Substring fallback, checked in order, if a value is neither canonical nor
# in ALIAS. Keeps an unseen future variant from silently becoming the default.
KEYWORD_ORDER = [
    ("dehuman", "dehumanizing framing"),
    ("sensational", "sensationalism"),
    ("generaliz", "false generalizations"),
    ("stereotyp", "false generalizations"),
    ("euphemism", "euphemism/dysphemism"),
    ("dysphemism", "euphemism/dysphemism"),
    ("politic", "politically charged terminology"),
    ("terminolog", "politically charged terminology"),
    ("framing", "framing bias"),
    ("editorial", "framing bias"),
    ("loaded", "loaded language"),
]

DEFAULT_TYPE = "loaded language"  # most frequent; only if nothing else matches


def normalize_bias_type(raw: Any) -> tuple[str, str]:
    """Map a raw bias_type to a canonical type. Returns (canonical, how)."""
    if not isinstance(raw, str) or not raw.strip():
        return DEFAULT_TYPE, "empty->default"

    val = raw.strip().lower()
    compound = " / " in val  # canonical "euphemism/dysphemism" has no spaces
    if compound:
        val = val.split(" / ", 1)[0].strip()

    if val in CANONICAL:
        return val, "compound->primary" if compound else "exact"
    if val in ALIAS:
        return ALIAS[val], "alias(compound)" if compound else "alias"
    for needle, canon in KEYWORD_ORDER:
        if needle in val:
            return canon, "keyword"
    return DEFAULT_TYPE, "UNMAPPED->default"


# ---------------------------------------------------------------------------
# Rewrite repair
# ---------------------------------------------------------------------------


def surgical_rewrite(row: dict[str, Any]) -> str:
    """Rebuild unbiased_text by applying each segment's original->replacement."""
    text: str = str(row["article_text"])
    for seg in row["biased_segments"]:
        original = seg.get("original", "")
        replacement = seg.get("replacement", "")
        if original:
            text = text.replace(original, replacement)
    return text


def similarity(a: str, b: str) -> float:
    """Char-level similarity ratio in [0, 1]."""
    return difflib.SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def clean(
    data: list[dict[str, Any]], rewrite_threshold: float, repair: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize bias_type and (optionally) repair divergent rewrites.

    Returns the cleaned data and a stats dict for the report.
    """
    raw_type_counts: Counter[str] = Counter()
    canon_type_counts: Counter[str] = Counter()
    how_counts: Counter[str] = Counter()
    mapping: dict[str, tuple[str, str]] = {}  # raw -> (canonical, how)
    unmapped: set[str] = set()

    repaired_idx: list[int] = []

    for row in data:
        # --- 1. bias_type normalization (biased rows only have segments) ---
        for seg in row["biased_segments"]:
            raw = seg.get("bias_type", "")
            raw_type_counts[raw if isinstance(raw, str) else str(raw)] += 1
            canon, how = normalize_bias_type(raw)
            seg["bias_type"] = canon
            canon_type_counts[canon] += 1
            how_counts[how] += 1
            if isinstance(raw, str):
                mapping[raw] = (canon, how)
            if how == "UNMAPPED->default":
                unmapped.add(str(raw))

        # --- 2. rewrite repair (biased rows only) ---
        if repair and row["binary_label"] == "biased" and row["biased_segments"]:
            rebuilt = surgical_rewrite(row).strip()
            current = (row.get("unbiased_text") or "").strip()
            if similarity(rebuilt, current) <= rewrite_threshold:
                row["unbiased_text"] = rebuilt
                repaired_idx.append(row.get("index", -1))

    stats = {
        "raw_type_counts": raw_type_counts,
        "canon_type_counts": canon_type_counts,
        "how_counts": how_counts,
        "mapping": mapping,
        "unmapped": unmapped,
        "repaired_idx": repaired_idx,
    }
    return data, stats


def validate(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Post-clean checks: taxonomy compliance + segment/rewrite consistency."""
    bad_type = 0
    repl_total = repl_in = 0
    for row in data:
        for seg in row["biased_segments"]:
            if seg.get("bias_type") not in CANONICAL:
                bad_type += 1
            repl = seg.get("replacement", "")
            repl_total += 1
            if repl and repl in (row.get("unbiased_text") or ""):
                repl_in += 1
    return {
        "segments_off_taxonomy": bad_type,
        "replacement_in_unbiased_pct": (100 * repl_in / repl_total)
        if repl_total
        else 0.0,
    }


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def _format_mapping_lines(
    raw_type_counts: Counter[str],
    mapping: dict[str, tuple[str, str]],
) -> list[str]:
    """Format the raw->canonical mapping lines for the report."""
    lines = []
    for rawval, c in raw_type_counts.most_common():
        mapped, how = mapping.get(rawval, (DEFAULT_TYPE, "?"))
        flag = "   <-- REVIEW" if how == "UNMAPPED->default" else ""
        shown = repr(rawval) if rawval != mapped else f"'{rawval}'"
        lines.append(f"  {shown:42s} -> {mapped:32s} [{how}] x{c}{flag}")
    return lines


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_report(  # noqa: PLR0915
    n_rows: int,
    stats: dict[str, Any],
    post: dict[str, Any],
    rewrite_threshold: float,
    repair: bool,
) -> str:
    """Build a human-readable text report of the cleanup run."""
    lines: list[str] = []
    add = lines.append

    add("=" * 70)
    add("UNBIAS-PLUS DATASET CLEANUP REPORT")
    add("=" * 70)
    add(f"Rows processed: {n_rows}")
    add("")

    # --- bias_type ---
    add("-" * 70)
    add("1. BIAS_TYPE NORMALIZATION")
    add("-" * 70)
    raw = stats["raw_type_counts"]
    canon = stats["canon_type_counts"]
    add(f"Distinct raw bias_type values : {len(raw)}")
    add(f"Distinct after normalization  : {len(canon)}  (target: 7)")
    add("")
    add("How each segment was mapped:")
    for how, c in stats["how_counts"].most_common():
        add(f"  {how:22s} {c:6d}")
    add("")
    add("Canonical distribution after cleanup:")
    for t, c in canon.most_common():
        add(f"  {t:34s} {c:6d}")
    add("")
    add("Full mapping (raw value  ->  canonical  [how]  x count):")
    lines.extend(_format_mapping_lines(raw, stats["mapping"]))
    if stats["unmapped"]:
        add("")
        add(f"!! UNMAPPED values forced to '{DEFAULT_TYPE}' (review these):")
        for u in sorted(stats["unmapped"]):
            add(f"     {u!r}")
    add("")

    # --- rewrites ---
    add("-" * 70)
    add("2. REWRITE REPAIR")
    add("-" * 70)
    if not repair:
        add("Skipped (--no-repair).")
    else:
        idx = stats["repaired_idx"]
        add(f"Divergence threshold      : similarity <= {rewrite_threshold:.2f}")
        add(f"Biased rows repaired      : {len(idx)}")
        add("  (unbiased_text rebuilt as the faithful original->replacement swap)")
        if idx:
            preview = ", ".join(str(i) for i in idx[:25])
            more = f" ... (+{len(idx) - 25} more)" if len(idx) > 25 else ""
            add(f"  repaired row indices    : {preview}{more}")
    add("")

    # --- post-clean validation ---
    add("-" * 70)
    add("3. POST-CLEAN VALIDATION")
    add("-" * 70)
    add(
        f"Segments still off-taxonomy        : {post['segments_off_taxonomy']}  (want 0)"
    )
    add(
        f"Replacements present in unbiased   : {post['replacement_in_unbiased_pct']:.1f}%"
    )
    add("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Normalize bias_type and repair divergent rewrites.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=Path, required=True, help="Source dataset JSON.")
    p.add_argument("--output", type=Path, required=True, help="Cleaned dataset JSON.")
    p.add_argument(
        "--report", type=Path, default=None, help="Optional text report path."
    )
    p.add_argument(
        "--rewrite-threshold",
        type=float,
        default=0.95,
        help="Repair biased rows whose unbiased_text is <= this similar to the "
        "faithful substitution.",
    )
    p.add_argument(
        "--no-repair",
        action="store_true",
        help="Only normalize bias_type; leave all unbiased_text untouched.",
    )
    return p.parse_args()


def main() -> None:
    """Run the full dataset cleanup pipeline."""
    args = parse_args()

    with args.input.open(encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)

    cleaned, stats = clean(
        data, rewrite_threshold=args.rewrite_threshold, repair=not args.no_repair
    )
    post = validate(cleaned)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    report = build_report(
        len(data), stats, post, args.rewrite_threshold, not args.no_repair
    )
    print(report)
    if args.report:
        args.report.write_text(report + "\n", encoding="utf-8")

    print(f"\nCleaned dataset written to: {args.output}")
    if args.report:
        print(f"Report written to        : {args.report}")


if __name__ == "__main__":
    main()
