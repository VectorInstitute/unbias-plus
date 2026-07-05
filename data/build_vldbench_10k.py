#!/usr/bin/env python
"""Build a 10k article_text subset from VLDBench.

Pipeline (cheap filters first, expensive ones last):
  1. ads & boilerplate removal (line + inline regex), quote stripping,
     whitespace cleaning
  2. length filter 100-1500 words
  3. English-only (langdetect)
  4. exact dedupe
  5. take first 10k that survive

Run ``python build_vldbench_10k.py --smoke`` for a quick end-to-end check
(~300 rows scanned, 20 kept, written to separate ``vldbench_smoke.*`` files).
"""

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median

from datasets import load_dataset
from langdetect import DetectorFactory, LangDetectException, detect


DetectorFactory.seed = 0  # deterministic language detection

TARGET = 10_000
MIN_WORDS, MAX_WORDS = 100, 1500
# outputs land next to this script; override with VLDBENCH_OUT_DIR
OUT_DIR = Path(os.environ.get("VLDBENCH_OUT_DIR", Path(__file__).resolve().parent))

# --- ads / boilerplate patterns ----------------------------------------------
# Whole-line junk: if a line matches, drop the whole line.
LINE_PATTERNS = [
    r"^\s*adverti[sz]e?ment\.?\s*$",
    r"^\s*sponsored( content| by .*)?\.?\s*$",
    r"^\s*(story )?continues? below( advertisement)?\.?\s*$",
    r"^\s*(article )?continues? after (the )?adverti[sz]e?ment\.?\s*$",
    r"^\s*(read|see) ?more\b.*$",
    r"^\s*(related|related stories?|related articles?|more|also read|read also|recommended|trending|most read|editor'?s picks?)\s*:?.*$",
    r"^\s*sign up (for|to).*newsletter.*$",
    r"^\s*subscribe( to| now| today).*$",
    r"^\s*(get|join) .*(newsletter|our free).*$",
    r"^\s*follow (us|@|.*on (twitter|x|facebook|instagram|tiktok|linkedin|youtube)).*$",
    r"^\s*(share|tweet|pin) this.*$",
    r"^\s*(click|tap) here.*$",
    r"^\s*(watch|video|listen|photos?|gallery|slideshow)\s*:.*$",
    r"^\s*(photo|image|picture|credit|source)\s*:.*$",
    r"^\s*download (the|our) app.*$",
    r"^\s*(this (site|website) uses cookies|by continuing to (use|browse)|we use cookies|accept (all )?cookies).*$",
    r"^\s*(copyright|©|\(c\))\s*\d{0,4}.*$",
    r"^\s*all rights reserved\.?.*$",
    r"^\s*(leave a comment|comments?|view comments)\s*$",
    r"^\s*(contact us|email us|reach (us|out)).*$",
    r"^\s*https?://\S+\s*$",  # bare URL-only line
    r"^\s*(reuters|ap|afp|getty images|associated press|bloomberg)\s*$",
    r"^\s*we may (earn|receive) a commission.*$",
    r"^\s*.*(affiliate links?|at no (extra|additional) cost to you|purchases? (made )?through (our|these) links).*$",
    r"^\s*we (independently|may) (review|research|test|earn).*$",
]
LINE_RES = [re.compile(p, re.IGNORECASE) for p in LINE_PATTERNS]
LINE_RE = re.compile("|".join(f"(?:{p})" for p in LINE_PATTERNS), re.IGNORECASE)

# Inline junk removed anywhere in the text (case-insensitive).
# NOTE: kept conservative so legitimate mentions survive: only the standalone
# token "advertisement" is stripped (usually an injected ad marker); derived
# forms like "advertisements" (e.g. "purchase television advertisements",
# "Advertisements Act") survive via the word boundary, as do plain sentences
# that merely mention subscribing/ads. CTA patterns target full promo sentences.
INLINE_PATTERNS = [
    r"\bADVERTI[SZ]E?MENT\b",
    r"\bStory continues below advertisement\b",
    r"\(Photo:[^)]*\)",
    r"\(Image:[^)]*\)",
    r"\(Reporting by[^)]*\)",
    r"\(Additional reporting by[^)]*\)",
    r"\(Editing by[^)]*\)",
    r"\bGetty Images\b",
    # high-confidence promo / call-to-action sentences that survive as inline text
    r"\bSubscribe to [^.?!]{0,80}?(?:newsletter|here|today|Pro)\b[^.?!]*[.?!]",
    r"\bSign up (?:for|to)[^.?!]{0,80}?newsletter[^.?!]*[.?!]",
    r"\bWant to (?:receive|read|get)[^.?!]{0,80}?(?:newsletter|this article)[^.?!]*[.?!]",
    r"\bSupport journalism like this[^.?!]*[.?!]",
    r"\bClick here to [^.?!]{0,80}?[.?!]",
    r"\b(?:Follow|Reach) (?:us|the author|[A-Z][a-z]+) on (?:Twitter|X|Facebook|Instagram|TikTok|LinkedIn)[^.?!]*[.?!]",
    r"\bThe [A-Z][A-Za-z ]{0,40} is a (?:snappy|quick|daily)[^.?!]*roundup[^.?!]*[.?!]",
]
INLINE_RE = re.compile("|".join(f"(?:{p})" for p in INLINE_PATTERNS), re.IGNORECASE)

# Boilerplate discovered during a post-run audit of the 10k output. These
# fragments only become contiguous after newline collapsing (site footers
# split across lines), so they are scrubbed after ``clean_whitespace``.
# Deliberately case-sensitive: the caps variants are unambiguous CTA markers,
# while e.g. a lowercase "click here for more" sentence could be quoted prose.
POST_COLLAPSE_PATTERNS = [
    r"CLICK HERE FOR MORE ENTERTAINMENT NEWS",
    r"APP USERS CLICK HERE (?:TO VIEW|FOR) POST",
    r"Upcoming Webinar Join us on .*$",
    r"©\d{4} Nikkei Inc\. All rights reserved\.",
]
POST_COLLAPSE_RE = re.compile("|".join(f"(?:{p})" for p in POST_COLLAPSE_PATTERNS))

# Every straight/curly/angled single- and double-quote character. Stripped
# (replaced with nothing, not a space) so contractions collapse to one token
# ("don't" -> "dont") instead of splitting in two.
QUOTES_RE = re.compile(r"[\"'`´“”‘’„‟‚‛«»‹›]")

WS_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"\b\w+\b")


def strip_ads(text: str, audit: "Counter[str] | None" = None) -> str:
    """Drop boilerplate/ad lines and scrub inline promo fragments.

    When ``audit`` is given, it accumulates the number of lines dropped per
    line pattern (keyed by the pattern) and inline substitutions (keyed
    ``"<inline>"``), so the regexes' real-world hit rates can be reviewed.
    """
    kept = []
    for ln in re.split(r"[\r\n]+", text):
        if not ln.strip():
            continue
        if LINE_RE.match(ln):
            if audit is not None:
                audit[_first_matching_line_pattern(ln)] += 1
            continue
        kept.append(ln)
    out, n_inline = INLINE_RE.subn(" ", "\n".join(kept))
    if audit is not None and n_inline:
        audit["<inline>"] += n_inline
    return out


def _first_matching_line_pattern(line: str) -> str:
    """Return the first LINE_PATTERNS entry that matches *line* (for audit)."""
    for pat in LINE_RES:
        if pat.match(line):
            return pat.pattern
    return "<unknown>"


def strip_quotes(text: str, audit: "Counter[str] | None" = None) -> str:
    """Remove all quote characters (see ``QUOTES_RE``).

    Runs after ``strip_ads`` so the ad regexes see the original text, and
    before ``clean_whitespace`` so any leftover space runs get collapsed.
    """
    out, n = QUOTES_RE.subn("", text)
    if audit is not None and n:
        audit["<quotes>"] += n
    return out


def scrub_post_collapse(text: str, audit: "Counter[str] | None" = None) -> str:
    """Remove footer boilerplate that is only contiguous after collapsing."""
    out, n = POST_COLLAPSE_RE.subn(" ", text)
    if audit is not None and n:
        audit["<post-collapse>"] += n
    return WS_RE.sub(" ", out).strip() if n else text


def clean_whitespace(text: str) -> str:
    """Collapse all whitespace runs (incl. newlines/tabs/nbsp) to single spaces."""
    text = text.replace(" ", " ")
    return WS_RE.sub(" ", text).strip()


def count_words(text: str) -> int:
    """Count word tokens only — punctuation, symbols and URL glue don't count."""
    return len(WORD_RE.findall(text))


def norm_key(text: str) -> str:
    """Return a case-insensitive content hash used for exact dedupe.

    VLDBench is already deduplicated upstream (near-duplicates, syndicated
    copies and updated versions were handled at dataset construction), so a
    cheap exact hash is sufficient here — no MinHash/SimHash needed.
    """
    return hashlib.md5(text.lower().encode("utf-8")).hexdigest()


def reject_reason(text: str, nwords: int, key: str, seen: set[str]) -> str | None:
    """Return the stats key describing why *text* is rejected, or None to keep.

    Ordered cheapest-first: word-count bounds, then language detection
    (comparatively expensive), then the dedupe lookup.
    """
    if nwords < MIN_WORDS:
        return "too_short"
    if nwords > MAX_WORDS:
        return "too_long"
    try:
        if detect(text) != "en":
            return "non_en"
    except LangDetectException:
        return "lang_err"
    if key in seen:
        return "dup"
    return None


def main(
    target: int = TARGET, max_rows: int | None = None, stem: str = "vldbench_10k"
) -> None:
    """Stream VLDBench, apply the cleaning pipeline, and write ``<stem>.jsonl``."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"{stem}.jsonl"
    stats_file = OUT_DIR / f"{stem}.stats.json"

    ds = load_dataset("vector-institute/VLDBench", split="train", streaming=True)
    ds = ds.select_columns(["unique_id", "article_text"])  # skip image decode

    seen: set[str] = set()
    audit: Counter[str] = Counter()
    kept_lengths: list[int] = []
    stats = {
        "seen_rows": 0,
        "empty": 0,
        "too_short": 0,
        "too_long": 0,
        "non_en": 0,
        "lang_err": 0,
        "dup": 0,
        "kept": 0,
    }

    with open(out_file, "w", encoding="utf-8") as fout:
        for row in ds:
            if max_rows is not None and stats["seen_rows"] >= max_rows:
                break
            stats["seen_rows"] += 1
            raw = row.get("article_text")
            if not raw or not isinstance(raw, str):
                stats["empty"] += 1
                continue

            text = scrub_post_collapse(
                clean_whitespace(strip_quotes(strip_ads(raw, audit), audit)), audit
            )
            if not text:
                stats["empty"] += 1
                continue

            nwords = count_words(text)
            key = norm_key(text)
            reason = reject_reason(text, nwords, key, seen)
            if reason:
                stats[reason] += 1
                continue

            seen.add(key)
            kept_lengths.append(nwords)
            record = {
                "unique_id": row.get("unique_id"),
                "article_text": text,
                "word_count": nwords,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["kept"] += 1
            if stats["kept"] % 500 == 0:
                print(f"  kept={stats['kept']}  seen={stats['seen_rows']}", flush=True)
            if stats["kept"] >= target:
                break

    report: dict[str, object] = dict(stats)
    if kept_lengths:
        report["kept_word_count_summary"] = {
            "min": min(kept_lengths),
            "mean": round(mean(kept_lengths), 1),
            "median": median(kept_lengths),
            "max": max(kept_lengths),
        }
    # per-pattern hit counts, most frequent first — sanity check that no
    # pattern is over-triggering on legitimate article content
    report["boilerplate_removals"] = dict(audit.most_common())

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {stats['kept']} rows -> {out_file}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="quick end-to-end run (300 rows scanned, 20 kept, vldbench_smoke.* outputs)",
    )
    parser.add_argument(
        "--target", type=int, default=None, help="articles to keep (default 10000)"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="stop after scanning this many source rows",
    )
    parser.add_argument(
        "--stem", default=None, help="output filename stem (default vldbench_10k)"
    )
    args = parser.parse_args()
    if args.smoke:
        main(
            target=args.target or 20,
            max_rows=args.max_rows or 300,
            stem=args.stem or "vldbench_smoke",
        )
    else:
        main(
            target=args.target or TARGET,
            max_rows=args.max_rows,
            stem=args.stem or "vldbench_10k",
        )
    # All outputs are written and closed at this point. Skip interpreter
    # finalization: tearing down the HF streaming iterator's pyarrow/torch
    # threads at shutdown aborts with a GIL error in some environments,
    # which would turn a fully successful run into exit code 134.
    os._exit(0)
