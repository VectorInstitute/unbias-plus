#!/usr/bin/env python
"""Build a 10k article_text subset from VLDBench.

Pipeline:
  1. clean whitespace (remove newlines / collapse spaces)
  2. ads & boilerplate removal (line + inline regex)
  3. exact dedupe
  4. English-only (langdetect)
  5. length filter 100-1500 words
  6. take first 10k that survive
"""

import hashlib
import json
import os
import re
from pathlib import Path

from datasets import load_dataset
from langdetect import DetectorFactory, LangDetectException, detect


DetectorFactory.seed = 0  # deterministic language detection

TARGET = 10_000
MIN_WORDS, MAX_WORDS = 100, 1500
# outputs land next to this script; override with VLDBENCH_OUT_DIR
OUT_DIR = Path(os.environ.get("VLDBENCH_OUT_DIR", Path(__file__).resolve().parent))
OUT_FILE = OUT_DIR / "vldbench_10k.jsonl"
STATS_FILE = OUT_DIR / "vldbench_10k.stats.json"

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
    # boilerplate discovered during post-run audit of the 10k output
    r"CLICK HERE FOR MORE ENTERTAINMENT NEWS",
    r"APP USERS CLICK HERE (?:TO VIEW|FOR) POST",
    r"Upcoming Webinar Join us on .*$",
    r"©\d{4} Nikkei Inc\. All rights reserved\.",
]
INLINE_RE = re.compile("|".join(f"(?:{p})" for p in INLINE_PATTERNS), re.IGNORECASE)

WS_RE = re.compile(r"\s+")


def strip_ads(text: str) -> str:
    """Drop boilerplate/ad lines and scrub inline promo fragments."""
    lines = re.split(r"[\r\n]+", text)
    kept = [ln for ln in lines if ln.strip() and not LINE_RE.match(ln)]
    return INLINE_RE.sub(" ", "\n".join(kept))


def clean_whitespace(text: str) -> str:
    """Collapse all whitespace runs (incl. newlines/tabs/nbsp) to single spaces."""
    text = text.replace(" ", " ")
    return WS_RE.sub(" ", text).strip()


def norm_key(text: str) -> str:
    """Return a case-insensitive content hash used for exact dedupe."""
    return hashlib.md5(text.lower().encode("utf-8")).hexdigest()


def main() -> None:
    """Stream VLDBench, apply the cleaning pipeline, and write the 10k JSONL."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("vector-institute/VLDBench", split="train", streaming=True)
    ds = ds.select_columns(["unique_id", "article_text"])  # skip image decode

    seen: set[str] = set()
    kept = 0
    stats = {
        "seen_rows": 0,
        "empty": 0,
        "dup": 0,
        "non_en": 0,
        "too_short": 0,
        "too_long": 0,
        "lang_err": 0,
        "kept": 0,
    }

    with open(OUT_FILE, "w", encoding="utf-8") as fout:
        for row in ds:
            stats["seen_rows"] += 1
            raw = row.get("article_text")
            if not raw or not isinstance(raw, str):
                stats["empty"] += 1
                continue

            # 1+2: ads removal (needs line structure), then whitespace cleaning
            text = clean_whitespace(strip_ads(raw))
            if not text:
                stats["empty"] += 1
                continue

            # 3: exact dedupe
            key = norm_key(text)
            if key in seen:
                stats["dup"] += 1
                continue

            # 4: English-only
            try:
                if detect(text) != "en":
                    stats["non_en"] += 1
                    continue
            except LangDetectException:
                stats["lang_err"] += 1
                continue

            # 5: length filter 100-1500 words
            nwords = len(text.split())
            if nwords < MIN_WORDS:
                stats["too_short"] += 1
                continue
            if nwords > MAX_WORDS:
                stats["too_long"] += 1
                continue

            seen.add(key)
            record = {
                "unique_id": row.get("unique_id"),
                "article_text": text,
                "word_count": nwords,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
            stats["kept"] = kept
            if kept % 500 == 0:
                print(f"  kept={kept}  seen={stats['seen_rows']}", flush=True)
            if kept >= TARGET:
                break

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")
    print(f"Wrote {kept} rows -> {OUT_FILE}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
