"""Smoke tests for the pure helpers in ``data/build_vldbench_10k.py``.

Everything here runs offline: only the text-cleaning and filtering helpers
are exercised, never the VLDBench stream. For a live end-to-end smoke run
use ``python data/build_vldbench_10k.py --smoke``.
"""

from __future__ import annotations

from collections import Counter

from build_vldbench_10k import (
    MAX_WORDS,
    MIN_WORDS,
    clean_whitespace,
    count_words,
    norm_key,
    reject_reason,
    scrub_post_collapse,
    strip_ads,
)


# A clearly-English paragraph comfortably above MIN_WORDS once repeated.
_EN_SENTENCE = (
    "The city council voted on Tuesday to approve the new transit plan, "
    "which includes additional bus routes, expanded subway service, and "
    "dedicated cycling lanes across several downtown neighbourhoods. "
)
EN_TEXT = _EN_SENTENCE * 8  # ~200 words

# A clearly-French paragraph of comparable length.
_FR_SENTENCE = (
    "Le conseil municipal a voté mardi pour approuver le nouveau plan de "
    "transport, qui comprend des lignes de bus supplémentaires, un service "
    "de métro étendu et des pistes cyclables dans plusieurs quartiers. "
)
FR_TEXT = _FR_SENTENCE * 8


def test_strip_ads_drops_boilerplate_lines() -> None:
    text = "\n".join(
        [
            "The mayor announced the budget today.",
            "ADVERTISEMENT",
            "Subscribe to our newsletter today!",
            "Council members debated for three hours.",
        ],
    )
    audit: Counter[str] = Counter()
    out = strip_ads(text, audit)
    assert "ADVERTISEMENT" not in out
    assert "Subscribe" not in out
    assert "mayor announced" in out
    assert "debated for three hours" in out
    assert sum(audit.values()) == 2


def test_strip_ads_keeps_legitimate_ad_mentions() -> None:
    text = "The REAL Political Advertisements Act would require disclaimers."
    assert strip_ads(text) == text


def test_clean_whitespace_collapses_everything() -> None:
    assert clean_whitespace("a\r\n\nb\tc  d e ") == "a b c d e"


def test_count_words_ignores_punctuation_and_url_glue() -> None:
    # split() would count 4 tokens here; word tokens are what we filter on
    assert count_words("Read this: https://ex.com/a-b now") == 8
    assert count_words("Hello, world!") == 2


def test_norm_key_is_case_insensitive() -> None:
    assert norm_key("Some Article Text") == norm_key("some article text")


def test_reject_reason_length_bounds_run_before_detection() -> None:
    seen: set[str] = set()
    short = "too short"
    assert (
        reject_reason(short, count_words(short), norm_key(short), seen) == "too_short"
    )
    long_text = "word " * (MAX_WORDS + 1)
    assert (
        reject_reason(long_text, MAX_WORDS + 1, norm_key(long_text), seen) == "too_long"
    )
    assert count_words(EN_TEXT) >= MIN_WORDS  # fixtures must clear the bound


def test_reject_reason_language_and_dedupe() -> None:
    seen: set[str] = set()
    key = norm_key(EN_TEXT)
    assert reject_reason(EN_TEXT, count_words(EN_TEXT), key, seen) is None
    seen.add(key)
    assert reject_reason(EN_TEXT, count_words(EN_TEXT), key, seen) == "dup"
    fr_key = norm_key(FR_TEXT)
    assert reject_reason(FR_TEXT, count_words(FR_TEXT), fr_key, seen) == "non_en"


def test_scrub_post_collapse_removes_footer_ctas() -> None:
    audit: Counter[str] = Counter()
    text = 'She said no more. CLICK HERE FOR MORE ENTERTAINMENT NEWS "It was fine."'
    out = scrub_post_collapse(text, audit)
    assert "CLICK HERE" not in out
    assert out.startswith("She said no more.")
    assert audit["<post-collapse>"] == 1
    # lowercase variants could be quoted prose — deliberately untouched
    prose = "he told fans to click here for more entertainment news yesterday"
    assert scrub_post_collapse(prose) == prose
