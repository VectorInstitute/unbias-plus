"""Data schemas for unbias-plus output."""

import logging
import re

from pydantic import BaseModel, field_validator


logger = logging.getLogger(__name__)

# Maps string severity labels the model may return for global severity
# to the correct integer scale (0, 2, 3, 4).
_STR_TO_INT_SEVERITY: dict[str, int] = {
    "none": 0,
    "low": 2,
    "medium": 3,
    "high": 4,
}


class BiasedSegment(BaseModel):
    """A single biased segment detected in the text.

    Attributes
    ----------
    original : str
        The original biased phrase from the input text.
    replacement : str
        The suggested neutral replacement. Defaults to empty string
        if the model omits it (e.g. under 4-bit quantization).
    severity : str
        Severity level: 'low', 'medium', or 'high'.
        Defaults to 'medium' if omitted by the model.
    bias_type : str
        Type of bias (e.g. 'loaded language', 'framing bias').
    reasoning : str
        Explanation of why this segment is considered biased.
    start : int | None
        Character offset start in the original text. Computed
        by the pipeline after parsing.
    end : int | None
        Character offset end in the original text. Computed
        by the pipeline after parsing.
    replacement_start : int | None
        Character offset start of ``replacement`` in ``unbiased_text``.
        Computed by the pipeline after parsing.
    replacement_end : int | None
        Character offset end of ``replacement`` in ``unbiased_text``.
        Computed by the pipeline after parsing.

    Examples
    --------
    >>> seg = BiasedSegment(
    ...     original="Sharia-obsessed fanatics",
    ...     replacement="extremist groups",
    ...     severity="high",
    ...     bias_type="dehumanizing framing",
    ...     reasoning="Uses inflammatory religious language.",
    ... )
    >>> seg.severity
    'high'

    """

    original: str
    replacement: str = ""  # optional — model may omit under 4-bit quantization
    severity: str = "medium"  # optional — defaults to medium if omitted
    bias_type: str = ""
    reasoning: str = ""
    start: int | None = None
    end: int | None = None
    replacement_start: int | None = None
    replacement_end: int | None = None

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        """Validate and normalise segment severity to low/medium/high."""
        allowed = {"low", "medium", "high"}
        normalized = v.lower().strip()
        if normalized not in allowed:
            logger.warning(
                "Unexpected segment severity '%s', defaulting to 'medium'", v
            )
            return "medium"
        return normalized


class BiasResult(BaseModel):
    """Full bias analysis result for an input text.

    Attributes
    ----------
    binary_label : str
        Overall label: 'biased' or 'unbiased'.
    severity : int
        Overall severity score:
          0 = neutral / no bias
          2 = recurring biased framing
          3 = strong persuasive tone
          4 = inflammatory rhetoric
        If the model returns a string ('low', 'medium', 'high'),
        it is coerced to the nearest integer value.
    bias_found : bool
        Whether any bias was detected in the text.
    biased_segments : list[BiasedSegment]
        List of biased segments found in the text, each with
        character-level start/end offsets.
    unbiased_text : str
        Full neutral rewrite of the input text.
    original_text : str | None
        The original input text. Set by the pipeline.

    Examples
    --------
    >>> result = BiasResult(
    ...     binary_label="biased",
    ...     severity=3,
    ...     bias_found=True,
    ...     biased_segments=[],
    ...     unbiased_text="A neutral version of the text.",
    ... )
    >>> result.binary_label
    'biased'

    """

    binary_label: str
    severity: int
    bias_found: bool
    biased_segments: list[BiasedSegment]
    unbiased_text: str
    original_text: str | None = None

    @field_validator("binary_label")
    @classmethod
    def validate_binary_label(cls, v: str) -> str:
        """Validate binary_label is 'biased' or 'unbiased'."""
        allowed = {"biased", "unbiased"}
        normalized = v.lower().strip()
        if normalized not in allowed:
            raise ValueError(f"binary_label must be one of {allowed}, got '{v}'")
        return normalized

    @field_validator("severity", mode="before")
    @classmethod
    def validate_severity(cls, v: int | str) -> int:
        """Coerce and validate global severity.

        Accepts:
          - int 0, 2, 3, 4  (correct model output)
          - str 'low', 'medium', 'high', 'none'  (model confused scales)
          - any other int   (clamped to nearest valid value)
        """
        # String coercion — model confused global vs segment severity scale
        if isinstance(v, str):
            normalized = v.lower().strip()
            if normalized in _STR_TO_INT_SEVERITY:
                coerced = _STR_TO_INT_SEVERITY[normalized]
                logger.warning(
                    "Global severity returned as string '%s', coerced to %d",
                    v,
                    coerced,
                )
                return coerced
            # Try parsing as int string e.g. "3"
            try:
                v = int(v)
            except ValueError:
                logger.warning("Unrecognized severity '%s', defaulting to 2", v)
                return 2

        # Clamp out-of-range integer values gracefully
        if v <= 0:
            return 0
        if v in {2, 3, 4}:
            return v
        if v == 1:
            return 2
        return 4  # anything > 4


def _find_case_insensitive(text: str, phrase: str, start: int = 0) -> int:
    return text.lower().find(phrase.lower(), start)


def _normalize_for_match(s: str) -> str:
    """Map curly quotes and dashes to ASCII.

    This keeps matching stable when model output typography differs from input.
    """
    return (
        s.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )


def _collapse_whitespace(s: str) -> str:
    return " ".join(s.split())


def _flexible_whitespace_pattern(phrase: str) -> re.Pattern[str] | None:
    """Build a regex that matches *phrase* with flexible internal whitespace."""
    tokens = phrase.split()
    if not tokens:
        return None
    return re.compile(r"\s+".join(re.escape(t) for t in tokens), re.IGNORECASE)


def _phrase_candidates(phrase: str) -> list[str]:
    """Variants to try in order before flexible-regex matching."""
    seen: set[str] = set()
    out: list[str] = []
    for cand in (
        phrase,
        phrase.strip(),
        _normalize_for_match(phrase),
        _normalize_for_match(phrase).strip(),
        _collapse_whitespace(phrase),
        _collapse_whitespace(_normalize_for_match(phrase)),
    ):
        if cand and cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out


def _find_span_in_text(
    haystack: str,
    phrase: str,
    cursor: int,
    *,
    label: str = "phrase",
    log_failure: bool = True,
) -> tuple[int, int] | None:
    """Return (start, end) in *haystack* at or after *cursor*, or None."""
    if not phrase:
        return None

    for cand in _phrase_candidates(phrase):
        start = _find_case_insensitive(haystack, cand, cursor)
        if start != -1:
            logger.debug("Matched %r via exact candidate %r at %d", phrase, cand, start)
            return (start, start + len(cand))

    pattern = _flexible_whitespace_pattern(phrase.strip())
    if pattern is not None:
        match = pattern.search(haystack, cursor)
        if match is not None:
            logger.debug(
                "Matched %r via flexible whitespace at %d", phrase, match.start()
            )
            return (match.start(), match.end())

    if log_failure:
        logger.warning(
            "Could not find %s in text (cursor=%d): %r", label, cursor, phrase
        )
    return None


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def _all_spans_in_text(
    haystack: str, phrase: str, *, label: str = "phrase"
) -> list[tuple[int, int]]:
    """Return every non-overlapping occurrence of *phrase* in *haystack*."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(haystack):
        span = _find_span_in_text(
            haystack, phrase, cursor, label=label, log_failure=False
        )
        if span is None:
            break
        spans.append(span)
        cursor = span[0] + 1
    return spans


def _assign_non_overlapping_spans(
    haystack: str,
    segments: list[BiasedSegment],
    phrase_attr: str,
    start_field: str,
    end_field: str,
    *,
    label: str,
) -> list[BiasedSegment]:
    """Map each segment to the first unused span of its phrase in *haystack*.

    Model segment order is not always left-to-right in the text. A linear cursor
    walk leaves later segments unmatched once the cursor reaches EOF.
    """
    used: list[tuple[int, int]] = []
    enriched: list[BiasedSegment] = []

    for seg in segments:
        phrase = getattr(seg, phrase_attr, "")
        if not phrase:
            enriched.append(seg)
            continue

        chosen: tuple[int, int] | None = None
        for span in _all_spans_in_text(haystack, phrase, label=label):
            if not any(_spans_overlap(span, u) for u in used):
                chosen = span
                break

        if chosen is None:
            logger.warning("Could not find %s in text: %r", label, phrase)
            enriched.append(seg)
            continue

        used.append(chosen)
        enriched.append(
            seg.model_copy(update={start_field: chosen[0], end_field: chosen[1]})
        )

    return enriched


def compute_offsets(
    original_text: str, segments: list[BiasedSegment]
) -> list[BiasedSegment]:
    """Compute character start/end offsets for each biased segment."""
    enriched = _assign_non_overlapping_spans(
        original_text,
        segments,
        "original",
        "start",
        "end",
        label="segment",
    )
    enriched.sort(key=lambda s: s.start if s.start is not None else 0)
    return deduplicate_by_span(enriched)


def compute_replacement_offsets(
    unbiased_text: str, segments: list[BiasedSegment]
) -> list[BiasedSegment]:
    """Compute character offsets for each replacement inside *unbiased_text*."""
    return _assign_non_overlapping_spans(
        unbiased_text,
        segments,
        "replacement",
        "replacement_start",
        "replacement_end",
        label="replacement",
    )


def deduplicate_by_span(segments: list[BiasedSegment]) -> list[BiasedSegment]:
    """Drop segments that share the same (start, end) after offset assignment.

    The parser already merges identical ``original`` strings; this catches
    near-duplicates (whitespace variants) that still map to the same span.
    """
    seen: set[tuple[int, int]] = set()
    unique: list[BiasedSegment] = []
    for seg in segments:
        if seg.start is None or seg.end is None:
            unique.append(seg)
            continue
        key = (seg.start, seg.end)
        if key in seen:
            continue
        seen.add(key)
        unique.append(seg)
    return unique
