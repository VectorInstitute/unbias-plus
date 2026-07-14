"""Data schemas for unbias-plus output."""

import difflib
import logging
import re

from pydantic import BaseModel, field_validator


logger = logging.getLogger(__name__)

# Maps string severity labels the model may return for global severity
# onto the integer scale (0-10).
_STR_TO_INT_SEVERITY: dict[str, int] = {
    "none": 0,
    "low": 3,
    "medium": 5,
    "high": 8,
}

# Canonical bias type identifiers used by the model.
VALID_BIAS_TYPES = frozenset(
    {
        "loaded_language",
        "euphemism",
        "dehumanizing_language",
        "opinion_as_fact",
        "unsupported_generalization",
        "stereotypical_association",
        "sensationalism",
        "informational_bias",
    }
)


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
        Severity level: 'low', 'medium', or 'high' (normalized lowercase
        for API/UI; model may emit 'Low' | 'Medium' | 'High').
    bias_type : str
        Type of bias (e.g. ``loaded_language``, ``stereotypical_association``).
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
    ...     original="flood of migrants",
    ...     replacement="arrival of migrants",
    ...     severity="High",
    ...     bias_type="dehumanizing_language",
    ...     reasoning="Treats people as a threatening mass.",
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

    @field_validator("severity", mode="before")
    @classmethod
    def validate_severity(cls, v: object) -> str:
        """Validate and normalise segment severity to low/medium/high.

        Accepts:
          - str 'low' | 'medium' | 'high'  (correct model output)
          - int 0-10  (model confused segment vs. global severity scale;
            bucketed the same way the global score is bucketed elsewhere)
          - anything else — defaults to 'medium'
        """
        allowed = {"low", "medium", "high"}
        if isinstance(v, str):
            normalized = v.lower().strip()
            if normalized in allowed:
                return normalized
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            logger.warning(
                "Segment severity returned as int '%s', coerced by bucket", v
            )
            if v >= 6:
                return "high"
            if v >= 3:
                return "medium"
            return "low"
        logger.warning("Unexpected segment severity '%s', defaulting to 'medium'", v)
        return "medium"


class BiasResult(BaseModel):
    """Full bias analysis result for an input text.

    Attributes
    ----------
    binary_label : str
        Overall label: 'biased' or 'unbiased'. Derived from severity
        when the model omits it.
    severity : int
        Overall severity score:
          0      = no bias
          1-5    = limited / low / moderate bias
          6-10   = strong / recurring / highly distorting bias
        If the model returns a string ('low', 'medium', 'high'),
        it is coerced to a nearby integer on this scale.
    bias_found : bool
        Whether any bias was detected in the text. Derived from
        severity / segments when the model omits it.
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
    ...     severity=6,
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
        """Coerce and validate global severity on the 0-10 scale.

        Accepts:
          - int 0-10  (correct model output)
          - str 'low', 'medium', 'high', 'none'  (model confused scales)
          - any other int   (clamped into 0-10)
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
                logger.warning("Unrecognized severity '%s', defaulting to 3", v)
                return 3

        # Clamp out-of-range integer values gracefully onto 0-10
        if v < 0:
            return 0
        if v > 10:
            return 10
        return int(v)


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
    """Variants to try in order before flexible-regex matching.

    Prefer stripped forms first so a model ``original`` with a leading/trailing
    space does not pull the preceding whitespace into the highlight span.
    """
    seen: set[str] = set()
    out: list[str] = []
    for cand in (
        phrase.strip(),
        _normalize_for_match(phrase).strip(),
        _collapse_whitespace(phrase),
        _collapse_whitespace(_normalize_for_match(phrase)),
        phrase,
        _normalize_for_match(phrase),
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


def _word_start(text: str, pos: int) -> int:
    pos = min(max(pos, 0), len(text))
    while pos > 0 and not text[pos - 1].isspace():
        pos -= 1
    return pos


def _word_end(text: str, pos: int) -> int:
    pos = min(max(pos, 0), len(text))
    while pos < len(text) and not text[pos].isspace():
        pos += 1
    return pos


def _build_orig_to_unb_map(original_text: str, unbiased_text: str) -> list[int]:
    """Map each original index (plus end sentinel) to an unbiased index."""
    matcher = difflib.SequenceMatcher(
        None, original_text, unbiased_text, autojunk=False
    )
    n = len(original_text)
    mapping = [0] * (n + 1)
    mapping[n] = len(unbiased_text)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                mapping[i1 + k] = j1 + k
            if i2 <= n:
                mapping[i2] = j2
        elif tag == "replace":
            olen = i2 - i1
            ulen = j2 - j1
            for k in range(olen):
                mapping[i1 + k] = j1 + (k * ulen // olen if olen else 0)
            if i2 <= n:
                mapping[i2] = j2
        elif tag == "delete":
            for k in range(i1, i2):
                mapping[k] = j1
            if i2 <= n:
                mapping[i2] = j1
        elif tag == "insert":
            if i1 <= n:
                mapping[i1] = j1

    return mapping


def _find_replacement_span_in_unbiased(
    unbiased_text: str,
    replacement: str,
    *,
    cursor: int = 0,
) -> tuple[int, int] | None:
    """Locate *replacement* in *unbiased_text* with flexible matching."""
    if not replacement:
        return None
    span = _find_span_in_text(
        unbiased_text, replacement, cursor, label="replacement", log_failure=False
    )
    if span is not None:
        return span

    tokens = replacement.split()
    drop = {"were", "was", "is", "are", "a", "an", "the"}
    for idx in range(len(tokens)):
        reduced = " ".join(tokens[idx:])
        if reduced:
            span = _find_span_in_text(
                unbiased_text,
                reduced,
                cursor,
                label="replacement",
                log_failure=False,
            )
            if span is not None:
                return span
    reduced = " ".join(t for t in tokens if t.lower() not in drop)
    if reduced and reduced != replacement:
        span = _find_span_in_text(
            unbiased_text,
            reduced,
            cursor,
            label="replacement",
            log_failure=False,
        )
        if span is not None:
            return span
    return None


def _boundary_replacement_span(
    original_text: str,
    unbiased_text: str,
    seg_start: int,
    seg_end: int,
    orig_to_unb: list[int],
) -> tuple[int, int] | None:
    """Map a biased segment's original span to the full rewrite region."""
    u_start = _word_start(unbiased_text, orig_to_unb[seg_start])
    u_end = _word_end(unbiased_text, orig_to_unb[seg_end])
    if u_end <= u_start:
        return None

    orig_len = seg_end - seg_start
    unb_len = u_end - u_start
    if orig_len > 0 and unb_len > orig_len * 3 + 40:
        return None
    return (u_start, u_end)


def compute_replacement_offsets(
    original_text: str,
    unbiased_text: str,
    segments: list[BiasedSegment],
) -> list[BiasedSegment]:
    """Compute highlight spans in *unbiased_text* for each biased segment.

    Primary strategy: align segment ``[start, end)`` boundaries from the
    original text onto the rewrite so the full neutralized region is
    highlighted. Falls back to flexible ``replacement`` string search when
    alignment fails.
    """
    if original_text == unbiased_text:
        return segments

    orig_to_unb = _build_orig_to_unb_map(original_text, unbiased_text)
    used: list[tuple[int, int]] = []
    ordered = sorted(
        enumerate(segments),
        key=lambda item: (item[1].start is None, item[1].start or 0),
    )
    updates: dict[int, tuple[int, int]] = {}

    for orig_idx, seg in ordered:
        span: tuple[int, int] | None = None

        if seg.start is not None and seg.end is not None:
            span = _boundary_replacement_span(
                original_text, unbiased_text, seg.start, seg.end, orig_to_unb
            )

        if span is None and seg.replacement:
            cursor = orig_to_unb[seg.start] if seg.start is not None else 0
            span = _find_replacement_span_in_unbiased(
                unbiased_text, seg.replacement, cursor=cursor
            )

        if span is None:
            logger.warning(
                "No replacement span for segment %r at [%s:%s)",
                seg.original,
                seg.start,
                seg.end,
            )
            continue

        for u in used:
            if not (span[1] <= u[0] or span[0] >= u[1]) and span[0] < u[1]:
                span = (u[1], span[1])
        if span[1] <= span[0]:
            if seg.replacement:
                span = _find_replacement_span_in_unbiased(
                    unbiased_text,
                    seg.replacement,
                    cursor=used[-1][1] if used else 0,
                )
            if span is None or span[1] <= span[0]:
                continue

        used.append(span)
        updates[orig_idx] = span

    enriched: list[BiasedSegment] = []
    for idx, seg in enumerate(segments):
        span = updates.get(idx)
        if span is None:
            enriched.append(seg)
            continue
        enriched.append(
            seg.model_copy(
                update={
                    "replacement_start": span[0],
                    "replacement_end": span[1],
                }
            )
        )

    return enriched


def _normalize_for_equality(s: str) -> str:
    """Normalise a phrase for original-vs-replacement comparison.

    Collapses whitespace, folds typographic quotes/dashes to ASCII, and
    casefolds so trivially-identical phrases compare equal despite the model's
    typography or spacing drift.
    """
    return _collapse_whitespace(_normalize_for_match(s)).casefold()


def drop_unchanged_segments(segments: list[BiasedSegment]) -> list[BiasedSegment]:
    """Drop segments whose replacement is identical to the original phrase.

    Under vLLM stochasticity the model occasionally flags a span but returns a
    replacement equal to the original (no actual edit). There is nothing to
    highlight, so these add noise and are removed. Segments with an empty
    replacement are kept: an empty replacement means "delete the phrase", which
    is a genuine edit.
    """
    kept: list[BiasedSegment] = []
    for seg in segments:
        replacement = seg.replacement.strip()
        if replacement and _normalize_for_equality(seg.original) == (
            _normalize_for_equality(replacement)
        ):
            logger.debug(
                "Dropping no-op segment (replacement == original): %r", seg.original
            )
            continue
        kept.append(seg)
    return kept


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