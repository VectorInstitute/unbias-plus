"""Data schemas for unbias-plus output."""

import logging

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


def compute_offsets(
    original_text: str, segments: list[BiasedSegment]
) -> list[BiasedSegment]:
    """Compute character start/end offsets for each biased segment.

    Walks the original text with a cursor so that duplicate phrases
    are matched in order of appearance, not just the first occurrence.

    Parameters
    ----------
    original_text : str
        The original input text.
    segments : list[BiasedSegment]
        Parsed segments from the LLM (without offsets).

    Returns
    -------
    list[BiasedSegment]
        Segments with start/end fields populated, sorted by start offset.

    """
    cursor = 0
    enriched = []

    for seg in segments:
        phrase = seg.original
        if not phrase:
            continue

        start = _find_case_insensitive(original_text, phrase, cursor)
        if start == -1:
            start = _find_case_insensitive(original_text, phrase, 0)

        if start == -1:
            logger.warning("Could not find segment in text: '%s'", phrase)
            enriched.append(seg)
            continue

        end = start + len(phrase)
        enriched.append(seg.model_copy(update={"start": start, "end": end}))
        cursor = end

    enriched.sort(key=lambda s: s.start if s.start is not None else 0)
    return enriched
