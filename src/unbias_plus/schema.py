"""Data schemas for unbias-plus output."""

import logging

from pydantic import BaseModel, field_validator


logger = logging.getLogger(__name__)


class BiasedSegment(BaseModel):
    """A single biased segment detected in the text.

    Attributes
    ----------
    original : str
        The original biased phrase from the input text.
    replacement : str
        The suggested neutral replacement.
    severity : str
        Severity level: 'low', 'medium', or 'high'.
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
    replacement: str
    severity: str
    bias_type: str
    reasoning: str
    start: int | None = None
    end: int | None = None

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        """Validate severity is one of the allowed values.

        Parameters
        ----------
        v : str
            The severity value to validate.

        Returns
        -------
        str
            The validated severity value.

        Raises
        ------
        ValueError
            If severity is not 'low', 'medium', or 'high'.

        """
        allowed = {"low", "medium", "high"}
        if v.lower() not in allowed:
            raise ValueError(f"severity must be one of {allowed}, got '{v}'")
        return v.lower()


class BiasResult(BaseModel):
    """Full bias analysis result for an input text.

    Attributes
    ----------
    binary_label : str
        Overall label: 'biased' or 'unbiased'.
    severity : int
        Overall severity score from 1 (low) to 5 (high).
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
        """Validate binary_label is 'biased' or 'unbiased'.

        Parameters
        ----------
        v : str
            The binary label value to validate.

        Returns
        -------
        str
            The validated binary label.

        Raises
        ------
        ValueError
            If label is not 'biased' or 'unbiased'.

        """
        allowed = {"biased", "unbiased"}
        if v.lower() not in allowed:
            raise ValueError(f"binary_label must be one of {allowed}, got '{v}'")
        return v.lower()

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: int) -> int:
        """Validate severity is between 1 and 5.

        Parameters
        ----------
        v : int
            The severity score to validate.

        Returns
        -------
        int
            The validated severity score.

        Raises
        ------
        ValueError
            If severity is not between 1 and 5.

        """
        if not 1 <= v <= 5:
            raise ValueError(f"severity must be between 1 and 5, got {v}")
        return v

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
