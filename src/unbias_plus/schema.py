"""Data models for unbias-plus output."""

from pydantic import BaseModel


class BiasedSegment(BaseModel):
    """A single biased segment detected in the text.

    Attributes
    ----------
    original : str
        The original biased phrase from the input text.
    replacement : str
        The suggested neutral replacement.
    severity : str
        Severity level of the bias ('low', 'medium', 'high').
    bias_type : str
        Type of bias detected (e.g. 'loaded language', 'framing bias').
    reasoning : str
        Explanation of why this segment is considered biased.

    """

    original: str
    replacement: str
    severity: str
    bias_type: str
    reasoning: str


class BiasResult(BaseModel):
    """Full bias analysis result for an input text.

    Attributes
    ----------
    binary_label : str
        Overall label, either 'biased' or 'unbiased'.
    severity : int
        Overall severity score from 1 (low) to 5 (high).
    bias_found : bool
        Whether any bias was detected in the text.
    biased_segments : list[BiasedSegment]
        List of biased segments found in the text.
    unbiased_text : str
        Full neutral rewrite of the input text.

    """

    binary_label: str
    severity: int
    bias_found: bool
    biased_segments: list[BiasedSegment]
    unbiased_text: str
