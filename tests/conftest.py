"""Shared test fixtures for unbias-plus."""

import pytest

from unbias_plus.schema import BiasedSegment, BiasResult


@pytest.fixture
def sample_segment() -> BiasedSegment:
    """Return a sample BiasedSegment for testing.

    Returns
    -------
    BiasedSegment
        A sample biased segment fixture.

    """
    return BiasedSegment(
        original="Sharia-obsessed fanatics",
        replacement="extremist groups",
        severity="high",
        bias_type="dehumanizing framing",
        reasoning="Uses inflammatory religious language to dehumanize.",
    )


@pytest.fixture
def sample_result(sample_segment: BiasedSegment) -> BiasResult:
    """Return a sample BiasResult for testing.

    Parameters
    ----------
    sample_segment : BiasedSegment
        A sample biased segment fixture.

    Returns
    -------
    BiasResult
        A sample bias result fixture.

    """
    return BiasResult(
        binary_label="biased",
        severity=3,
        bias_found=True,
        biased_segments=[sample_segment],
        unbiased_text="They are surrounded by extremist groups.",
    )


@pytest.fixture
def sample_json() -> str:
    """Return a sample valid JSON string matching BiasResult schema.

    Returns
    -------
    str
        A valid JSON string for testing the parser.

    """
    return """
    {
        "binary_label": "biased",
        "severity": 3,
        "bias_found": true,
        "biased_segments": [
            {
                "original": "Sharia-obsessed fanatics",
                "replacement": "extremist groups",
                "severity": "high",
                "bias_type": "dehumanizing framing",
                "reasoning": "Uses inflammatory language."
            }
        ],
        "unbiased_text": "They are surrounded by extremist groups."
    }
    """


@pytest.fixture
def sample_text() -> str:
    """Return a sample biased text for testing.

    Returns
    -------
    str
        A sample biased input text.

    """
    return "Women are too emotional to lead."
