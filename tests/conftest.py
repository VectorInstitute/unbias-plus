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
        original="flood of migrants",
        replacement="arrival of migrants",
        severity="High",
        bias_type="dehumanizing_language",
        reasoning="Treats people as a threatening mass.",
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
        severity=6,
        bias_found=True,
        biased_segments=[sample_segment],
        unbiased_text="They are surrounded by arrival of migrants.",
    )


@pytest.fixture
def sample_json() -> str:
    """Return a sample valid JSON string matching the model output schema.

    Returns
    -------
    str
        A valid JSON string for testing the parser.

    """
    return """
    {
        "severity": 6,
        "biased_segments": [
            {
                "original": "flood of migrants",
                "replacement": "arrival of migrants",
                "severity": "High",
                "bias_type": "dehumanizing_language",
                "reasoning": "Treats people as a threatening mass."
            }
        ],
        "unbiased_text": "They are surrounded by arrival of migrants."
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
