"""Tests for schema module."""

import pytest
from pydantic import ValidationError

from unbias_plus.schema import BiasResult, BiasedSegment


def test_biased_segment_valid() -> None:
    """Test BiasedSegment accepts valid input."""
    seg = BiasedSegment(
        original="desperate for clicks",
        replacement="seeking audience engagement",
        severity="medium",
        bias_type="loaded language",
        reasoning="Pejorative motive attributed without evidence.",
    )
    assert seg.original == "desperate for clicks"
    assert seg.severity == "medium"


def test_biased_segment_severity_normalized() -> None:
    """Test BiasedSegment normalizes severity to lowercase."""
    seg = BiasedSegment(
        original="test",
        replacement="test",
        severity="HIGH",
        bias_type="test",
        reasoning="test",
    )
    assert seg.severity == "high"


def test_biased_segment_invalid_severity() -> None:
    """Test BiasedSegment raises ValidationError for invalid severity."""
    with pytest.raises(ValidationError):
        BiasedSegment(
            original="test",
            replacement="test",
            severity="extreme",
            bias_type="test",
            reasoning="test",
        )


def test_bias_result_valid(sample_result: BiasResult) -> None:
    """Test BiasResult accepts valid input."""
    assert sample_result.binary_label == "biased"
    assert sample_result.severity == 3
    assert sample_result.bias_found is True
    assert len(sample_result.biased_segments) == 1


def test_bias_result_label_normalized() -> None:
    """Test BiasResult normalizes binary_label to lowercase."""
    result = BiasResult(
        binary_label="BIASED",
        severity=1,
        bias_found=True,
        biased_segments=[],
        unbiased_text="test",
    )
    assert result.binary_label == "biased"


def test_bias_result_invalid_label() -> None:
    """Test BiasResult raises ValidationError for invalid binary_label."""
    with pytest.raises(ValidationError):
        BiasResult(
            binary_label="maybe",
            severity=1,
            bias_found=False,
            biased_segments=[],
            unbiased_text="test",
        )


def test_bias_result_severity_out_of_range() -> None:
    """Test BiasResult raises ValidationError for severity out of range."""
    with pytest.raises(ValidationError):
        BiasResult(
            binary_label="biased",
            severity=6,
            bias_found=True,
            biased_segments=[],
            unbiased_text="test",
        )


def test_bias_result_unbiased_empty_segments() -> None:
    """Test BiasResult accepts empty biased_segments when unbiased."""
    result = BiasResult(
        binary_label="unbiased",
        severity=1,
        bias_found=False,
        biased_segments=[],
        unbiased_text="This text is neutral.",
    )
    assert result.bias_found is False
    assert result.biased_segments == []
