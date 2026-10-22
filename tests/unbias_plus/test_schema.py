"""Tests for schema module."""

import pytest
from pydantic import ValidationError

from unbias_plus.schema import (
    BiasedSegment,
    BiasResult,
    compute_offsets,
    compute_replacement_offsets,
)


def test_biased_segment_valid() -> None:
    """Test BiasedSegment accepts valid input."""
    seg = BiasedSegment(
        original="desperate for clicks",
        replacement="seeking audience engagement",
        severity="Medium",
        bias_type="loaded_language",
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
        bias_type="loaded_language",
        reasoning="test",
    )
    assert seg.severity == "high"


def test_biased_segment_invalid_severity() -> None:
    """Test BiasedSegment coerces invalid severity to 'medium'."""
    seg = BiasedSegment(
        original="test",
        replacement="test",
        severity="extreme",
        bias_type="loaded_language",
        reasoning="test",
    )
    assert seg.severity == "medium"


def test_bias_result_valid(sample_result: BiasResult) -> None:
    """Test BiasResult accepts valid input."""
    assert sample_result.binary_label == "biased"
    assert sample_result.severity == 6
    assert sample_result.bias_found is True
    assert len(sample_result.biased_segments) == 1


def test_bias_result_label_normalized() -> None:
    """Test BiasResult normalizes binary_label to lowercase."""
    result = BiasResult(
        binary_label="BIASED",
        severity=4,
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
            severity=2,
            bias_found=False,
            biased_segments=[],
            unbiased_text="test",
        )


def test_bias_result_severity_out_of_range() -> None:
    """Test BiasResult clamps severity > 10 to 10."""
    result = BiasResult(
        binary_label="biased",
        severity=15,
        bias_found=True,
        biased_segments=[],
        unbiased_text="test",
    )
    assert result.severity == 10


def test_bias_result_severity_accepts_full_scale() -> None:
    """Test BiasResult accepts full 0-10 severity scale."""
    for sev in range(0, 11):
        result = BiasResult(
            binary_label="biased" if sev > 0 else "unbiased",
            severity=sev,
            bias_found=sev > 0,
            biased_segments=[],
            unbiased_text="test",
        )
        assert result.severity == sev


def test_bias_result_unbiased_empty_segments() -> None:
    """Test BiasResult accepts empty biased_segments when unbiased."""
    result = BiasResult(
        binary_label="unbiased",
        severity=0,
        bias_found=False,
        biased_segments=[],
        unbiased_text="This text is neutral.",
    )
    assert result.bias_found is False
    assert result.biased_segments == []


def test_compute_replacement_offsets_from_diff() -> None:
    """Replacement highlights follow the actual rewrite, not the replacement field."""
    original = (
        "When the nursing staff raised concerns about the new schedule, it was the male doctor "
        "who stepped in to explain the situation clearly. The nurses, mostly women, had been "
        "overreacting as usual — their complaints driven more by feelings than by facts. "
        "Hospital management agreed that having a man in charge helped bring some much-needed "
        "rationality to what had become a needlessly emotional debate."
    )
    unbiased = (
        "When the nursing staff raised concerns about the new schedule, it was the male doctor "
        "who stepped in to explain the situation clearly. The nurses, mostly women, had been "
        "expressing concerns as usual — their concerns based on personal perspectives. "
        "Hospital management agreed that having a man in charge helped bring some much-needed "
        "rationality to what had become an emotional discussion."
    )
    segments = [
        BiasedSegment(
            original="overreacting as usual",
            replacement="expressing concerns",
            severity="medium",
            bias_type="loaded_language",
            reasoning="",
        ),
        BiasedSegment(
            original="their complaints driven more by feelings than by facts",
            replacement="their concerns were based on personal perspectives",
            severity="medium",
            bias_type="stereotypical_association",
            reasoning="",
        ),
        BiasedSegment(
            original="needlessly emotional debate",
            replacement="emotional discussion",
            severity="medium",
            bias_type="loaded_language",
            reasoning="",
        ),
    ]

    with_offsets = compute_offsets(original, segments)
    with_replacements = compute_replacement_offsets(original, unbiased, with_offsets)

    seg1, seg2, seg3 = with_replacements
    assert unbiased[seg1.replacement_start : seg1.replacement_end] == (
        "expressing concerns as usual"
    )
    assert (
        unbiased[seg2.replacement_start : seg2.replacement_end]
        == "their concerns based on personal perspectives."
    )
    assert (
        unbiased[seg3.replacement_start : seg3.replacement_end]
        == "an emotional discussion."
    )


def test_compute_replacement_offsets_identical_text() -> None:
    """No replacement spans when original and unbiased text match."""
    text = "This text is neutral."
    segments = [
        BiasedSegment(
            original="neutral",
            replacement="neutral",
            severity="low",
            bias_type="",
            reasoning="",
            start=10,
            end=17,
        )
    ]
    result = compute_replacement_offsets(text, text, segments)
    assert result[0].replacement_start is None
    assert result[0].replacement_end is None


def test_compute_offsets_ignores_leading_space_in_phrase() -> None:
    """Model originals with a leading space must not highlight the prior char."""
    text = "to the white engineer even though the Latina developer wrote it."
    segments = [
        BiasedSegment(
            original=" even though the Latina developer wrote it",
            replacement="even though the developer wrote it",
            severity="low",
            bias_type="stereotypical_association",
            reasoning="test",
        )
    ]
    result = compute_offsets(text, segments)
    assert result[0].start is not None
    assert result[0].end is not None
    assert text[result[0].start : result[0].end] == (
        "even though the Latina developer wrote it"
    )
    assert not text[result[0].start].isspace()
