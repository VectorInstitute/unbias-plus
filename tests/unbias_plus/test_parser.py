"""Tests for parser module."""

import pytest

from unbias_plus.parser import _extract_json, parse_llm_output
from unbias_plus.schema import BiasResult


def test_parse_llm_output_valid(sample_json: str) -> None:
    """Test parse_llm_output returns BiasResult for valid JSON."""
    result = parse_llm_output(sample_json)
    assert isinstance(result, BiasResult)
    assert result.binary_label == "biased"
    assert result.severity == 3
    assert len(result.biased_segments) == 1


def test_parse_llm_output_fenced_json(sample_json: str) -> None:
    """Test parse_llm_output handles JSON in markdown code fences."""
    fenced = f"```json\n{sample_json}\n```"
    result = parse_llm_output(fenced)
    assert isinstance(result, BiasResult)


def test_parse_llm_output_with_extra_text(sample_json: str) -> None:
    """Test parse_llm_output handles JSON with extra text around it."""
    noisy = f"Here is the analysis:\n{sample_json}\nHope that helps!"
    result = parse_llm_output(noisy)
    assert isinstance(result, BiasResult)


def test_parse_llm_output_invalid_json() -> None:
    """Test parse_llm_output raises ValueError for invalid JSON."""
    with pytest.raises(ValueError, match="could not be parsed as JSON"):
        parse_llm_output("this is not json at all")


def test_parse_llm_output_wrong_schema() -> None:
    """Test parse_llm_output raises ValueError for wrong schema."""
    with pytest.raises(ValueError, match="does not match expected schema"):
        parse_llm_output('{"wrong_field": "wrong_value"}')


def test_extract_json_clean() -> None:
    """Test _extract_json returns clean JSON as-is."""
    raw = '{"key": "value"}'
    assert _extract_json(raw) == raw


def test_extract_json_fenced() -> None:
    """Test _extract_json strips markdown code fences."""
    raw = '```json\n{"key": "value"}\n```'
    assert _extract_json(raw) == '{"key": "value"}'


def test_extract_json_buried() -> None:
    """Test _extract_json finds JSON buried in extra text."""
    raw = 'Some text before {"key": "value"} some text after'
    assert _extract_json(raw) == '{"key": "value"}'


def test_parse_llm_output_unbiased() -> None:
    """Test parse_llm_output handles unbiased result correctly."""
    json_str = """
    {
        "binary_label": "unbiased",
        "severity": 1,
        "bias_found": false,
        "biased_segments": [],
        "unbiased_text": "This is a neutral text."
    }
    """
    result = parse_llm_output(json_str)
    assert result.binary_label == "unbiased"
    assert result.bias_found is False
    assert result.biased_segments == []
