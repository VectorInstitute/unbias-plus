"""Tests for formatter module."""

import json

from unbias_plus.formatter import format_cli, format_dict, format_json
from unbias_plus.schema import BiasResult


def test_format_cli_returns_string(sample_result: BiasResult) -> None:
    """Test format_cli returns a string."""
    output = format_cli(sample_result)
    assert isinstance(output, str)


def test_format_cli_contains_label(sample_result: BiasResult) -> None:
    """Test format_cli output contains the bias label."""
    output = format_cli(sample_result)
    assert "BIASED" in output


def test_format_cli_contains_original(sample_result: BiasResult) -> None:
    """Test format_cli output contains the original biased phrase."""
    output = format_cli(sample_result)
    assert "Sharia-obsessed fanatics" in output


def test_format_cli_contains_replacement(sample_result: BiasResult) -> None:
    """Test format_cli output contains the replacement phrase."""
    output = format_cli(sample_result)
    assert "extremist groups" in output


def test_format_cli_contains_unbiased_text(sample_result: BiasResult) -> None:
    """Test format_cli output contains the unbiased text."""
    output = format_cli(sample_result)
    assert sample_result.unbiased_text in output


def test_format_dict_returns_dict(sample_result: BiasResult) -> None:
    """Test format_dict returns a dictionary."""
    output = format_dict(sample_result)
    assert isinstance(output, dict)


def test_format_dict_has_expected_keys(sample_result: BiasResult) -> None:
    """Test format_dict output has all expected keys."""
    output = format_dict(sample_result)
    assert "binary_label" in output
    assert "severity" in output
    assert "bias_found" in output
    assert "biased_segments" in output
    assert "unbiased_text" in output


def test_format_dict_segments_are_dicts(sample_result: BiasResult) -> None:
    """Test format_dict biased_segments are plain dicts."""
    output = format_dict(sample_result)
    assert isinstance(output["biased_segments"][0], dict)


def test_format_json_returns_string(sample_result: BiasResult) -> None:
    """Test format_json returns a string."""
    output = format_json(sample_result)
    assert isinstance(output, str)


def test_format_json_is_valid_json(sample_result: BiasResult) -> None:
    """Test format_json output is valid JSON."""
    output = format_json(sample_result)
    parsed = json.loads(output)
    assert parsed["binary_label"] == "biased"


def test_format_json_matches_dict(sample_result: BiasResult) -> None:
    """Test format_json and format_dict produce consistent data."""
    json_output = json.loads(format_json(sample_result))
    dict_output = format_dict(sample_result)
    assert json_output == dict_output
