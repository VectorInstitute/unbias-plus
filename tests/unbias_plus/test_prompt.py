"""Tests for prompt module."""

from unbias_plus.prompt import SYSTEM_PROMPT, build_prompt


def test_build_prompt_contains_text(sample_text: str) -> None:
    """Test build_prompt includes the input text in output."""
    prompt = build_prompt(sample_text)
    assert sample_text in prompt


def test_build_prompt_contains_system_prompt(sample_text: str) -> None:
    """Test build_prompt includes the system prompt."""
    prompt = build_prompt(sample_text)
    assert "JSON" in prompt


def test_build_prompt_returns_string(sample_text: str) -> None:
    """Test build_prompt returns a string."""
    prompt = build_prompt(sample_text)
    assert isinstance(prompt, str)


def test_build_prompt_ends_with_json_cue(sample_text: str) -> None:
    """Test build_prompt ends with JSON output cue."""
    prompt = build_prompt(sample_text)
    assert prompt.strip().endswith("JSON output:")


def test_system_prompt_mentions_binary_label() -> None:
    """Test SYSTEM_PROMPT mentions binary_label field."""
    assert "binary_label" in SYSTEM_PROMPT


def test_system_prompt_mentions_biased_segments() -> None:
    """Test SYSTEM_PROMPT mentions biased_segments field."""
    assert "biased_segments" in SYSTEM_PROMPT


def test_system_prompt_mentions_unbiased_text() -> None:
    """Test SYSTEM_PROMPT mentions unbiased_text field."""
    assert "unbiased_text" in SYSTEM_PROMPT
