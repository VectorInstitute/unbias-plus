"""Tests for prompt module."""

from unbias_plus.prompt import SYSTEM_PROMPT, build_messages


def test_build_messages_contains_text(sample_text: str) -> None:
    """Test build_messages includes the input text in user message."""
    messages = build_messages(sample_text)
    assert sample_text in messages[1]["content"]


def test_build_messages_contains_system_prompt(sample_text: str) -> None:
    """Test build_messages includes JSON in system prompt."""
    messages = build_messages(sample_text)
    assert "JSON" in messages[0]["content"]


def test_build_messages_returns_list(sample_text: str) -> None:
    """Test build_messages returns a list of dicts."""
    messages = build_messages(sample_text)
    assert isinstance(messages, list)


def test_build_messages_has_system_and_user(sample_text: str) -> None:
    """Test build_messages returns system and user roles."""
    messages = build_messages(sample_text)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_system_prompt_mentions_binary_label() -> None:
    """Test SYSTEM_PROMPT mentions binary_label field."""
    assert "binary_label" in SYSTEM_PROMPT


def test_system_prompt_mentions_biased_segments() -> None:
    """Test SYSTEM_PROMPT mentions biased_segments field."""
    assert "biased_segments" in SYSTEM_PROMPT


def test_system_prompt_mentions_unbiased_text() -> None:
    """Test SYSTEM_PROMPT mentions unbiased_text field."""
    assert "unbiased_text" in SYSTEM_PROMPT
