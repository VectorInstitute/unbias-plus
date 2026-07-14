"""Tests for input cleaning (quote stripping)."""

from unbias_plus.cleaning import prepare_input, strip_quotes


def test_strip_quotes_removes_ascii_and_curly_doubles() -> None:
    """Double quotes of common styles are removed."""
    assert strip_quotes('He said "false" claims.') == "He said false claims."
    assert strip_quotes("She said \u201cmisleading\u201d news.") == "She said misleading news."


def test_strip_quotes_removes_edge_singles_keeps_apostrophes() -> None:
    """Edge single quotes go; mid-word apostrophes stay."""
    assert strip_quotes("It's 'biased' framing.") == "It's biased framing."


def test_prepare_input_matches_strip_quotes() -> None:
    """prepare_input is the inference entry point for strip_quotes."""
    raw = 'The "experts" don\'t agree.'
    assert prepare_input(raw) == strip_quotes(raw)
    assert '"' not in prepare_input(raw)
