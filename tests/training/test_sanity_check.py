"""Unit tests for the pure helpers in ``training/sanity_check.py``.

Only ``extract_json`` and ``parse_args`` can be tested without GPU + unsloth.
``load_model``, ``generate_response``, and ``report`` all touch the model and
tokenizer, so they live outside the testable surface.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from training.sanity_check import extract_json, parse_args


# ---------------------------------------------------------------------------
# extract_json — handles 3 completion shapes documented in the function:
#   - <think> block closed, JSON follows
#   - <think> block hit max_tokens and was never closed
#   - Pure JSON with no thinking block
# ---------------------------------------------------------------------------


class TestExtractJson:
    """Behaviour of the brace-walking JSON extractor."""

    def test_pure_json_no_thinking(self) -> None:
        """Plain JSON with no ``<think>`` block is returned verbatim."""
        text = '{"binary_label": "biased", "severity": 3}'
        assert extract_json(text) == text

    def test_json_after_closed_think_block(self) -> None:
        """A closed ``</think>`` block is skipped; JSON after it is returned."""
        text = (
            "<think>\nLet me analyze...\nOK done.\n</think>\n"
            '{"binary_label": "unbiased"}'
        )
        assert extract_json(text) == '{"binary_label": "unbiased"}'

    def test_unclosed_think_block_finds_json(self) -> None:
        """An unclosed ``<think>`` block is tolerated — JSON found anyway."""
        text = '<think>\nReasoning that never closed... {"binary_label": "biased"}'
        # The first ``{`` belongs to the JSON object; the function returns from there.
        assert extract_json(text) == '{"binary_label": "biased"}'

    def test_no_brace_returns_empty(self) -> None:
        """No ``{`` anywhere in output → empty string."""
        assert extract_json("just some text with no json at all") == ""

    def test_empty_input(self) -> None:
        """Empty string in → empty string out."""
        assert extract_json("") == ""

    def test_nested_braces_match_outer(self) -> None:
        """Nested objects: matched closing brace is the outer one."""
        text = '{"a": 1, "b": {"c": 2, "d": {"e": 3}}}'
        assert extract_json(text) == text

    def test_nested_braces_with_thinking(self) -> None:
        """Nested objects after a ``</think>`` block are extracted whole."""
        text = (
            "<think>...</think>"
            '{"binary_label": "biased", "biased_segments": [{"original": "x"}]}'
        )
        expected = '{"binary_label": "biased", "biased_segments": [{"original": "x"}]}'
        assert extract_json(text) == expected

    def test_truncated_json_returns_everything_from_brace(self) -> None:
        """Generation that hit max_tokens mid-JSON returns from ``{`` onward."""
        text = '<think>...</think>{"binary_label": "biased", "biased_segments": [{"orig'
        result = extract_json(text)
        assert result.startswith("{")
        # No matching close brace, so the function returns the unbalanced tail
        # for downstream JSONDecodeError diagnosis.
        assert result.count("{") > result.count("}")

    def test_returns_only_first_complete_object(self) -> None:
        """If model emits two objects, only the first closes our depth and returns."""
        text = '{"a": 1}{"b": 2}'
        assert extract_json(text) == '{"a": 1}'

    def test_whitespace_before_brace(self) -> None:
        """Leading whitespace before the JSON object is fine."""
        text = '\n\n  {"x": 1}'
        assert extract_json(text) == '{"x": 1}'

    def test_text_before_brace_is_skipped(self) -> None:
        """Any prose before the first ``{`` is skipped."""
        text = 'Some preamble explaining things {"binary_label": "biased"}'
        assert extract_json(text) == '{"binary_label": "biased"}'

    def test_only_first_close_of_thinking_block_is_used(self) -> None:
        """``split('</think>', 1)`` means only the first close splits."""
        text = '<think>a</think>preamble{"x":1}</think>{"y":2}'
        # After splitting once on the first </think>, we look for { in the rest.
        # The first { found is the one in {"x":1} (preamble has none), so we get it.
        assert extract_json(text) == '{"x":1}'


# ---------------------------------------------------------------------------
# parse_args — argparse wiring (required vs optional, defaults, bool action)
# ---------------------------------------------------------------------------


class TestParseArgs:
    """CLI argument parsing for sanity_check.py."""

    def test_required_args_minimal(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Just the two required args populates the namespace correctly."""
        model_path = tmp_path / "model"
        article_path = tmp_path / "article.txt"
        monkeypatch.setattr(
            "sys.argv",
            [
                "sanity_check.py",
                "--model-path",
                str(model_path),
                "--article-file",
                str(article_path),
            ],
        )
        args = parse_args()
        assert args.model_path == model_path
        assert args.article_file == article_path

    def test_defaults_applied(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Optional args fall back to their documented defaults."""
        monkeypatch.setattr(
            "sys.argv",
            [
                "sanity_check.py",
                "--model-path",
                str(tmp_path / "m"),
                "--article-file",
                str(tmp_path / "a.txt"),
            ],
        )
        args = parse_args()
        assert args.max_seq_length == 8192
        assert args.max_new_tokens == 4096
        assert args.temperature == pytest.approx(0.1)
        assert args.thinking is False  # default --no-thinking

    def test_thinking_flag_explicit_on(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """``--thinking`` enables the Qwen thinking block."""
        monkeypatch.setattr(
            "sys.argv",
            [
                "sanity_check.py",
                "--model-path",
                str(tmp_path / "m"),
                "--article-file",
                str(tmp_path / "a.txt"),
                "--thinking",
            ],
        )
        args = parse_args()
        assert args.thinking is True

    def test_thinking_flag_explicit_off(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """``--no-thinking`` keeps the default off, idempotent with default."""
        monkeypatch.setattr(
            "sys.argv",
            [
                "sanity_check.py",
                "--model-path",
                str(tmp_path / "m"),
                "--article-file",
                str(tmp_path / "a.txt"),
                "--no-thinking",
            ],
        )
        args = parse_args()
        assert args.thinking is False

    def test_overrides(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Numeric overrides flow through correctly."""
        monkeypatch.setattr(
            "sys.argv",
            [
                "sanity_check.py",
                "--model-path",
                str(tmp_path / "m"),
                "--article-file",
                str(tmp_path / "a.txt"),
                "--max-seq-length",
                "4096",
                "--max-new-tokens",
                "1024",
                "--temperature",
                "0.7",
            ],
        )
        args = parse_args()
        assert args.max_seq_length == 4096
        assert args.max_new_tokens == 1024
        assert args.temperature == pytest.approx(0.7)

    def test_missing_required_args_exits(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Argparse exits non-zero when --model-path or --article-file is missing."""
        monkeypatch.setattr("sys.argv", ["sanity_check.py"])
        with pytest.raises(SystemExit):
            parse_args()
