"""Main pipeline for unbias-plus."""

from pathlib import Path

from unbias_plus.cleaning import prepare_input
from unbias_plus.formatter import format_cli, format_dict, format_json
from unbias_plus.model import DEFAULT_MODEL, UnBiasModel
from unbias_plus.parser import parse_llm_output
from unbias_plus.prompt import build_messages
from unbias_plus.schema import (
    BiasResult,
    compute_offsets,
    compute_replacement_offsets,
    drop_unchanged_segments,
)


class UnBiasPlus:
    """Main pipeline for bias detection and debiasing.

    Loads a fine-tuned LLM and exposes a simple interface for
    analyzing text for bias. Combines prompt building, inference,
    JSON parsing, offset computation, and formatting.

    Parameters
    ----------
    model_name_or_path : str | Path
        HuggingFace model ID or local path to the fine-tuned
        model. Defaults to ``DEFAULT_MODEL``
        (``vector-institute/Qwen3-8B-UnBias-Plus-SFT-Instruct-V2``).
    device : str | None, optional
        Device to run on ('cuda' or 'cpu'). Auto-detected if None.
    load_in_4bit : bool, optional
        Load model in 4-bit quantization. Default is False.
    max_new_tokens : int, optional
        Maximum tokens to generate. Default is 8096.

    Examples
    --------
    >>> from unbias_plus import UnBiasPlus  # doctest: +SKIP
    >>> pipe = UnBiasPlus()  # doctest: +SKIP
    >>> result = pipe.analyze("Women are too emotional to lead.")  # doctest: +SKIP
    >>> print(result.binary_label)  # doctest: +SKIP
    biased

    """

    def __init__(
        self,
        model_name_or_path: str | Path = DEFAULT_MODEL,
        device: str | None = None,
        load_in_4bit: bool = False,
        max_new_tokens: int = 8096,
    ) -> None:
        self._model = UnBiasModel(
            model_name_or_path=model_name_or_path,
            device=device,
            load_in_4bit=load_in_4bit,
            max_new_tokens=max_new_tokens,
        )

    def analyze(self, text: str) -> BiasResult:
        """Analyze input text for bias.

        Runs the full pipeline: builds chat messages, runs inference,
        parses JSON output, computes character offsets for each
        segment, and attaches the original text to the result.

        Parameters
        ----------
        text : str
            The input text to analyze.

        Returns
        -------
        BiasResult
            Structured bias result with start/end offsets on each
            segment and original_text populated.

        Raises
        ------
        ValueError
            If the LLM output cannot be parsed into a valid BiasResult.

        Examples
        --------
        >>> result = pipe.analyze("All politicians are liars.")  # doctest: +SKIP
        >>> result.bias_found  # doctest: +SKIP
        True

        """
        text = prepare_input(text)
        messages = build_messages(text)
        raw_output = self._model.generate(messages)
        result = parse_llm_output(raw_output)
        return finalize_result(text, result)

    def analyze_to_cli(self, text: str) -> str:
        """Analyze text and return a formatted CLI string.

        Parameters
        ----------
        text : str
            The input text to analyze.

        Returns
        -------
        str
            Human-readable colored string for terminal display.

        """
        return format_cli(self.analyze(text))

    def analyze_to_dict(self, text: str) -> dict:
        """Analyze text and return result as a plain dictionary.

        Parameters
        ----------
        text : str
            The input text to analyze.

        Returns
        -------
        dict
            Plain dictionary representation of the result.

        """
        return format_dict(self.analyze(text))

    def analyze_to_json(self, text: str) -> str:
        """Analyze text and return result as a JSON string.

        Parameters
        ----------
        text : str
            The input text to analyze.

        Returns
        -------
        str
            Pretty-printed JSON string of the result.

        """
        return format_json(self.analyze(text))


def finalize_result(text: str, result: BiasResult) -> BiasResult:
    """Attach original text and character offsets for each biased segment.

    Segments whose replacement equals the original are dropped as no-ops. If no
    segments survive, the result is reconciled to a clean unbiased state
    (severity 0, no bias, rewrite == original) since, per the model contract,
    an empty ``biased_segments`` means there is no bias to report or rewrite.
    """
    segments = drop_unchanged_segments(result.biased_segments)
    segments = compute_offsets(text, segments)
    segments = compute_replacement_offsets(text, result.unbiased_text, segments)

    if not segments:
        return result.model_copy(
            update={
                "biased_segments": [],
                "binary_label": "unbiased",
                "bias_found": False,
                "severity": 0,
                "unbiased_text": text,
                "original_text": text,
            }
        )

    return result.model_copy(
        update={
            "biased_segments": segments,
            "original_text": text,
        }
    )
