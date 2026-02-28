"""Main pipeline for unbias-plus."""

from pathlib import Path

from unbias_plus.formatter import format_cli, format_dict, format_json
from unbias_plus.model import DEFAULT_MODEL, UnBiasModel
from unbias_plus.parser import parse_llm_output
from unbias_plus.prompt import build_prompt
from unbias_plus.schema import BiasResult, compute_offsets


class UnBiasPlus:
    """Main pipeline for bias detection and debiasing.

    Loads a fine-tuned LLM and exposes a simple interface for
    analyzing text for bias. Combines prompt building, inference,
    JSON parsing, offset computation, and formatting.

    Parameters
    ----------
    model_name_or_path : str | Path
        HuggingFace model ID or local path to the fine-tuned
        model. Defaults to 'meta-llama/Llama-3.2-3B'.
    device : str | None, optional
        Device to run on ('cuda' or 'cpu'). Auto-detected if None.
    load_in_4bit : bool, optional
        Load model in 4-bit quantization. Default is False.
    max_new_tokens : int, optional
        Maximum tokens to generate. Default is 1024.

    Examples
    --------
    >>> from unbias_plus import UnBiasPlus  # doctest: +SKIP
    >>> pipe = UnBiasPlus("meta-llama/Llama-3.2-3B")  # doctest: +SKIP
    >>> result = pipe.analyze("Women are too emotional to lead.")  # doctest: +SKIP
    >>> print(result.binary_label)  # doctest: +SKIP
    biased

    """

    def __init__(
        self,
        model_name_or_path: str | Path = DEFAULT_MODEL,
        device: str | None = None,
        load_in_4bit: bool = False,
        max_new_tokens: int = 1024,
    ) -> None:
        """Initialize the pipeline with the LLM.

        Parameters
        ----------
        model_name_or_path : str | Path
            HuggingFace model ID or local path.
        device : str | None
            Target device. Auto-detected if None.
        load_in_4bit : bool
            Whether to use 4-bit quantization.
        max_new_tokens : int
            Max tokens to generate per call.

        """
        self._model = UnBiasModel(
            model_name_or_path=model_name_or_path,
            device=device,
            load_in_4bit=load_in_4bit,
            max_new_tokens=max_new_tokens,
        )

    def analyze(self, text: str) -> BiasResult:
        """Analyze input text for bias.

        Runs the full pipeline: builds prompt, runs inference,
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
        prompt = build_prompt(text)
        raw_output = self._model.generate(prompt)
        result = parse_llm_output(raw_output)

        # Compute character-level offsets for frontend highlighting
        segments_with_offsets = compute_offsets(text, result.biased_segments)

        return result.model_copy(
            update={
                "biased_segments": segments_with_offsets,
                "original_text": text,
            }
        )

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
