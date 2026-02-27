"""Main pipeline for unbias-plus."""

from pathlib import Path

from unbias_plus.formatter import format_cli, format_dict, format_json
from unbias_plus.model import UnBiasModel
from unbias_plus.parser import parse_llm_output
from unbias_plus.prompt import build_prompt
from unbias_plus.schema import BiasResult


class UnBiasPlus:
    """Main pipeline for bias detection and debiasing.

    Loads the fine-tuned LLM and exposes a simple interface
    for analyzing text for bias.

    Parameters
    ----------
    model_name_or_path : str | Path
        HuggingFace model ID or local path to the fine-tuned model.
    device : str, optional
        Device to run inference on ('cuda' or 'cpu').
    load_in_4bit : bool, optional
        Whether to load model in 4-bit quantization. Default is False.

    Examples
    --------
    >>> from unbias_plus import UnBiasPlus
    >>> model = UnBiasPlus("vectorinstitute/unbias-plus-qwen3-8b")
    >>> result = model.analyze("Women are too emotional to lead.")
    >>> print(result.binary_label)
    'biased'

    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        device: str | None = None,
        load_in_4bit: bool = False,
    ) -> None:
        """Initialize the pipeline with the LLM."""
        pass

    def analyze(self, text: str) -> BiasResult:
        """Analyze input text for bias.

        Parameters
        ----------
        text : str
            The input text to analyze.

        Returns
        -------
        BiasResult
            Structured bias analysis result.

        """
        pass

    def analyze_to_cli(self, text: str) -> str:
        """Analyze text and return formatted CLI string.

        Parameters
        ----------
        text : str
            The input text to analyze.

        Returns
        -------
        str
            Human-readable formatted result for terminal display.

        """
        pass

    def analyze_to_dict(self, text: str) -> dict:  # type: ignore[type-arg]
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
        pass

    def analyze_to_json(self, text: str) -> str:
        """Analyze text and return result as a JSON string.

        Parameters
        ----------
        text : str
            The input text to analyze.

        Returns
        -------
        str
            JSON string representation of the result.

        """
        pass
