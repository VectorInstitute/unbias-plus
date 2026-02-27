"""LLM model loader and inference for unbias-plus."""

from pathlib import Path


class UnBiasModel:
    """Loads and runs the fine-tuned bias detection LLM.

    Parameters
    ----------
    model_name_or_path : str | Path
        HuggingFace model ID or local path to the fine-tuned model.
    device : str, optional
        Device to run inference on ('cuda' or 'cpu').
        Defaults to 'cuda' if available, else 'cpu'.
    load_in_4bit : bool, optional
        Whether to load model in 4-bit quantization. Default is False.

    Examples
    --------
    >>> model = UnBiasModel("vectorinstitute/unbias-plus-qwen3-8b")
    >>> output = model.generate("Women are too emotional to lead.")

    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        device: str | None = None,
        load_in_4bit: bool = False,
    ) -> None:
        """Initialize the model and tokenizer."""
        pass

    def generate(self, prompt: str) -> str:
        """Run inference and return the raw LLM output string.

        Parameters
        ----------
        prompt : str
            The fully formatted prompt to send to the LLM.

        Returns
        -------
        str
            Raw string output from the LLM.

        """
        pass
