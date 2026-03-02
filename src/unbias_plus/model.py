"""LLM model loader and inference for unbias-plus."""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL = "vector-institute/Unbias-plus-Qwen2.5"


class UnBiasModel:
    """Loads and runs the fine-tuned bias detection LLM.

    Wraps a HuggingFace causal LM with a simple generate()
    interface. Handles device placement and optional 4-bit
    quantization automatically.

    Parameters
    ----------
    model_name_or_path : str | Path
        HuggingFace model ID or local path to the model.
        Defaults to 'Qwen/Qwen3-4B'.
    device : str | None, optional
        Device to run on ('cuda' or 'cpu').
        Auto-detects if not provided.
    load_in_4bit : bool, optional
        Load model in 4-bit quantization via bitsandbytes.
        Reduces VRAM usage significantly. Default is False.
    max_new_tokens : int, optional
        Maximum number of new tokens to generate. Default 1024.

    Examples
    --------
    >>> model = UnBiasModel()  # doctest: +SKIP
    >>> raw = model.generate("Analyze this text for bias...")  # doctest: +SKIP
    >>> isinstance(raw, str)  # doctest: +SKIP
    True

    """

    def __init__(
        self,
        model_name_or_path: str | Path = DEFAULT_MODEL,
        device: str | None = None,
        load_in_4bit: bool = False,
        max_new_tokens: int = 2048,
    ) -> None:
        """Initialize the model and tokenizer.

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
        self.model_name_or_path = str(model_name_or_path)
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # If the caller did not explicitly choose a different model, they are
        # using the default LoRA-4bit checkpoint. For that model we need to
        # enable 4-bit loading automatically, even if load_in_4bit=False was
        # passed through higher-level defaults (CLI / API). For any other
        # model, 4-bit remains opt-in.
        effective_load_in_4bit = load_in_4bit or (
            self.model_name_or_path == DEFAULT_MODEL
        )

        quantization_config = None
        if effective_load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto" if effective_load_in_4bit else None,
            quantization_config=quantization_config,
        )

        if not effective_load_in_4bit:
            self.model = self.model.to(self.device)  # type: ignore[arg-type]

        self.model.eval()

    def generate(self, prompt: str) -> str:
        """Run inference and return the raw LLM output string.

        Parameters
        ----------
        prompt : str
            The fully formatted prompt to send to the LLM.

        Returns
        -------
        str
            Raw string output from the LLM, with the prompt
            prefix stripped.

        Examples
        --------
        >>> model = UnBiasModel()  # doctest: +SKIP
        >>> output = model.generate("Analyze: Women can't lead.")  # doctest: +SKIP
        >>> isinstance(output, str)  # doctest: +SKIP
        True

        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Strip the prompt tokens, decode only new tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        return str(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
