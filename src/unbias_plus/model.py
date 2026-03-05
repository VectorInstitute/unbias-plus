"""LLM model loader and inference for unbias-plus."""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL = "vector-institute/Qwen3-8B-UnBias-Plus-SFT"
MAX_SEQ_LENGTH = 8192


class UnBiasModel:
    """Loads and runs the fine-tuned bias detection LLM.

    Wraps a HuggingFace causal LM with a simple generate()
    interface. Compatible with any HuggingFace causal LM —
    thinking mode is opt-in for Qwen3 models only.

    Parameters
    ----------
    model_name_or_path : str | Path
        HuggingFace model ID or local path to the model.
        Defaults to 'vector-institute/Qwen3-8B-UnBias-Plus-SFT'.
    device : str | None, optional
        Device to run on ('cuda' or 'cpu').
        Auto-detects if not provided.
    load_in_4bit : bool, optional
        Load model in 4-bit quantization via bitsandbytes.
        Reduces VRAM to ~3GB (4B) or ~5GB (8B). Default is False.
    max_new_tokens : int, optional
        Maximum number of new tokens to generate. Default 2048.
    enable_thinking : bool, optional
        Enable Qwen3 chain-of-thought thinking mode. Only supported
        by Qwen3 models — do not set for other models. Default is False.
    thinking_budget : int, optional
        Maximum tokens allocated to the thinking block when
        enable_thinking=True. Default is 512.

    Examples
    --------
    >>> model = UnBiasModel()  # doctest: +SKIP
    >>> raw = model.generate([{"role": "user", "content": "..."}])  # doctest: +SKIP
    >>> isinstance(raw, str)  # doctest: +SKIP
    True
    """

    def __init__(
        self,
        model_name_or_path: str | Path = DEFAULT_MODEL,
        device: str | None = None,
        load_in_4bit: bool = False,
        max_new_tokens: int = 2048,
        enable_thinking: bool = False,
        thinking_budget: int = 512,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # --- Tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        # --- Quantization config ---
        # Default model always loads in 4bit to keep VRAM manageable (~5GB).
        # For any custom model, load_in_4bit remains opt-in.
        effective_load_in_4bit = load_in_4bit or (
            self.model_name_or_path == DEFAULT_MODEL
        )
        quantization_config = None
        if effective_load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        # --- Model ---
        # device_map={'': device_index} ensures the full model lands on one
        # specific GPU, avoiding multi-GPU conflicts from device_map="auto".
        device_index = 0 if self.device == "cuda" else self.device
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            dtype=torch.bfloat16,
            device_map={"": device_index},
            quantization_config=quantization_config,
        )
        self.model.eval()

    def generate(self, messages: list[dict]) -> str:
        """Run inference on a list of chat messages and return the raw output.

        Uses greedy decoding (do_sample=False) for deterministic, consistent
        JSON output across runs. Works with any HuggingFace causal LM.

        Parameters
        ----------
        messages : list[dict]
            List of {"role": ..., "content": ...} dicts.
            Should include system prompt and user message.

        Returns
        -------
        str
            Raw string output from the model with the input prompt stripped.
            Special tokens are removed for clean downstream parsing.

        Examples
        --------
        >>> model = UnBiasModel()  # doctest: +SKIP
        >>> msgs = [{"role": "user", "content": "..."}]  # doctest: +SKIP
        >>> output = model.generate(msgs)  # doctest: +SKIP
        >>> isinstance(output, str)  # doctest: +SKIP
        True
        """
        # Build template kwargs as a literal — only pass thinking args when
        # explicitly enabled so the code works with any HF model, not just Qwen3.
        # enable_thinking is always passed explicitly (even as False) so
        # Qwen3's jinja template doesn't fall back to its own default of True.
        template_kwargs: dict = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
            "truncation": True,
            "max_length": MAX_SEQ_LENGTH,
            # Always set enable_thinking explicitly for Qwen3 models so the
            # jinja template respects our setting rather than its own default.
            # For non-Qwen3 models this key is simply ignored by the tokenizer.
            "enable_thinking": self.enable_thinking,
        }
        if self.enable_thinking:
            template_kwargs["thinking_budget"] = self.thinking_budget

        tokenized = self.tokenizer.apply_chat_template(messages, **template_kwargs)

        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # greedy decoding — deterministic output
                temperature=None,  # must be None when do_sample=False
                top_p=None,  # must be None when do_sample=False
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens — strip the input prompt.
        # skip_special_tokens=True removes <|im_start|>, <|endoftext|> etc.
        # so the parser receives clean text without special token artifacts
        # that could corrupt JSON extraction.
        new_tokens = output_ids[0][input_ids.shape[-1] :]
        return str(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
