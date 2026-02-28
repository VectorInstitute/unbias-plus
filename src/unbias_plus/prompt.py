"""Prompt templates for the unbias-plus LLM."""

SYSTEM_PROMPT = (
    "You are an expert bias detection and debiasing assistant. "
    "Analyze the given text for bias and return ONLY a valid JSON object "
    "with no additional commentary, markdown, or code blocks.\n\n"
    "The JSON must have exactly these fields:\n"
    "- binary_label: 'biased' or 'unbiased'\n"
    "- severity: integer from 1 (low) to 5 (high)\n"
    "- bias_found: true or false\n"
    "- biased_segments: list of objects, each with:\n"
    "    - original: the biased phrase as it appears in the text\n"
    "    - replacement: a neutral replacement phrase\n"
    "    - severity: 'low', 'medium', or 'high'\n"
    "    - bias_type: type of bias detected\n"
    "    - reasoning: explanation of why this is biased\n"
    "- unbiased_text: full neutral rewrite of the entire input text\n\n"
    "Return only the JSON object. No other text."
)


def build_prompt(text: str) -> str:
    """Build the full prompt for the LLM given input text.

    Formats the system instructions and user text into a single
    prompt string suitable for Llama-3.2 text generation.

    Parameters
    ----------
    text : str
        The input text to analyze for bias.

    Returns
    -------
    str
        The formatted prompt string ready for the LLM.

    Examples
    --------
    >>> prompt = build_prompt("Women are too emotional to lead.")
    >>> "Women are too emotional to lead." in prompt
    True
    >>> "JSON" in prompt
    True

    """
    return f"{SYSTEM_PROMPT}\n\nText to analyze:\n{text}\n\nJSON output:"
