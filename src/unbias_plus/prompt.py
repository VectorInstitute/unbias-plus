"""Prompt templates for the unbias-plus LLM."""


SYSTEM_PROMPT = """You are an expert bias detection and debiasing assistant.
Given a text, you must analyze it for bias and return a JSON object with:
- binary_label: 'biased' or 'unbiased'
- severity: integer from 1 (low) to 5 (high)
- bias_found: true or false
- biased_segments: list of objects, each with:
    - original: the biased phrase
    - replacement: neutral replacement
    - severity: 'low', 'medium', or 'high'
    - bias_type: type of bias
    - reasoning: explanation
- unbiased_text: full neutral rewrite of the input text
Return only valid JSON. No additional commentary."""


def build_prompt(text: str) -> str:
    """Build the full prompt for the LLM given input text.

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
    >>> isinstance(prompt, str)
    True

    """
    pass
