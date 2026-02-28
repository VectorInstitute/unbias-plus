"""Parser for LLM JSON output into BiasResult objects."""

import json
import re

from unbias_plus.schema import BiasResult


def parse_llm_output(raw_output: str) -> BiasResult:
    """Parse raw LLM output string into a BiasResult object.

    Attempts to extract and parse a JSON object from the raw
    LLM output, then validates it against the BiasResult schema.

    Parameters
    ----------
    raw_output : str
        Raw string returned by the LLM, may include extra text
        or markdown code fences around the JSON.

    Returns
    -------
    BiasResult
        Validated and structured bias analysis result.

    Raises
    ------
    ValueError
        If the output cannot be parsed as valid JSON or does
        not match the expected BiasResult schema.

    Examples
    --------
    >>> raw = '''
    ... {
    ...   "binary_label": "biased",
    ...   "severity": 3,
    ...   "bias_found": true,
    ...   "biased_segments": [],
    ...   "unbiased_text": "A neutral version."
    ... }
    ... '''
    >>> result = parse_llm_output(raw)
    >>> result.binary_label
    'biased'

    """
    cleaned = _extract_json(raw_output)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM output could not be parsed as JSON.\n"
            f"Raw output:\n{raw_output}\n"
            f"Error: {e}"
        ) from e

    try:
        return BiasResult(**data)
    except Exception as e:
        raise ValueError(
            f"LLM JSON does not match expected schema.\nData: {data}\nError: {e}"
        ) from e


def _extract_json(raw_output: str) -> str:
    """Extract a JSON block from raw LLM output.

    Handles cases where the LLM wraps JSON in markdown code
    fences (```json ... ```) or adds extra text before/after.

    Parameters
    ----------
    raw_output : str
        Raw string that may contain JSON wrapped in markdown
        code blocks or surrounded by extra text.

    Returns
    -------
    str
        Cleaned JSON string ready for parsing.

    Examples
    --------
    >>> raw = '```json\\n{\\"key\\": \\"value\\"}\\n```'
    >>> _extract_json(raw)
    '{"key": "value"}'

    """
    # Strip markdown code fences if present
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    # Find the first { ... } block
    brace_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if brace_match:
        return brace_match.group(0).strip()

    return raw_output.strip()
