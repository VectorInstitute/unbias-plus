"""Parser for LLM JSON output into BiasResult objects."""

from unbias_plus.schema import BiasResult


def parse_llm_output(raw_output: str) -> BiasResult:
    """Parse raw LLM output string into a BiasResult object.

    Parameters
    ----------
    raw_output : str
        Raw JSON string returned by the LLM.

    Returns
    -------
    BiasResult
        Validated and structured bias analysis result.

    Raises
    ------
    ValueError
        If the output cannot be parsed as valid JSON or does
        not match the expected schema.

    Examples
    --------
    >>> result = parse_llm_output('{"binary_label": "biased", ...}')
    >>> isinstance(result, BiasResult)
    True

    """
    pass


def _extract_json(raw_output: str) -> str:
    """Extract JSON block from raw LLM output.

    Parameters
    ----------
    raw_output : str
        Raw string that may contain JSON wrapped in markdown
        code blocks or extra text.

    Returns
    -------
    str
        Cleaned JSON string ready for parsing.

    """
    pass
