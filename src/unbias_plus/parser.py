"""Parser for LLM JSON output into BiasResult objects."""

import json
import re
from typing import Any

from unbias_plus.schema import BiasResult


def parse_llm_output(raw_output: str) -> BiasResult:
    """Parse raw LLM output string into a BiasResult object.

    Attempts multiple strategies to extract and parse a JSON object
    from the raw LLM output, then validates it against the BiasResult schema.

    Strategies (in order):
    1. Direct JSON parse of extracted block
    2. Fix truncated strings (LLM cut off mid-output)
    3. Fix missing commas between JSON items
    4. Aggressive key-by-key extraction as last resort

    Parameters
    ----------
    raw_output : str
        Raw string returned by the LLM, may include extra text,
        markdown code fences, or be truncated/malformed.

    Returns
    -------
    BiasResult
        Validated and structured bias analysis result.

    Raises
    ------
    ValueError
        If the output cannot be parsed as valid JSON or does
        not match the expected BiasResult schema after all
        repair attempts.

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

    # Strategy 1: Direct parse
    data = _try_parse(cleaned)

    # Strategy 2: Fix truncated JSON (most common LLM failure)
    if data is None:
        data = _try_parse(_fix_truncated_json(cleaned))

    # Strategy 3: Fix missing commas
    if data is None:
        data = _try_parse(_fix_missing_commas(cleaned))

    # Strategy 4: Fix truncated + missing commas combined
    if data is None:
        data = _try_parse(_fix_missing_commas(_fix_truncated_json(cleaned)))

    # Strategy 5: Regex-based field extraction (last resort)
    if data is None:
        data = _extract_fields_by_regex(raw_output)

    if data is None:
        raise ValueError(
            f"LLM output could not be parsed as JSON after all repair attempts.\n"
            f"Raw output:\n{raw_output}"
        )

    try:
        return BiasResult(**data)
    except Exception as e:
        raise ValueError(
            f"LLM JSON does not match expected schema.\nData: {data}\nError: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _try_parse_json(text: str) -> Any | None:
    """Return parsed JSON value or None on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _try_parse(text: str) -> dict[Any, Any] | None:
    """Return parsed JSON object (dict) or None on failure."""
    parsed = _try_parse_json(text)
    if isinstance(parsed, dict):
        return parsed
    return None


def _extract_json(raw_output: str) -> str:
    """Extract a JSON block from raw LLM output.

    Handles markdown code fences and leading/trailing prose.

    Parameters
    ----------
    raw_output : str
        Raw string that may contain JSON wrapped in markdown
        code blocks or surrounded by extra text.

    Returns
    -------
    str
        Best-effort JSON string ready for parsing.
    """
    # Strip markdown code fences (greedy to grab full block)
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw_output, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    # Find the outermost { ... } block using brace counting
    # (greedy regex fails on nested braces with truncation)
    start = raw_output.find("{")
    if start == -1:
        return raw_output.strip()

    # Walk forward tracking brace depth to find where JSON ends
    # (or where it was cut off)
    depth = 0
    last_valid_end = start
    in_string = False
    escape_next = False

    for i, ch in enumerate(raw_output[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_valid_end = i + 1
                break
            if depth < 0:
                break
        last_valid_end = i + 1  # track furthest point reached

    return raw_output[start:last_valid_end].strip()


def _fix_truncated_json(text: str) -> str:
    """Attempt to close a JSON object that was cut off mid-stream.

    LLMs running out of tokens often leave strings or arrays open.
    This function counts open structures and appends the necessary
    closing characters.

    Parameters
    ----------
    text : str
        Potentially truncated JSON string.

    Returns
    -------
    str
        JSON string with best-effort closing brackets/braces appended.
    """
    # Track structure with a stack
    stack = []
    in_string = False
    escape_next = False

    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()

    # If we ended mid-string, close the string first
    if in_string:
        text += '"'

    # Close any open arrays/objects in reverse order
    closers = {"{": "}", "[": "]"}
    text += "".join(closers[ch] for ch in reversed(stack))

    return text


def _fix_missing_commas(text: str) -> str:
    """Insert missing commas between JSON key-value pairs.

    Some LLMs omit commas between items, especially in long outputs.

    Parameters
    ----------
    text : str
        JSON string that may have missing commas.

    Returns
    -------
    str
        JSON string with commas inserted where clearly missing.
    """
    # Insert comma between } or ] or " or number/bool/null followed by "
    # e.g. ..."value"\n  "key": ... -> ..."value",\n  "key": ...
    return re.sub(
        r'(["\d\]}\w])\s*\n(\s*")',
        lambda m: f"{m.group(1)},\n{m.group(2)}",
        text,
    )


def _extract_fields_by_regex(raw_output: str) -> dict | None:
    """Last-resort field extraction using regex patterns.

    Attempts to pull known BiasResult fields directly from the raw
    LLM output when JSON parsing has completely failed.

    Parameters
    ----------
    raw_output : str
        Raw LLM output string.

    Returns
    -------
    dict or None
        Extracted fields as a dict, or None if extraction fails.
    """
    data: dict = {}

    # binary_label
    m = re.search(r'"binary_label"\s*:\s*"([^"]+)"', raw_output)
    if m:
        data["binary_label"] = m.group(1)

    # severity
    m = re.search(r'"severity"\s*:\s*(\d+)', raw_output)
    if m:
        data["severity"] = int(m.group(1))

    # bias_found
    m = re.search(r'"bias_found"\s*:\s*(true|false)', raw_output, re.IGNORECASE)
    if m:
        data["bias_found"] = m.group(1).lower() == "true"

    # unbiased_text — grab whatever we can, even if truncated
    m = re.search(r'"unbiased_text"\s*:\s*"(.*?)(?:"|$)', raw_output, re.DOTALL)
    if m:
        data["unbiased_text"] = m.group(1).replace('\\"', '"').strip()

    # biased_segments — try to grab full array, fall back to empty list
    m = re.search(r'"biased_segments"\s*:\s*(\[.*?\])\s*[,}]', raw_output, re.DOTALL)
    if m:
        segments = _try_parse_json(m.group(1))
        data["biased_segments"] = segments if isinstance(segments, list) else []
    else:
        data["biased_segments"] = []

    # Only return if we got the minimum required fields
    required = {"binary_label", "severity", "bias_found", "biased_segments"}
    if required.issubset(data.keys()):
        return data

    return None
