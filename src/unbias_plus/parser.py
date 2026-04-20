"""Parser for LLM JSON output into BiasResult objects."""

import json
import re
from typing import Any

from unbias_plus.schema import BiasResult


SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}


def parse_llm_output(raw_output: str) -> BiasResult:
    """Parse raw LLM output string into a BiasResult object.

    Handles Qwen3 thinking blocks (<think>...</think>) as well as
    plain JSON output from any model. Attempts multiple strategies
    to extract and parse a JSON object from the raw LLM output,
    then validates it against the BiasResult schema.

    Strategies (in order):
    1. Fast path: direct parse of the full stripped text before any extraction
    2. Strip thinking block if present (Qwen3 with enable_thinking=True)
    3. Direct JSON parse of extracted block
    4. Fix truncated strings (LLM cut off mid-output)
    5. Fix missing commas between JSON items
    6. Aggressive key-by-key extraction as last resort

    Parameters
    ----------
    raw_output : str
        Raw string returned by the LLM, may include a thinking block,
        extra text, markdown code fences, or be truncated/malformed.

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
    text = _strip_thinking_block(raw_output)

    # Fast path: try direct parse on the full stripped text first.
    # This catches well-formed output before any extraction logic can corrupt it.
    data = _try_parse(text.strip())

    if data is None:
        cleaned = _extract_json(text)
        # NOTE: do NOT call _strip_thinking_block again here.
        # If the model appends a trailing "assistant<think>...</think>" after
        # the JSON (a common turn-delimiter artifact), a second call would see
        # the unclosed/trailing <think> tag and wipe the entire extracted string.
        # _extract_json already stops at the first closing brace, so any trailing
        # tags are already excluded from `cleaned`.
        text = cleaned

        # Strategy 1: Direct parse
        data = _try_parse(text)

        # Strategy 2: Fix truncated JSON (most common LLM failure)
        if data is None:
            data = _try_parse(_fix_truncated_json(text))

        # Strategy 3: Fix missing commas
        if data is None:
            data = _try_parse(_fix_missing_commas(text))

        # Strategy 4: Fix truncated + missing commas combined
        if data is None:
            data = _try_parse(_fix_missing_commas(_fix_truncated_json(text)))

        # Strategy 5: Regex-based field extraction (last resort)
        if data is None:
            data = _extract_fields_by_regex(text)

    if data is None:
        raise ValueError(
            f"LLM output could not be parsed as JSON after all repair attempts.\n"
            f"Raw output:\n{raw_output}"
        )

    # Deduplicate segments with the same original phrase before schema validation
    if "biased_segments" in data and isinstance(data["biased_segments"], list):
        data["biased_segments"] = _deduplicate_segments(data["biased_segments"])
        data["biased_segments"] = _remove_contained_segments(data["biased_segments"])

    try:
        return BiasResult(**data)
    except Exception as e:
        raise ValueError(
            f"LLM JSON does not match expected schema.\nData: {data}\nError: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _strip_thinking_block(raw_output: str) -> str:
    """Remove Qwen3 <think>...</think> block from model output.

    Handles three cases:
    1. Leading block: <think>...</think> before the JSON — strip and return
       everything after </think>.
    2. Trailing block: JSON then "assistant<think>..." turn delimiter — strip
       from <think> onward since the JSON is already complete before it.
    3. Unclosed leading block: model hit max_new_tokens mid-think — return
       empty string so fallback strategies handle it gracefully.

    Safe to call on any model output — if no thinking block is present
    the string is returned unchanged.

    Parameters
    ----------
    raw_output : str
        Raw LLM output, possibly containing a thinking block.

    Returns
    -------
    str
        Output with thinking block removed, ready for JSON extraction.
    """
    if "<think>" not in raw_output:
        # No thinking block — return as-is (any other model)
        return raw_output

    think_idx = raw_output.find("<think>")
    brace_idx = raw_output.find("{")

    # Case 1 & 3: <think> appears before the JSON (or there is no JSON yet)
    if brace_idx == -1 or think_idx < brace_idx:
        if "</think>" in raw_output:
            # Clean leading block: strip it and return everything after
            return raw_output.split("</think>", 1)[-1].strip()
        # Unclosed leading block — nothing useful follows
        return ""

    # Case 2: <think> appears AFTER the JSON opening brace — it is a trailing
    # turn delimiter. Strip from <think> onward; the JSON before it is intact.
    return raw_output[:think_idx].strip()


def _deduplicate_segments(segments: list[dict]) -> list[dict]:
    """Merge duplicate segments that share the same original phrase.

    When the LLM returns the same original text multiple times with
    different bias_types, this merges them into a single segment:
    - keeps the first replacement
    - joins all unique bias_types with ' / '
    - joins all unique reasonings together
    - keeps the highest severity

    Parameters
    ----------
    segments : list[dict]
        Raw list of segment dicts from the parsed JSON.

    Returns
    -------
    list[dict]
        Deduplicated list with one entry per unique original phrase.
    """
    seen: dict[str, dict] = {}
    for seg in segments:
        original = seg.get("original", "").strip()
        if not original:
            continue
        if original not in seen:
            seen[original] = dict(seg)
        else:
            merged = seen[original]
            # Merge bias_type — append only if not already present
            existing_types = {t.strip() for t in merged.get("bias_type", "").split("/")}
            new_type = seg.get("bias_type", "").strip()
            if new_type and new_type not in existing_types:
                merged["bias_type"] = merged["bias_type"].strip() + " / " + new_type
            # Merge reasoning — append only if not already present
            existing_reasoning = merged.get("reasoning", "")
            new_reasoning = seg.get("reasoning", "").strip()
            if new_reasoning and new_reasoning not in existing_reasoning:
                merged["reasoning"] = existing_reasoning.strip() + " " + new_reasoning
            # Keep highest severity
            existing_rank = SEVERITY_RANK.get(merged.get("severity", "low").lower(), 1)
            new_rank = SEVERITY_RANK.get(seg.get("severity", "low").lower(), 1)
            if new_rank > existing_rank:
                merged["severity"] = seg["severity"]
    return list(seen.values())


def _remove_contained_segments(segments: list[dict]) -> list[dict]:
    """Remove segments whose original text is fully contained within another segment.

    After deduplication, the model sometimes returns both a longer phrase
    and a shorter sub-phrase that is entirely contained within it. The longer
    segment already captures the bias — the shorter one is redundant and
    produces overlapping highlights in the frontend.

    Strategy: sort by length descending so longer segments are kept
    preferentially. For each remaining segment, drop any other segment
    whose original text appears as a substring of it.

    Parameters
    ----------
    segments : list[dict]
        Deduplicated list of segment dicts.

    Returns
    -------
    list[dict]
        Filtered list with contained sub-segments removed.
    """
    if len(segments) <= 1:
        return segments
    # Sort longest original first so we always prefer the broader segment
    sorted_segs = sorted(
        segments, key=lambda s: len(s.get("original", "")), reverse=True
    )
    kept: list[dict] = []
    kept_originals: list[str] = []
    for seg in sorted_segs:
        original = seg.get("original", "").strip()
        if not original:
            continue
        # Check if this segment's text is a substring of any already-kept segment
        is_contained = any(original in kept_orig for kept_orig in kept_originals)
        if not is_contained:
            kept.append(seg)
            kept_originals.append(original)
    return kept


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
    Uses brace counting to stop exactly at the closing } of the
    first complete root JSON object — any text after (including
    repeated copies from token-limit loops, or trailing turn
    delimiters like "assistant<think></think>") is excluded.

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
    # Non-greedy match: grab the FIRST fenced block only.
    # The original greedy `.*` would span across repeated JSON copies.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    # Find the outermost { ... } block using brace counting.
    # Return immediately on depth==0 so repeated output never bleeds in.
    start = raw_output.find("{")
    if start == -1:
        return raw_output.strip()

    depth = 0
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
                # Return immediately — everything after is noise
                # (repeated copies, trailing "assistant<think>..." etc.)
                return raw_output[start : i + 1].strip()
            if depth < 0:
                # Malformed — return what we have so far
                return raw_output[start:i].strip()

    # Never closed — return everything from start
    return raw_output[start:].strip()


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
    if in_string:
        text += '"'
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
    m = re.search(r'"binary_label"\s*:\s*"([^"]+)"', raw_output)
    if m:
        data["binary_label"] = m.group(1)
    m = re.search(r'"severity"\s*:\s*(\d+)', raw_output)
    if m:
        data["severity"] = int(m.group(1))
    m = re.search(r'"bias_found"\s*:\s*(true|false)', raw_output, re.IGNORECASE)
    if m:
        data["bias_found"] = m.group(1).lower() == "true"
    m = re.search(r'"unbiased_text"\s*:\s*"(.*?)(?:"|$)', raw_output, re.DOTALL)
    if m:
        data["unbiased_text"] = m.group(1).replace('\\"', '"').strip()
    m = re.search(r'"biased_segments"\s*:\s*(\[.*?\])\s*[,}]', raw_output, re.DOTALL)
    if m:
        segments = _try_parse_json(m.group(1))
        data["biased_segments"] = segments if isinstance(segments, list) else []
    else:
        data["biased_segments"] = []
    required = {"binary_label", "severity", "bias_found", "biased_segments"}
    if required.issubset(data.keys()):
        return data
    return None
