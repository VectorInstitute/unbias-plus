"""Parser for LLM JSON output into BiasResult objects."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from json_repair import repair_json

from unbias_plus.schema import BiasResult


logger = logging.getLogger(__name__)

SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}


def parse_llm_output(raw_output: str) -> BiasResult:
    """Parse raw LLM output string into a BiasResult object.

    Handles Qwen3 thinking blocks, markdown fences, truncated JSON, and the
    common failure mode where the model embeds unescaped ``"`` characters
    inside ``unbiased_text`` (e.g. quoting the word ``"false"``). In that
    case a naive ``json.loads`` / regex fallback used to return
    ``severity`` with an empty ``biased_segments`` list.

    Strategy order:
    1. Strip thinking / extract outermost JSON object
    2. Direct ``json.loads``
    3. Escape raw control characters in strings, retry
    4. Truncation / missing-comma repairs
    5. Optional ``json_repair``
    6. Schema-aware field extraction (severity, segments, unbiased_text)

    Parameters
    ----------
    raw_output : str
        Raw string returned by the LLM.

    Returns
    -------
    BiasResult
        Validated bias analysis result.

    Raises
    ------
    ValueError
        If the output cannot be parsed into a BiasResult.
    """
    text = _strip_thinking_block(raw_output)
    candidates = _candidate_json_blobs(text)

    parsed_options: list[dict[Any, Any]] = []
    for blob in candidates:
        data = _parse_json_blob(blob)
        if data is not None:
            parsed_options.append(data)
        schema_data = _extract_schema_fields(blob)
        if schema_data is not None:
            parsed_options.append(schema_data)

    schema_full = _extract_schema_fields(text)
    if schema_full is not None:
        parsed_options.append(schema_full)

    if not parsed_options:
        raise ValueError(
            "LLM output could not be parsed as JSON after all repair attempts.\n"
            f"Raw output:\n{raw_output}"
        )

    # Prefer the parse that keeps the most segments, then the longest rewrite.
    # json_repair / early-quote truncation can yield severity + empty/short
    # unbiased_text while schema-aware extract recovers both.
    data = max(parsed_options, key=_parse_quality_score)

    if "biased_segments" in data and isinstance(data["biased_segments"], list):
        data["biased_segments"] = _deduplicate_segments(data["biased_segments"])
        data["biased_segments"] = _remove_contained_segments(data["biased_segments"])

    data = _derive_label_fields(data)

    try:
        return BiasResult(**data)
    except Exception as e:
        raise ValueError(
            f"LLM JSON does not match expected schema.\nData: {data}\nError: {e}"
        ) from e


def _parse_quality_score(data: dict[Any, Any]) -> tuple[int, int, int]:
    """Higher is better: more segments, longer rewrite, severity present."""
    segs = data.get("biased_segments")
    n_segs = len(segs) if isinstance(segs, list) else 0
    ub = data.get("unbiased_text")
    ub_len = len(ub) if isinstance(ub, str) else 0
    has_sev = 1 if data.get("severity") is not None else 0
    return (n_segs, ub_len, has_sev)


def _candidate_json_blobs(text: str) -> list[str]:
    """Return likely JSON object strings to try, longest-first."""
    stripped = text.strip()
    blobs: list[str] = []
    if stripped:
        blobs.append(stripped)

    extracted = _extract_json(stripped)
    if extracted and extracted not in blobs:
        blobs.append(extracted)

    # Prefer longer blobs first (more complete objects beat short early closes).
    blobs.sort(key=len, reverse=True)
    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for b in blobs:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def _parse_json_blob(text: str) -> dict[Any, Any] | None:
    """Try several structural repairs; return a dict or None."""
    escaped = _escape_control_chars_in_strings(text)
    for candidate in (
        text,
        escaped,
        _fix_truncated_json(text),
        _fix_truncated_json(escaped),
        _fix_missing_commas(text),
        _fix_missing_commas(_fix_truncated_json(escaped)),
    ):
        data = _try_parse(candidate)
        if data is not None:
            return data
    return _try_json_repair(escaped) or _try_json_repair(text)


def _extract_schema_fields(text: str) -> dict[Any, Any] | None:
    """Pull severity / segments / unbiased_text without requiring valid JSON.

    ``unbiased_text`` is always the last field in the trained schema, so its
    value is taken from the opening quote after the key through the *last*
    quote that precedes the final closing ``}``. That survives unescaped
    interior quotes such as ``"false"``.
    """
    severity = _extract_severity(text)
    segments = _extract_json_array_after_key(text, "biased_segments")
    unbiased = _extract_trailing_string_field(text, "unbiased_text")

    if severity is None and segments is None and unbiased is None:
        return None

    data: dict[Any, Any] = {}
    if severity is not None:
        data["severity"] = severity
    if segments is not None:
        data["biased_segments"] = segments
    else:
        data["biased_segments"] = []
    if unbiased is not None:
        data["unbiased_text"] = unbiased
    else:
        data["unbiased_text"] = ""

    # Need at least severity or some segments to be a plausible result.
    if "severity" not in data and not data["biased_segments"]:
        return None
    return data


def _extract_severity(text: str) -> int | None:
    m = re.search(r'"severity"\s*:\s*(\d+)', text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _extract_trailing_string_field(text: str, key: str) -> str | None:
    """Extract a JSON string field that is the last value before root ``}``.

    Finds ``"key": "`` then takes content until the last ``"`` before the
    final ``}``. Interior unescaped quotes (e.g. ``"false"``) are kept as
    literal characters — ``str.rfind`` picks the true closer.
    """
    token = f'"{key}"'
    idx = text.find(token)
    if idx == -1:
        return None
    i = idx + len(token)
    while i < len(text) and text[i] in " \t\r\n:":
        i += 1
    if i >= len(text) or text[i] != '"':
        return None
    content_start = i + 1

    brace = text.rfind("}")
    search_end = brace if brace > content_start else len(text)
    close_quote = text.rfind('"', content_start, search_end)
    if close_quote == -1 or close_quote < content_start:
        return _decode_json_string_content(text[content_start:search_end])
    return _decode_json_string_content(text[content_start:close_quote])


def _decode_json_string_content(content: str) -> str:
    """Decode JSON string body escapes; leave bare quotes as literals."""
    out: list[str] = []
    escape_next = False
    i = 0
    while i < len(content):
        ch = content[i]
        if escape_next:
            escapes = {
                "n": "\n",
                "r": "\r",
                "t": "\t",
                '"': '"',
                "\\": "\\",
                "/": "/",
                "b": "\b",
                "f": "\f",
            }
            if ch == "u" and i + 4 < len(content):
                hex_str = content[i + 1 : i + 5]
                if re.fullmatch(r"[0-9a-fA-F]{4}", hex_str):
                    out.append(chr(int(hex_str, 16)))
                    i += 5
                    escape_next = False
                    continue
            out.append(escapes.get(ch, ch))
            escape_next = False
            i += 1
            continue
        if ch == "\\":
            escape_next = True
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _derive_label_fields(data: dict[Any, Any]) -> dict[Any, Any]:
    """Fill ``binary_label`` / ``bias_found`` when omitted by the model."""
    out = dict(data)
    segments = out.get("biased_segments")
    if not isinstance(segments, list):
        segments = []
        out["biased_segments"] = segments

    has_segments = len(segments) > 0
    if "severity" not in out and not has_segments:
        return out

    severity = out.get("severity")
    try:
        severity_int = int(severity) if severity is not None else None
    except (TypeError, ValueError):
        severity_int = None

    if severity_int is None:
        severity_int = 3 if has_segments else 0
        out["severity"] = severity_int

    biased = severity_int > 0 or has_segments
    if "binary_label" not in out or out["binary_label"] in (None, ""):
        out["binary_label"] = "biased" if biased else "unbiased"
    if "bias_found" not in out or out["bias_found"] is None:
        out["bias_found"] = biased

    if "unbiased_text" not in out or out["unbiased_text"] is None:
        out["unbiased_text"] = ""

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _strip_thinking_block(raw_output: str) -> str:
    """Remove Qwen3 ``<think>...</think>`` block from model output."""
    if "<think>" not in raw_output:
        return raw_output

    think_idx = raw_output.find("<think>")
    brace_idx = raw_output.find("{")

    if brace_idx == -1 or think_idx < brace_idx:
        if "</think>" in raw_output:
            return raw_output.split("</think>", 1)[-1].strip()
        return ""

    return raw_output[:think_idx].strip()


def _deduplicate_segments(segments: list[dict]) -> list[dict]:
    """Merge duplicate segments that share the same original phrase."""
    seen: dict[str, dict] = {}
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        original = seg.get("original", "").strip()
        if not original:
            continue
        if original not in seen:
            seen[original] = dict(seg)
        else:
            merged = seen[original]
            existing_types = {
                t.strip() for t in str(merged.get("bias_type", "")).split("/")
            }
            new_type = str(seg.get("bias_type", "")).strip()
            if new_type and new_type not in existing_types:
                merged["bias_type"] = (
                    str(merged.get("bias_type", "")).strip() + " / " + new_type
                )
            existing_reasoning = str(merged.get("reasoning", ""))
            new_reasoning = str(seg.get("reasoning", "")).strip()
            if new_reasoning and new_reasoning not in existing_reasoning:
                merged["reasoning"] = existing_reasoning.strip() + " " + new_reasoning
            existing_rank = SEVERITY_RANK.get(
                str(merged.get("severity", "low")).lower(), 1
            )
            new_rank = SEVERITY_RANK.get(str(seg.get("severity", "low")).lower(), 1)
            if new_rank > existing_rank:
                merged["severity"] = seg["severity"]
    return list(seen.values())


def _remove_contained_segments(segments: list[dict]) -> list[dict]:
    """Remove segments whose original is fully contained in a longer segment."""
    if len(segments) <= 1:
        return segments
    sorted_segs = sorted(
        segments, key=lambda s: len(str(s.get("original", ""))), reverse=True
    )
    kept: list[dict] = []
    kept_originals: list[str] = []
    for seg in sorted_segs:
        original = str(seg.get("original", "")).strip()
        if not original:
            continue
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


def _try_json_repair(text: str) -> dict[Any, Any] | None:
    """Best-effort parse via ``json_repair``."""
    try:
        repaired = repair_json(text, return_objects=True)
    except Exception:
        return None
    if isinstance(repaired, dict):
        return repaired
    if isinstance(repaired, str):
        return _try_parse(repaired)
    return None


def _escape_control_chars_in_strings(text: str) -> str:
    """Escape raw control characters inside JSON string literals."""
    out: list[str] = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            out.append(ch)
            escape_next = False
            continue
        if ch == "\\" and in_string:
            out.append(ch)
            escape_next = True
            continue
        if ch == '"':
            out.append(ch)
            in_string = not in_string
            continue
        if in_string and ch in ("\n", "\r", "\t", "\b", "\f"):
            escapes = {
                "\n": "\\n",
                "\r": "\\r",
                "\t": "\\t",
                "\b": "\\b",
                "\f": "\\f",
            }
            out.append(escapes[ch])
            continue
        if in_string and ord(ch) < 0x20:
            out.append(f"\\u{ord(ch):04x}")
            continue
        out.append(ch)
    return "".join(out)


def _extract_json(raw_output: str) -> str:
    """Extract first balanced ``{...}`` JSON object from raw LLM output."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

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
                return raw_output[start : i + 1].strip()
            if depth < 0:
                return raw_output[start:i].strip()

    return raw_output[start:].strip()


def _fix_truncated_json(text: str) -> str:
    """Close a JSON object/array that was cut off mid-stream."""
    stack: list[str] = []
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
    """Insert missing commas between JSON key-value pairs on new lines."""
    return re.sub(
        r'(["\d\]}\w])\s*\n(\s*")',
        lambda m: f"{m.group(1)},\n{m.group(2)}",
        text,
    )


def _parse_list_blob(raw: str) -> list | None:
    """Parse a JSON array blob with the same repair ladder as objects."""
    escaped = _escape_control_chars_in_strings(raw)
    for candidate in (raw, escaped, _fix_truncated_json(escaped)):
        parsed = _try_parse_json(candidate)
        if isinstance(parsed, list):
            return parsed
    repaired = _try_json_repair('{"_":' + escaped + "}")
    value = repaired.get("_") if isinstance(repaired, dict) else None
    if isinstance(value, list):
        return value
    return None


def _extract_json_array_after_key(text: str, key: str) -> list | None:
    """Return the JSON array value for *key* using bracket counting."""
    token = f'"{key}"'
    idx = text.find(token)
    if idx == -1:
        return None
    i = idx + len(token)
    while i < len(text) and text[i] in " \t\r\n:":
        i += 1
    if i >= len(text) or text[i] != "[":
        return None
    start = i
    depth = 0
    in_string = False
    escape_next = False
    for j in range(start, len(text)):
        ch = text[j]
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
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return _parse_list_blob(text[start : j + 1])
            if depth < 0:
                break
    return _parse_list_blob(text[start:])
