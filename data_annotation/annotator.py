"""Core annotation logic: model call, span resolution, and record building."""

import json
import logging
import random
import time

from json_repair import repair_json

from cleaning import DOUBLE_QUOTES, strip_quotes
from config import (
    BASE_SLEEP,
    DEFAULT_CALL_SLEEP,
    MAX_COMPLETION_TOKENS,
    MAX_RETRIES,
    MODEL,
    VALID_BIAS_TYPES,
    VALID_SEGMENT_SEVERITY,
)
from prompts import ANNOTATE_SYSTEM_PROMPT, ANNOTATE_TOOL


log = logging.getLogger("annotate")


def call_tool(client, article_text):
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                messages=[
                    {"role": "system", "content": ANNOTATE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Article:\n{article_text}"},
                ],
                tools=[ANNOTATE_TOOL],
                tool_choice={
                    "type": "function",
                    "function": {"name": "submit_bias_annotation"},
                },
            )

            if response.choices[0].finish_reason == "length":
                raise ValueError("response was truncated")

            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                raise ValueError("model did not call submit_bias_annotation")

            raw_args = tool_calls[0].function.arguments

            try:
                return json.loads(raw_args)
            except json.JSONDecodeError:
                return json.loads(repair_json(raw_args))

        except Exception as error:
            last_error = error
            sleep_time = BASE_SLEEP * (2 ** (attempt - 1))

            log.warning(
                f"attempt {attempt}/{MAX_RETRIES} failed: {error}; "
                f"retrying in {sleep_time}s"
            )
            time.sleep(sleep_time)

    raise last_error


def find_unique_unused_span(text, original, used_spans, search_from=0):
    matches = []
    start = 0

    while True:
        start = text.find(original, start)

        if start == -1:
            break

        end = start + len(original)
        overlaps = any(
            start < used_end and end > used_start
            for used_start, used_end in used_spans
        )

        if not overlaps:
            matches.append(start)

        start += 1

    if not matches:
        if original in text:
            # The phrase exists but every occurrence is already claimed by an
            # earlier segment: the model over-flagged a recurring phrase more
            # times than it appears. Skip this redundant duplicate.
            log.warning(
                f"duplicate segment; all occurrences already used, "
                f"skipping: {original[:100]!r}"
            )
        else:
            # The phrase is not in the article at all (model hallucinated or
            # altered the text). Skip rather than fail the whole article.
            log.warning(f"segment not found, skipping: {original[:100]!r}")
        return None

    if len(matches) == 1:
        return matches[0]

    # Multiple occurrences: assume segments are emitted in reading order and
    # pick the first unused occurrence at or after the previously resolved
    # segment. The already-located neighbors bracket the ambiguous one.
    for candidate in matches:
        if candidate >= search_from:
            log.warning(
                f"ambiguous segment; {len(matches)} occurrences, "
                f"picked offset {candidate} (>= {search_from}): "
                f"{original[:100]!r}"
            )
            return candidate

    log.warning(
        f"ambiguous segment; {len(matches)} occurrences, none after "
        f"{search_from}, picked first offset {matches[0]}: {original[:100]!r}"
    )
    return matches[0]


def build_segments(raw_segments, article_text):
    segments = []
    used_spans = []
    search_from = 0

    for segment in raw_segments:
        original = segment.get("original")
        bias_type = segment.get("bias_type")
        severity = segment.get("severity")

        if not isinstance(original, str) or not original:
            continue

        if bias_type not in VALID_BIAS_TYPES:
            continue

        if severity not in VALID_SEGMENT_SEVERITY:
            continue

        char_start = find_unique_unused_span(
            article_text,
            original,
            used_spans,
            search_from,
        )

        if char_start is None:
            continue

        char_end = char_start + len(original)
        used_spans.append((char_start, char_end))
        search_from = char_end

        segments.append(
            {
                "original": original,
                "char_start": char_start,
                "char_end": char_end,
                "replacement": segment.get("replacement", ""),
                "severity": severity,
                "bias_type": bias_type,
                "reasoning": segment.get("reasoning", ""),
            }
        )

    return sorted(segments, key=lambda segment: segment["char_start"])


def build_record(idx, unique_id, article_text, parsed):
    severity = parsed.get("severity")
    raw_segments = parsed.get("biased_segments", [])
    unbiased_text = parsed.get("unbiased_text")

    if not isinstance(severity, int) or not 0 <= severity <= 10:
        raise ValueError(f"invalid article severity: {severity!r}")

    if not isinstance(raw_segments, list):
        raise ValueError("biased_segments must be a list")

    if not isinstance(unbiased_text, str):
        raise ValueError("unbiased_text must be a string")

    segments = build_segments(raw_segments, article_text)

    if severity == 0 and segments:
        raise ValueError("severity=0 but segments were returned")

    if severity > 0 and not segments:
        raise ValueError("severity>0 but no valid segments were returned")

    return {
        "index": idx,
        "unique_id": unique_id,
        "article_text": article_text,
        "word_count": len(article_text.split()),
        "binary_label": "biased" if severity > 0 else "unbiased",
        "severity": severity,
        "bias_found": severity > 0,
        "biased_segments": segments,
        "unbiased_text": unbiased_text,
    }


def annotate_one(client, idx, row, sleep=DEFAULT_CALL_SLEEP):
    unique_id = row["unique_id"]
    raw_article_text = row["article_text"]
    article_text = strip_quotes(raw_article_text)

    if sleep > 0:
        time.sleep(sleep + random.uniform(0, sleep))

    log.info(
        f"[{idx}] quotes before={len(DOUBLE_QUOTES.findall(raw_article_text))} "
        f"after={len(DOUBLE_QUOTES.findall(article_text))}"
    )

    start_time = time.monotonic()

    log.info(f"[{idx}] {unique_id} start")

    try:
        parsed = call_tool(client, article_text)
        record = build_record(idx, unique_id, article_text, parsed)

        elapsed = time.monotonic() - start_time
        log.info(
            f"[{idx}] {unique_id} done in {elapsed:.1f}s "
            f"| severity={record['severity']} "
            f"| segments={len(record['biased_segments'])}"
        )

        return record

    except Exception as error:
        elapsed = time.monotonic() - start_time
        log.error(f"[{idx}] {unique_id} failed after {elapsed:.1f}s: {error}")

        return {
            "index": idx,
            "unique_id": unique_id,
            "article_text": article_text,
            "word_count": len(article_text.split()),
            "binary_label": "unbiased",
            "severity": 0,
            "bias_found": False,
            "biased_segments": [],
            "unbiased_text": article_text,
            "annotation_error": str(error),
        }
