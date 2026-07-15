"""Dataset loading, validation, curation, and weighted formatting.

The single source of training data is the Hub dataset
``vector-institute/unbias-plus-dataset``, config ``train_4`` (the newest,
highest-quality span-level split). Rows are validated, a deterministic held-out
test set is carved off (never trained on), recipe-specific curation is applied,
and each surviving row is formatted into a token sequence with per-token loss
weights (and optional anti-laundering masks) for the shared weighted trainer.

See https://huggingface.co/datasets/vector-institute/unbias-plus-dataset.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from recipes.config import (
    SEGMENT_FIELDS,
    VALID_BIAS_TYPES,
    VALID_SEGMENT_SEVERITY,
    RecipeConfig,
)
from recipes.prompts import build_messages


HF_DATASET_ID = "vector-institute/unbias-plus-dataset"
HF_TRAIN_CONFIG = "train_4"

# A valid article needs enough text to be a meaningful training signal.
MIN_ARTICLE_CHARS = 200

# Regexes that flag "soft debiasing" / laundering in a replacement: the biased
# claim is hedged or attributed rather than removed. Used both to curate targets
# and to audit model outputs at eval time.
SOFT_DEBIAS_AUDIT_PATTERNS = [
    r"\bsome people (believe|think|say|argue|claim)\b",
    r"\b(critics|observers|some) (say|argue|claim|believe)\b",
    r"\b(men|women|immigrants|migrants|older|younger|black|white|muslim|jewish|asian|latino|latina|hispanic)\b[^.]{0,80}\b(may|might|often|tend|tends|commonly associated|seen as|perceived as)\b",
    r"\b(may|might|often|tend|tends|commonly associated|seen as|perceived as)\b[^.]{0,80}\b(men|women|immigrants|migrants|older|younger|black|white|muslim|jewish|asian|latino|latina|hispanic)\b",
    r"\bconcerns about whether\b[^.]{0,80}\b(men|women|older|younger|black|white|muslim|jewish|asian|latino|latina|hispanic)\b",
]

# Reasoning cue appended (when a recipe enables it) to hard-case targets so the
# rationale explicitly rejects laundering rather than merely naming the cue.
_REASONING_CUE = (
    "Hedging, attribution, or association language would preserve the biased claim."
)

# Tiny curated set of hard-neutral rewrites, anchored to observed qualitative
# failures. Only used by recipes that set ``curated_hard_case_repeat > 0``.
CURATED_HARD_CASES: list[dict[str, Any]] = [
    {
        "article_text": (
            "Men are naturally better suited for technical roles because they are more logical, "
            "competitive, and comfortable taking risks than women. In fast-paced engineering teams, "
            "women often overthink decisions or focus too much on communication and team dynamics "
            "instead of solving the technical problem directly. Although some women perform well in "
            "these fields, companies usually benefit from assigning the most demanding technical "
            "leadership roles to men."
        ),
        "binary_label": "biased",
        "severity": 9,
        "unbiased_text": "Companies should assign demanding technical leadership roles based on role-relevant qualifications.",
        "biased_segments": [
            {
                "original": "Men are naturally better suited for technical roles because they are more logical, competitive, and comfortable taking risks than women.",
                "replacement": "",
                "severity": "High",
                "bias_type": "stereotypical_association",
                "reasoning": "Claims inherent gender differences in technical suitability and traits; hedging or attribution would preserve the stereotype.",
            },
            {
                "original": "In fast-paced engineering teams, women often overthink decisions or focus too much on communication and team dynamics instead of solving the technical problem directly.",
                "replacement": "",
                "severity": "High",
                "bias_type": "stereotypical_association",
                "reasoning": "Attributes negative work behaviors to women as a group; the group claim should be removed, not softened.",
            },
            {
                "original": "Although some women perform well in these fields, companies usually benefit from assigning the most demanding technical leadership roles to men.",
                "replacement": "Companies should assign demanding technical leadership roles based on role-relevant qualifications.",
                "severity": "High",
                "bias_type": "stereotypical_association",
                "reasoning": "Frames gender as a basis for leadership assignment; the replacement uses role-relevant criteria instead.",
            },
        ],
    },
    {
        "article_text": (
            "We are seeking a candidate for our fast-paced engineering team. Previous experience shows "
            "that women and older team members have struggled to meet our technical benchmarks and "
            "delivery pace. We need someone decisive and assertive who can handle tough feedback without "
            "taking it personally."
        ),
        "binary_label": "biased",
        "severity": 8,
        "unbiased_text": (
            "We are seeking a candidate for our engineering team. Candidates should be evaluated using "
            "role-relevant technical benchmarks and delivery expectations. We need someone who can "
            "receive feedback professionally."
        ),
        "biased_segments": [
            {
                "original": "fast-paced engineering team",
                "replacement": "engineering team",
                "severity": "Low",
                "bias_type": "sensationalism",
                "reasoning": "Uses promotional pressure-oriented wording; the replacement keeps the role context without exaggeration.",
            },
            {
                "original": "Previous experience shows that women and older team members have struggled to meet our technical benchmarks and delivery pace.",
                "replacement": "Candidates should be evaluated using role-relevant technical benchmarks and delivery expectations.",
                "severity": "High",
                "bias_type": "stereotypical_association",
                "reasoning": "Links gender and age groups to poor performance; the replacement removes the demographic claim and uses role-relevant criteria.",
            },
            {
                "original": "We need someone decisive and assertive who can handle tough feedback without taking it personally.",
                "replacement": "We need someone who can receive feedback professionally.",
                "severity": "Low",
                "bias_type": "opinion_as_fact",
                "reasoning": "Presents subjective personality traits as requirements; the replacement keeps the job-relevant feedback requirement.",
            },
        ],
    },
    {
        "article_text": (
            "During a team meeting, a supervisor described a Black employee's questions about a revised "
            "project timeline as angry, hostile, and confrontational. The employee asked for clarification "
            "about workload expectations and proposed a staggered deadline to help the team meet its "
            "deliverables. The supervisor recorded the employee's comments and suggestions in the meeting notes."
        ),
        "binary_label": "biased",
        "severity": 4,
        "unbiased_text": (
            "During a team meeting, a supervisor described a Black employee's questions about a revised "
            "project timeline as requests for clarification. The employee asked for clarification about "
            "workload expectations and proposed a staggered deadline to help the team meet its deliverables. "
            "The supervisor recorded the employee's comments and suggestions in the meeting notes."
        ),
        "biased_segments": [
            {
                "original": "as angry, hostile, and confrontational",
                "replacement": "as requests for clarification",
                "severity": "Medium",
                "bias_type": "loaded_language",
                "reasoning": "Uses charged negative descriptors for workplace questions; the replacement describes the neutral action.",
            }
        ],
    },
]


# ---------------------------------------------------------------------------
# Loading + validation
# ---------------------------------------------------------------------------


def load_train4(limit: int) -> list[dict[str, Any]]:
    """Load up to ``limit`` rows of the ``train_4`` split from the Hub.

    Parameters
    ----------
    limit
        Maximum number of rows to keep.

    Returns
    -------
    list of dict
        Raw dataset rows as plain dictionaries.
    """
    dataset = load_dataset(HF_DATASET_ID, HF_TRAIN_CONFIG, split="train")
    rows: list[dict[str, Any]] = []
    for row in dataset:
        rows.append(dict(row))
        if len(rows) >= limit:
            break
    return rows


def _segment_ok(seg: Any, article: str) -> bool:
    """Return whether a single segment is well-formed for training."""
    if not isinstance(seg, dict):
        return False
    original = seg.get("original")
    replacement = seg.get("replacement", "")
    reasoning = seg.get("reasoning", "")
    return bool(
        isinstance(original, str)
        and original
        and original in article
        and isinstance(replacement, str)
        and isinstance(reasoning, str)
        and seg.get("bias_type") in VALID_BIAS_TYPES
        and seg.get("severity") in VALID_SEGMENT_SEVERITY
    )


def is_valid_sample(sample: dict[str, Any]) -> bool:
    """Return whether a raw sample is well-formed enough to train / eval on."""
    article = sample.get("article_text", "")
    unbiased = sample.get("unbiased_text", "")
    segments = sample.get("biased_segments")
    severity = sample.get("severity")

    if not isinstance(article, str) or len(article) < MIN_ARTICLE_CHARS:
        return False
    if not isinstance(unbiased, str) or not unbiased:
        return False
    if (
        not isinstance(segments, list)
        or not isinstance(severity, int)
        or not 0 <= severity <= 10
    ):
        return False

    # A biased sample must have >=1 segment; an unbiased one must have none.
    biased = sample.get("binary_label") == "biased"
    if biased == (len(segments) == 0):
        return False

    return all(_segment_ok(seg, article) for seg in segments)


def filter_valid(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only well-formed samples, reporting how many survived."""
    valid = [row for row in rows if is_valid_sample(row)]
    print(f"  Valid samples after filtering: {len(valid)} / {len(rows)}")
    return valid


def carve_heldout(
    valid: list[dict[str, Any]], heldout_size: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically split the valid pool into (train_pool, heldout).

    A seeded shuffle picks a representative held-out set (rather than the last N
    rows), and those rows are never formatted or fed to the trainer.
    """
    shuffled = list(valid)
    random.Random(seed).shuffle(shuffled)
    heldout = shuffled[:heldout_size]
    train_pool = shuffled[heldout_size:]
    print(f"  Held-out test: {len(heldout)} | Train pool: {len(train_pool)}")
    return train_pool, heldout


def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    """Write records to a JSONL file (one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records)} rows -> {path}")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


# ---------------------------------------------------------------------------
# Curation
# ---------------------------------------------------------------------------


def is_hard_debias_sample(sample: dict[str, Any], soft_types: set[str]) -> bool:
    """Return whether a sample contains a hard (soft-debias-prone) segment."""
    return any(
        isinstance(seg, dict) and seg.get("bias_type") in soft_types
        for seg in sample.get("biased_segments", [])
    )


def upsample_hard_debias_samples(
    records: list[dict[str, Any]], cfg: RecipeConfig
) -> list[dict[str, Any]]:
    """Duplicate hard-case rows per the recipe's upsampling knobs.

    Supports either an integer-style ``multiplier`` (>=1 adds full copies) or an
    ``extra_fraction`` (adds a fractional resample with replacement). If both are
    at their no-op defaults, the pool is returned unchanged.
    """
    soft_types = set(cfg.data.soft_debias_bias_types)
    hard = [r for r in records if is_hard_debias_sample(r, soft_types)]
    rng = random.Random(cfg.seed)

    multiplier = cfg.data.hard_upsample_multiplier
    extra_fraction = cfg.data.hard_upsample_extra_fraction

    if multiplier > 1:
        expanded = list(records)
        for _ in range(int(round(multiplier)) - 1):
            expanded.extend(hard)
        rng.shuffle(expanded)
        print(
            f"  Hard debias samples: {len(hard)} | multiplier: {multiplier} | "
            f"train rows after upsampling: {len(expanded)}"
        )
        return expanded

    if extra_fraction > 0 and hard:
        extra_n = int(round(len(hard) * extra_fraction))
        extras = [rng.choice(hard) for _ in range(extra_n)]
        expanded = [*records, *extras]
        rng.shuffle(expanded)
        print(
            f"  Hard debias samples: {len(hard)} | extra fraction: {extra_fraction:.2f} | "
            f"extra rows: {len(extras)} | train rows after upsampling: {len(expanded)}"
        )
        return expanded

    return records


def has_soft_debias_target(sample: dict[str, Any], soft_types: set[str]) -> bool:
    """Return whether any hard-case replacement matches a laundering pattern."""
    regexes = [re.compile(p, re.IGNORECASE) for p in SOFT_DEBIAS_AUDIT_PATTERNS]
    for seg in sample.get("biased_segments", []):
        if seg.get("bias_type") not in soft_types:
            continue
        replacement = seg.get("replacement", "") or ""
        if any(rx.search(replacement) for rx in regexes):
            return True
    return False


def filter_soft_debias_targets(
    records: list[dict[str, Any]], cfg: RecipeConfig
) -> list[dict[str, Any]]:
    """Drop rows whose hard-case targets themselves launder the bias."""
    if not cfg.data.filter_soft_debias_targets:
        return records
    soft_types = set(cfg.data.soft_debias_bias_types)
    kept = [r for r in records if not has_soft_debias_target(r, soft_types)]
    print(f"  Dropped soft-debias-like target rows: {len(records) - len(kept)}")
    return kept


def add_curated_hard_cases(
    records: list[dict[str, Any]], cfg: RecipeConfig
) -> list[dict[str, Any]]:
    """Append repeated curated hard-neutral exemplars to anchor the policy."""
    repeat = cfg.data.curated_hard_case_repeat
    if repeat <= 0:
        return records
    additions: list[dict[str, Any]] = []
    for _ in range(repeat):
        additions.extend(CURATED_HARD_CASES)
    out = [*records, *additions]
    random.Random(cfg.seed).shuffle(out)
    print(
        f"  Added curated hard-neutral cases: {len(additions)} "
        f"({len(CURATED_HARD_CASES)} x {repeat})"
    )
    return out


def audit_soft_debias_targets(
    records: list[dict[str, Any]], soft_types: set[str], max_examples: int = 8
) -> int:
    """Print and count target replacements that look like laundering."""
    regexes = [re.compile(p, re.IGNORECASE) for p in SOFT_DEBIAS_AUDIT_PATTERNS]
    hits: list[tuple[int, int, str, str, str]] = []
    for i, sample in enumerate(records):
        for j, seg in enumerate(sample.get("biased_segments", [])):
            if seg.get("bias_type") not in soft_types:
                continue
            replacement = seg.get("replacement", "") or ""
            if any(rx.search(replacement) for rx in regexes):
                hits.append(
                    (
                        i,
                        j,
                        seg.get("bias_type", "?"),
                        seg.get("original", ""),
                        replacement,
                    )
                )

    print(f"  Soft-debias-like target replacements: {len(hits)}")
    for i, j, bias_type, orig, repl in hits[:max_examples]:
        print(f"  --- audit hit sample={i} segment={j} bias_type={bias_type}")
        print(f"      original    : {orig[:200]!r}")
        print(f"      replacement : {repl[:200]!r}")
    return len(hits)


def apply_curation(
    train_pool: list[dict[str, Any]], cfg: RecipeConfig
) -> list[dict[str, Any]]:
    """Run the recipe's curation pipeline over the train pool, in order."""
    soft_types = set(cfg.data.soft_debias_bias_types)
    audit_soft_debias_targets(train_pool, soft_types)
    train_pool = filter_soft_debias_targets(train_pool, cfg)
    train_pool = add_curated_hard_cases(train_pool, cfg)
    return upsample_hard_debias_samples(train_pool, cfg)


# ---------------------------------------------------------------------------
# Weighted formatting
# ---------------------------------------------------------------------------


def _clean_segment(segment: dict[str, Any], cfg: RecipeConfig) -> dict[str, Any]:
    """Keep only the fields the model should generate, in a fixed order.

    Optionally augments the reasoning of hard-case segments with an explicit
    anti-laundering cue when the recipe enables it.
    """
    cleaned = {field: segment.get(field, "") for field in SEGMENT_FIELDS}
    if cfg.data.augment_reasoning_cue and cleaned.get("bias_type") in set(
        cfg.data.soft_debias_bias_types
    ):
        reason = str(cleaned.get("reasoning", "")).strip()
        if _REASONING_CUE.lower() not in reason.lower():
            cleaned["reasoning"] = (reason + " " + _REASONING_CUE).strip()
    return cleaned


def _completion_dict(sample: dict[str, Any], cfg: RecipeConfig) -> dict[str, Any]:
    """Build the assistant JSON target from a sample."""
    return {
        "severity": sample["severity"],
        "biased_segments": [
            _clean_segment(seg, cfg) for seg in sample["biased_segments"]
        ],
        "unbiased_text": sample["unbiased_text"],
    }


def _field_weight_for_segment(field: str, bias_type: str, cfg: RecipeConfig) -> float:
    """Return the loss weight for a segment field, honoring bias-type weighting."""
    weights = cfg.weights
    if field == "original":
        return weights.original
    btw = cfg.bias_type_weighting
    if field == "replacement":
        if btw is None:
            return weights.replacement
        mult = btw.field_multipliers.get(bias_type, {}).get("replacement", 1.0)
        return min(weights.replacement * mult, btw.max_replacement)
    if field == "reasoning":
        if btw is None:
            return weights.reasoning
        mult = btw.field_multipliers.get(bias_type, {}).get("reasoning", 1.0)
        return min(weights.reasoning * mult, btw.max_reasoning)
    return weights.base_completion


def _rewrite_weight(completion: dict[str, Any], cfg: RecipeConfig) -> float:
    """Return the loss weight for the ``unbiased_text`` rewrite."""
    btw = cfg.bias_type_weighting
    if btw is None:
        return cfg.weights.unbiased_text
    mult = 1.0
    for seg in completion.get("biased_segments", []) or []:
        mult = max(mult, btw.rewrite_multipliers.get(seg.get("bias_type"), 1.0))
    return min(cfg.weights.unbiased_text * mult, btw.max_unbiased_text)


def _json_value_span_after(
    text: str, field: str, cursor: int
) -> tuple[tuple[int, int] | None, int]:
    """Return the char span of a JSON field value appearing after ``cursor``.

    The span includes JSON string quotes so empty replacements (``""``) still
    receive replacement loss weight. Returns the span (or ``None``) and the new
    cursor position to continue scanning from.
    """
    key = json.dumps(field, ensure_ascii=False) + ":"
    key_pos = text.find(key, cursor)
    if key_pos < 0:
        return None, cursor

    value_start = key_pos + len(key)
    while value_start < len(text) and text[value_start] in " \t\r\n":
        value_start += 1

    decoder = json.JSONDecoder()
    try:
        _, rel_end = decoder.raw_decode(text[value_start:])
    except json.JSONDecodeError:
        return None, key_pos + len(key)

    value_end = value_start + rel_end
    return (value_start, value_end), value_end


def _weighted_value_spans(
    completion: dict[str, Any], assistant_turn: str, cfg: RecipeConfig
) -> list[tuple[int, int, float, str]]:
    """Find weighted (start, end, weight, field) spans in the assistant JSON."""
    spans: list[tuple[int, int, float, str]] = []
    cursor = 0

    # Article-level severity is left at base weight.
    _span, cursor = _json_value_span_after(assistant_turn, "severity", cursor)

    for seg in completion["biased_segments"]:
        bias_type = str(seg.get("bias_type", ""))
        for field in SEGMENT_FIELDS:
            span, cursor = _json_value_span_after(assistant_turn, field, cursor)
            if span is not None and field in {"original", "replacement", "reasoning"}:
                start, end = span
                weight = _field_weight_for_segment(field, bias_type, cfg)
                spans.append((start, end, weight, field))

    span, cursor = _json_value_span_after(assistant_turn, "unbiased_text", cursor)
    if span is not None:
        start, end = span
        spans.append((start, end, _rewrite_weight(completion, cfg), "unbiased_text"))

    return spans


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Return whether two half-open char intervals overlap."""
    return a_start < b_end and b_start < a_end


def make_forbidden_token_ids(tokenizer: Any, strings: list[str]) -> list[int]:
    """Tokenize the forbidden soft-debias strings into a sorted id set."""
    ids: set[int] = set()
    for text in strings:
        ids.update(
            int(t) for t in tokenizer(text, add_special_tokens=False)["input_ids"]
        )
    for special_id in (
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        tokenizer.bos_token_id,
    ):
        if special_id is not None:
            ids.discard(int(special_id))
    return sorted(ids)


def make_phrase_token_sequences(tokenizer: Any, phrases: list[str]) -> list[list[int]]:
    """Tokenize laundering phrases into unique id sequences (longest first)."""
    seqs: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    specials = {tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id}
    for phrase in phrases:
        variants = [phrase] if phrase.startswith(" ") else [phrase, " " + phrase]
        for variant in variants:
            ids = [
                int(t)
                for t in tokenizer(variant, add_special_tokens=False)["input_ids"]
                if t not in specials
            ]
            key = tuple(ids)
            if ids and key not in seen:
                seen.add(key)
                seqs.append(ids)
    seqs.sort(key=len, reverse=True)
    return seqs


def _phrase_forbidden_ids_at_position(
    input_ids: list[int],
    idx: int,
    phrase_token_sequences: list[list[int]],
    max_ids: int,
) -> list[int]:
    """Return forbidden next-token ids for phrase-level unlikelihood at ``idx``."""
    out: list[int] = []
    seen: set[int] = set()
    for phrase_ids in phrase_token_sequences:
        if not phrase_ids:
            continue
        # k = number of phrase tokens already matched immediately before idx;
        # k=0 forbids starting the phrase here.
        max_k = min(len(phrase_ids) - 1, idx)
        for k in range(max_k, -1, -1):
            if k > 0 and input_ids[idx - k : idx] != phrase_ids[:k]:
                continue
            next_id = int(phrase_ids[k])
            if next_id not in seen:
                seen.add(next_id)
                out.append(next_id)
            break
        if len(out) >= max_ids:
            break
    return out[:max_ids]


def phrase_width(cfg: RecipeConfig) -> int:
    """Return the fixed per-position width of the phrase-forbid tensor."""
    if cfg.phrase_unlikelihood is not None:
        return max(1, cfg.phrase_unlikelihood.max_ids_per_pos)
    return 1


def format_sample(
    sample: dict[str, Any],
    tokenizer: Any,
    cfg: RecipeConfig,
    phrase_token_sequences: list[list[int]],
) -> dict[str, Any]:
    """Format one sample into ids, labels, loss weights, and unlikelihood masks."""
    completion = _completion_dict(sample, cfg)
    messages = build_messages(cfg.prompt, sample["article_text"])
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    assistant_turn = json.dumps(completion, ensure_ascii=False, indent=2)
    eos = tokenizer.eos_token or ""
    full_text = prompt + assistant_turn + eos
    prompt_chars = len(prompt)

    enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids: list[int] = list(enc["input_ids"])
    attention_mask: list[int] = list(enc["attention_mask"])
    offsets = enc["offset_mapping"]

    assistant_spans = _weighted_value_spans(completion, assistant_turn, cfg)
    soft_types = set(cfg.data.soft_debias_bias_types)
    hard_debias = any(
        seg.get("bias_type") in soft_types
        for seg in completion.get("biased_segments", [])
    )

    labels: list[int] = []
    loss_weights: list[float] = []
    anti_soft_mask: list[float] = []

    for tok_id, (tok_start, tok_end) in zip(input_ids, offsets):
        if tok_end <= prompt_chars:
            labels.append(-100)
            loss_weights.append(0.0)
            anti_soft_mask.append(0.0)
            continue

        labels.append(tok_id)
        rel_start = max(0, tok_start - prompt_chars)
        rel_end = max(0, tok_end - prompt_chars)
        weight = cfg.weights.base_completion
        anti = 0.0
        for span_start, span_end, span_weight, field in assistant_spans:
            if _overlaps(rel_start, rel_end, span_start, span_end):
                weight = max(weight, span_weight)
                if hard_debias and field in {"replacement", "unbiased_text"}:
                    anti = 1.0
        loss_weights.append(float(weight))
        anti_soft_mask.append(anti)

    width = phrase_width(cfg)
    phrase_forbid_ids: list[list[int]] = []
    use_phrase = bool(phrase_token_sequences)
    for idx, label in enumerate(labels):
        if not use_phrase or label == -100 or anti_soft_mask[idx] <= 0:
            ids: list[int] = []
        else:
            ids = _phrase_forbidden_ids_at_position(
                input_ids, idx, phrase_token_sequences, width
            )
        phrase_forbid_ids.append((ids + [-100] * width)[:width])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "loss_weights": loss_weights,
        "anti_soft_mask": anti_soft_mask,
        "phrase_forbid_ids": phrase_forbid_ids,
    }


def format_dataset(
    records: list[dict[str, Any]],
    tokenizer: Any,
    cfg: RecipeConfig,
    phrase_token_sequences: list[list[int]],
) -> Dataset:
    """Format validated records into a weighted training :class:`Dataset`."""
    dataset = Dataset.from_list(records)
    return dataset.map(
        lambda row: format_sample(row, tokenizer, cfg, phrase_token_sequences),
        remove_columns=dataset.column_names,
        desc="Formatting weighted samples",
    )


def filter_by_token_length(dataset: Dataset, max_length: int) -> Dataset:
    """Drop formatted samples that exceed the model context window."""
    before = len(dataset)
    dataset = dataset.filter(
        lambda row: len(row["input_ids"]) <= max_length, desc="Length filter"
    )
    print(f"  Dropped {before - len(dataset)} overlength samples")
    return dataset


def print_token_stats(dataset: Dataset, max_seq_length: int) -> None:
    """Print a quick summary of the token-length distribution."""
    lengths = [len(dataset[i]["input_ids"]) for i in range(len(dataset))]
    if not lengths:
        print("  (empty dataset)")
        return
    print(f"  Samples    : {len(lengths)}")
    print(f"  Max tokens : {max(lengths)}")
    print(f"  Avg tokens : {sum(lengths) / len(lengths):.0f}")
    print(f"  > {max_seq_length} tokens : {sum(n > max_seq_length for n in lengths)}")
