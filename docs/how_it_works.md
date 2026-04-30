# How it works

unbias-plus is a thin pipeline around a fine-tuned language model. The interesting part isn't the model alone — it's that the model is prompted, parsed, and validated to return a **structured, auditable result** instead of free-form prose. This page walks through what happens between calling `analyze()` and getting a `BiasResult` back.

---

## The pipeline at a glance

<div class="up-flow" markdown>

<div class="up-flow__step" markdown>
**Input**{ .up-flow__title }

Raw text — a sentence, paragraph, or full article. Passed via CLI, REST, or `pipe.analyze(text)`.
</div>

<div class="up-flow__arrow">→</div>

<div class="up-flow__step" markdown>
**Prompt**{ .up-flow__title }

A system prompt fixes the bias categories, segment rules, and the JSON output schema.
</div>

<div class="up-flow__arrow">→</div>

<div class="up-flow__step" markdown>
**Model**{ .up-flow__title }

Fine-tuned `Qwen3-8B-UnBias-Plus-SFT-Instruct` generates the structured JSON. 4-bit quantization is optional.
</div>

<div class="up-flow__arrow">→</div>

<div class="up-flow__step" markdown>
**Parse & validate**{ .up-flow__title }

The JSON is extracted, validated with Pydantic, and character offsets are attached to each segment.
</div>

<div class="up-flow__arrow">→</div>

<div class="up-flow__step up-flow__step--out" markdown>
**`BiasResult`**{ .up-flow__title }

A typed object: render it, log it, persist it, pipe it anywhere.
</div>

</div>

---

## What the model is actually told to do

The system prompt defines a tight contract. Severity scales, segment rules, output shape — all explicit, so the parsing step has something stable to verify against. A condensed view:

```text
## BIAS TYPES
- loaded language             : words with strong emotional connotations
- dehumanizing framing        : language that strips dignity from groups
- false generalizations       : sweeping statements ("they always", "all of them")
- framing bias                : selective wording that implies a viewpoint
- euphemism/dysphemism        : softening or hardening language
- politically charged terminology : labels used to provoke rather than describe
- sensationalism              : exaggerated language to evoke emotional responses

## SEGMENT RULES
- "original" MUST be the EXACT substring as it appears in the input.
- Prefer fewer, longer segments over many short overlapping ones.
- Only modify phrases listed in biased_segments; preserve all factual content.

## OUTPUT SCHEMA (return ONLY valid JSON, no extra text)
{
  "binary_label": "biased" | "unbiased",
  "severity": 1-5,
  "bias_found": true | false,
  "biased_segments": [...],
  "unbiased_text": "Full rewritten neutral article"
}
```

Two design choices follow from this:

!!! info "Exact substrings, not paraphrases"

    The model is instructed to return the **exact original substring** for each biased segment. In practice, offset matching first tries direct substring lookup, then a small normalization fallback (trimmed text + quote/dash normalization). If no match is found, the segment is still returned and `start`/`end` remain `None`.

!!! info "Preserve facts, change framing"

    The rewriter is told to modify only the segments it flagged. Names, numbers, places, and quotes stay; loaded framing around them changes. The unbiased text shouldn't read as a *different* article — just a calmer one.

---

## The structured output is the point

Every public method ultimately returns a `BiasResult`. The schema is small enough to memorize and rich enough to drive a UI:

```python
class BiasedSegment(BaseModel):
    original: str          # exact substring from input
    replacement: str       # neutral alternative
    severity: str          # "low" | "medium" | "high"
    bias_type: str         # usually one core category; may be merged (e.g. "A / B")
    reasoning: str         # 1-2 sentence explanation
    start: int | None      # character offset (computed)
    end: int | None        # character offset (computed)


class BiasResult(BaseModel):
    binary_label: str               # "biased" | "unbiased"
    severity: int                   # 1-5 (article-level)
    bias_found: bool
    biased_segments: list[BiasedSegment]
    unbiased_text: str              # full neutral rewrite
    original_text: str | None
```

That shape is what unlocks the practical applications. With offsets and reasoning attached:

- A **frontend** can render the original text with biased phrases highlighted in place — and show the rationale on hover.
- A **moderation queue** can sort items by `severity` and route only `severity >= 3` to a human reviewer.
- An **annotation pipeline** can persist segments as labeled spans for training or analysis.
- A **diff view** can show the original and the rewritten text side by side, segment-by-segment.

A free-form "this text is biased because…" paragraph from a stock LLM doesn't enable any of that without bespoke parsing.

---

## How offsets get computed

The model returns the biased phrase as it appeared in the input, but it doesn't return character positions. `schema.compute_offsets()` walks the original text with a cursor:

1. For each segment, search for the exact phrase starting from the cursor's current position.
2. If found, attach `start` and `end` and advance the cursor past `end`. This handles repeated phrases — *"always"* appearing twice gets two distinct offset pairs in order of appearance.
3. If not found from the cursor, fall back to a search from position 0. If still missing, log a warning and leave offsets unset (the segment is still returned).
4. After processing all segments, sort by `start`.

The result: a list you can iterate in reading order, with positions that map cleanly onto the original text for highlighting.

---

## Three entry points, one pipeline

<div class="up-entry" markdown>

<div class="up-entry__col" markdown>
**CLI**{ .up-entry__title }

`unbias-plus --text "…"`, `--file`, `--serve` — quick experiments, scripts, batch runs.
</div>

<div class="up-entry__col" markdown>
**REST API**{ .up-entry__title }

`unbias-plus --serve` — `POST /analyze`, `GET /health`, plus a bundled demo UI.
</div>

<div class="up-entry__col" markdown>
**Python API**{ .up-entry__title }

`from unbias_plus import UnBiasPlus` — embed in pipelines, notebooks, services.
</div>

</div>

All three expose the same `BiasResult` shape. In local mode, CLI and Python call `pipeline.UnBiasPlus.analyze()` directly; REST uses the same local pipeline unless `VLLM_BASE_URL` is set (remote proxy mode).

---

## Performance notes

- **Default model**: [`vector-institute/Qwen3-8B-UnBias-Plus-SFT-Instruct`](https://huggingface.co/vector-institute/Qwen3-8B-UnBias-Plus-SFT-Instruct) — fine-tuned from Qwen3-8B for this task.
- **VRAM**: roughly 16 GB at BF16; pass `load_in_4bit=True` (or `--load-in-4bit` on the CLI) to fit on smaller GPUs at the cost of some replacement quality.
- **Input limits**: model context is 8,192 tokens. REST additionally enforces `MAX_INPUT_CHARS` (default `5000`, configurable via env var). CLI/Python do not enforce a fixed character cap; practical limits still come from model context and available memory.
- **CPU**: supported via `device="cpu"`, but expect slow inference. GPU is recommended for any real workload.

---

## Next

<div class="grid cards" markdown>

-   [__User Guide :material-arrow-right:__](user_guide.md)

    Install and run unbias-plus locally — CLI, REST, or Python.

-   [__GitHub repository :material-arrow-right:__](https://github.com/VectorInstitute/unbias-plus)

    Source code, issues, releases, and full module reference in `src/unbias_plus`.

</div>
