# What it does

UnBias-Plus takes any piece of text, locates biased phrases at the character level, explains each one, and returns a neutral rewrite — as a structured, validated object you can drop into any pipeline.

---

## See it on one example

<div class="up-diff" markdown>

<div class="up-diff__col up-diff__col--before" markdown>

**Input**{ .up-diff__label .up-diff__label--before }

> The senator's <span class="up-bias up-bias--high">reckless tax scheme</span> will <span class="up-bias up-bias--high">devastate</span> working families, and <span class="up-bias up-bias--medium">everyone knows</span> the opposition <span class="up-bias up-bias--medium">always</span> caves at the last minute.

</div>

<div class="up-diff__col up-diff__col--after" markdown>

**Neutral rewrite**{ .up-diff__label .up-diff__label--after }

> The senator's <span class="up-fix">proposed tax plan</span> will <span class="up-fix">significantly affect</span> working families, and <span class="up-fix">commentators have noted</span> the opposition has <span class="up-fix">often</span> changed its position late in negotiations.

</div>

</div>

<div class="up-segments" markdown>

<span class="up-pill up-pill--high">high · loaded language</span>
*"reckless tax scheme"* &mdash; emotionally charged framing presents the policy as inherently irresponsible before any analysis.

<span class="up-pill up-pill--high">high · loaded language</span>
*"devastate"* &mdash; catastrophizing verb implies certainty of severe harm.

<span class="up-pill up-pill--medium">medium · framing</span>
*"everyone knows"* &mdash; appeal to consensus presents an unsupported claim as common knowledge.

<span class="up-pill up-pill--medium">medium · framing</span>
*"always"* &mdash; universal quantifier turns a tendency into an inevitability.

</div>

---

## Three capabilities, one output

<div class="grid cards up-pillars" markdown>

-   :material-target:{ .up-pillar-icon } __Detect__

    ---

    Pinpoint biased phrases at the **character level**. Each segment ships with `start` and `end` offsets, ready for a highlighter, an annotator, or a diff renderer.

-   :material-comment-question-outline:{ .up-pillar-icon } __Explain__

    ---

    Every segment carries a **bias type**, a **severity**, and a **1&ndash;2 sentence rationale**. No black-box flag &mdash; every decision is auditable.

-   :material-pencil-outline:{ .up-pillar-icon } __Rewrite__

    ---

    Get a **neutral replacement per segment**, plus a **full rewritten** version of the input. Factual content preserved; framing neutralized.

</div>

[How the pipeline works :material-arrow-right:](how_it_works.md){ .md-button }

---

## Where teams use it

<div class="grid cards" markdown>

-   :material-newspaper-variant-outline: __Newsrooms and editors__

    Pre-publication checks for loaded framing, sensationalism, and politically charged terminology, with the exact phrases flagged.

-   :material-school-outline: __Researchers and educators__

    Build datasets, study framing effects, or teach media literacy with concrete annotated examples and reasoning trails.

-   :material-shield-check-outline: __Trust and safety teams__

    Triage user-generated content with structured signals: segment offsets, types, and rationales, instead of opaque scores.

-   :material-robot-outline: __ML and NLP teams__

    A reproducible bias-analysis stage for evaluation pipelines, RAG content systems, or LLM output guardrails.

</div>
