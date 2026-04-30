---
hide:
  - navigation
  - toc
---

<div class="up-hero" markdown>

<!-- <div class="up-hero__eyebrow">Vector Institute · AI Engineering</div> -->

# Don't just flag bias.<br><span class="up-hero__accent">Locate it. Explain it. Fix it.</span>

<p class="up-hero__lede">
<strong>unbias-plus</strong> is an open-source Python package and applied project for bias detection and debiasing in text. It does more than flag risk: it pinpoints the exact phrase, explains the bias type and severity, and gives a neutral rewrite you can use immediately.
</p>

<p class="up-hero__lede">
Built for real workflows, it gives teams one consistent way to review language quality across editorial, trust and safety, research, and AI product pipelines.
</p>

<div class="up-hero__cta" markdown>
[Try the live demo :material-arrow-right:](https://unbias-plus.vectorinstitute.ai/){ .md-button .md-button--primary }
[GitHub](https://github.com/VectorInstitute/unbias-plus){ .md-button }
</div>

[![PyPI version](https://img.shields.io/pypi/v/unbias-plus.svg)](https://pypi.org/project/unbias-plus/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FlUSeFFXr3VeADiStOIzX7hOV6IlliC8?usp=sharing)
[![License](https://img.shields.io/github/license/VectorInstitute/unbias-plus)](https://github.com/VectorInstitute/unbias-plus/blob/main/LICENSE.md)

</div>

---

## A package you can ship, a project you can trust

- **Use as a package**: install from PyPI and run via CLI, API, or Python.
- **Use as a workflow**: apply a repeatable review standard across teams.
- **Use as evidence**: every flag includes rationale and text-level location, so decisions are explainable.

## See it in one example

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
*"reckless tax scheme"* — emotionally charged framing presents the policy as inherently irresponsible before any analysis.

<span class="up-pill up-pill--high">high · loaded language</span>
*"devastate"* — catastrophizing verb implies certainty of severe harm.

<span class="up-pill up-pill--medium">medium · framing</span>
*"everyone knows"* — appeal to consensus presents an unsupported claim as common knowledge.

<span class="up-pill up-pill--medium">medium · framing</span>
*"always"* — universal quantifier turns a tendency into an inevitability.

</div>


---

## How it works in 3 steps

<div class="grid cards" markdown>

-   :material-text-box-search-outline: __1) Analyze text__

    Pass in any sentence, paragraph, or article snippet via CLI, API, or Python.

-   :material-format-list-bulleted-type: __2) Inspect findings__

    Review each flagged segment with type, severity, rationale, and character offsets.

-   :material-file-document-edit-outline: __3) Apply neutral rewrite__

    Use the full unbiased rewrite directly, or apply segment replacements selectively.

</div>

## Three capabilities, one output

<div class="grid cards up-pillars" markdown>

-   :material-target:{ .up-pillar-icon } __Detect__

    ---

    Pinpoint biased phrases at the **character level**. Each segment ships with `start`/`end` offsets — drop them straight into a highlighter, an annotator, or a diff renderer.

-   :material-comment-question-outline:{ .up-pillar-icon } __Explain__

    ---

    Every segment comes with a **bias type**, a **severity**, and a **1–2 sentence rationale**. No black-box flag — every decision is auditable.

-   :material-pencil-outline:{ .up-pillar-icon } __Rewrite__

    ---

    Get a **neutral replacement per segment** plus a **full rewritten** version of the input. Factual content preserved, spin removed.

</div>

[How the pipeline works :material-arrow-right:](how_it_works.md){ .md-button }

---

## Why leaders care

If your team publishes, moderates, or summarizes text at scale, biased wording creates business risk: loss of trust, avoidable escalations, and inconsistent review decisions.

unbias-plus turns that into an operational workflow:

- **Consistency**: apply the same bias-checking standard across teams and channels.
- **Transparency**: each flag includes a rationale, so decisions are explainable.
- **Actionability**: get a neutral rewrite immediately, not just a warning score.
- **Integration-ready**: use the same structured output in product UI, QA checks, or policy review.

For implementation details and data shape, see the [User Guide](user_guide.md).

---

## Where teams use it

<div class="grid cards" markdown>

-   :material-newspaper-variant-outline: __Newsrooms & editors__

    Pre-publication checks for loaded framing, sensationalism, and politically charged terminology — with the exact phrases flagged.

-   :material-school-outline: __Researchers & educators__

    Build datasets, study framing effects, or teach media literacy with concrete annotated examples and reasoning trails.

-   :material-shield-check-outline: __Trust & safety teams__

    Triage user-generated content with structured signals — segment offsets, types, and rationales — instead of opaque scores.

-   :material-robot-outline: __ML & NLP teams__

    A reproducible bias-analysis stage for evaluation pipelines, RAG-style content systems, or LLM output guardrails.

</div>

---

## Try it quickly

=== "CLI"

    ```bash
    uv sync
    source .venv/bin/activate
    unbias-plus --text "Women are too emotional to lead."
    ```

=== "API + demo UI"

    ```bash
    uv sync
    source .venv/bin/activate
    unbias-plus --serve
    # open http://localhost:8000
    ```

=== "Python"

    ```python
    from unbias_plus import UnBiasPlus

    pipe = UnBiasPlus()
    result = pipe.analyze("Women are too emotional to lead.")
    print(result.binary_label)
    print(result.unbiased_text)
    ```

Need setup details, deployment patterns, or advanced usage? See the [User Guide](user_guide.md).

---

## Video tutorials

Walkthroughs of the demo UI you get when you run `unbias-plus --serve`: [:material-volume-off: Silent walkthrough](https://drive.google.com/file/d/1aNh0bqeA2rTZ-uKi_cfrljo_UHP1M4Uq/view?usp=sharing) · [:material-volume-high: Voiced walkthrough](https://drive.google.com/file/d/1uPiLQ5GZKQH7cBeuV2QQeituxFPC6zTK/view?usp=sharing)

---

## Continue exploring

<div class="grid cards" markdown>

-   [__How it works :material-arrow-right:__](how_it_works.md)

    The pipeline: prompt → fine-tuned model → parser → validated `BiasResult`.

-   [__User Guide :material-arrow-right:__](user_guide.md)

    Install, CLI, REST API, Python API, and development setup.

</div>

---

## About

Shaina Raza, PhD, Ahmed Y. Radwan, and Amrit Krishnan — AI Engineering team at the [Vector Institute](https://vectorinstitute.ai).

Code under [Apache 2.0](https://github.com/VectorInstitute/unbias-plus/blob/main/LICENSE.md); tool under the [Vector Institute custom license](https://github.com/VectorInstitute/unbias-plus/blob/unbias-pretrained/LICENSE.md).

For questions, collaboration, or licensing inquiries: [shaina.raza@vectorinstitute.ai](mailto:shaina.raza@vectorinstitute.ai). For bugs and feature requests: [GitHub Issues](https://github.com/VectorInstitute/unbias-plus/issues).

??? note "Acknowledgements"
    Resources used in preparing this research are provided, in part, by the Province of Ontario, the Government of Canada through CIFAR, and companies sponsoring the Vector Institute. This research is also supported by the European Union's Horizon Europe research and innovation programme under the **AIXPERT** project (Grant Agreement No. 101214389).
