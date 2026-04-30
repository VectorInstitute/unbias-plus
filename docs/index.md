---
hide:
  - navigation
  - toc
---

<div class="up-hero" markdown>

# Don't just flag bias.<br><span class="up-hero__accent">Locate it. Explain it. Fix it.</span>

<p class="up-hero__lede">
<strong>UnBias-Plus</strong> (<code>unbias-plus</code>) is an open-source Python package for bias detection and debiasing in text. Every flag returns the exact phrase, the bias type, a severity rating, a 1&ndash;2 sentence rationale, and a neutral rewrite &mdash; as a structured, validated object you can drop into any pipeline.
</p>

<div class="up-hero__cta" markdown>
[Try the live demo :material-arrow-right:](https://unbias-plus.vectorinstitute.ai/){ .md-button .md-button--primary }
[GitHub](https://github.com/VectorInstitute/unbias-plus){ .md-button }
</div>

[![PyPI version](https://img.shields.io/pypi/v/unbias-plus.svg)](https://pypi.org/project/unbias-plus/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FlUSeFFXr3VeADiStOIzX7hOV6IlliC8?usp=sharing)
[![License: Vector Institute](https://img.shields.io/badge/License-Vector%20Institute-003049.svg)](https://github.com/VectorInstitute/unbias-plus/blob/unbias-pretrained/LICENSE.md)

</div>

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

## Try it in 30 seconds

=== "pip"

    ```bash
    pip install unbias-plus
    unbias-plus --text "Women are too emotional to lead."
    ```

=== "uv"

    ```bash
    uv sync
    source .venv/bin/activate
    unbias-plus --text "Women are too emotional to lead."
    ```

=== "API + demo UI"

    ```bash
    pip install unbias-plus
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

For setup details, deployment patterns, and advanced usage, see the [User Guide](user_guide.md).

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

---

## Walkthroughs

Tours of the demo UI you get when you run `unbias-plus --serve`:
[:material-volume-off: Silent walkthrough](https://drive.google.com/file/d/1aNh0bqeA2rTZ-uKi_cfrljo_UHP1M4Uq/view?usp=sharing) &middot;
[:material-volume-high: Voiced walkthrough](https://drive.google.com/file/d/1uPiLQ5GZKQH7cBeuV2QQeituxFPC6zTK/view?usp=sharing)

---

## Continue exploring

<div class="grid cards" markdown>

-   [__How it works :material-arrow-right:__](how_it_works.md)

    The pipeline: prompt &rarr; fine-tuned model &rarr; parser &rarr; validated `BiasResult`.

-   [__User Guide :material-arrow-right:__](user_guide.md)

    Install, CLI, REST API, Python API, and development setup.

-   [__API Reference :material-arrow-right:__](api.md)

    Auto-generated reference for every public class and function in `unbias_plus`.

-   [__FAQ :material-arrow-right:__](faq.md)

    Privacy, supported languages, training data, scope, and limitations.

</div>

---

## About

Shaina Raza, PhD, Ahmed Y. Radwan, and Amrit Krishnan &mdash; AI Engineering team at the [Vector Institute](https://vectorinstitute.ai).

The toolkit is released under the [Vector Institute License](https://github.com/VectorInstitute/unbias-plus/blob/unbias-pretrained/LICENSE.md).

For questions, collaboration, or licensing inquiries: [shaina.raza@vectorinstitute.ai](mailto:shaina.raza@vectorinstitute.ai). For bug reports and feature requests: [GitHub Issues](https://github.com/VectorInstitute/unbias-plus/issues).

??? note "Acknowledgements"
    Resources used in preparing this research are provided, in part, by the Province of Ontario, the Government of Canada through CIFAR, and companies sponsoring the Vector Institute. This research is also supported by the European Union's Horizon Europe research and innovation programme under the **AIXPERT** project (Grant Agreement No. 101214389).
