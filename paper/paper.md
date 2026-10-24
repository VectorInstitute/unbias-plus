---
title: 'UnBias-Plus: Detect, Explain, and Rewrite Bias'
tags:
  - Python
  - bias detection
  - debiasing
  - natural language processing
  - fairness
  - large language models
authors:
  - name: Ahmed Y. Radwan
    orcid: 0009-0003-6559-863X
    affiliation: 1
  - name: Ahmed Elkady
    orcid: 0009-0002-7960-954X
    affiliation: 1
  - name: Mohamed Hafez
    orcid: 0009-0006-2415-7709
    affiliation: 1
  - name: Amrit Krishnan
    orcid: 0000-0003-1892-4677
    affiliation: 1
  - name: Shaina Raza
    orcid: 0000-0003-1061-5845
    corresponding: true
    affiliation: 1
affiliations:
  - name: Vector Institute for Artificial Intelligence, Toronto, Ontario, Canada
    index: 1
date: 22 October 2026
bibliography: paper.bib
---

# Summary

Biased wording in human-written and artificial intelligence (AI)-generated text shapes how people and groups are represented, often through subtle word choices rather than explicitly harmful content. UnBias-Plus is an open-source Python toolkit that (1) classifies biased segments into eight bias types, (2) localizes each span with character offsets, (3) explains every decision, and (4) rewrites the text neutrally. It provides large language models (LLMs) fine-tuned on a curated, annotated bias dataset [@unbias-plus-dataset], a Python API, a command-line interface (CLI), a REST API, and a browser interface for non-technical users. It also includes a configurable fine-tuning framework that targets a failure mode of standard supervised fine-tuning: rewriting a biased claim by hedging it rather than removing it.

# Statement of Need

Earlier text-bias research framed the problem as separate tasks, such as sentence-level detection, span identification, and neutral rewriting from annotated examples [@gallegos2024bias]. Generative AI widens the problem because outputs are open-ended and context-dependent. Commercial moderation APIs [@openai_moderation_api] and LLM guardrails [@inan2023llama] can flag content against predefined safety policies, but they are designed for broad safety enforcement rather than detailed bias analysis, are often proprietary, and do not explain or mitigate the bias they detect.

Users such as editors, content reviewers, and fairness researchers need a single open tool that identifies what is biased, why, and how to phrase it neutrally, and that they can retrain for their own domain. UnBias-Plus provides this as an end-to-end, reproducible workflow, from inference to retraining.

# State of the Field

General fairness toolkits have emerged in this space, such as AI Fairness 360 (AIF360) [@bellamy2019aif360] which provides a suite for detecting and mitigating disparities across preprocessing, in-processing, and post-processing stages, and Fairlearn [@weerts2023fairlearn] that focuses on fairness constraints for machine learning (ML) models, but both are mainly designed for traditional ML settings For text, Dbias [@raza2022dbias], NBias [@raza2024nbias], and Biaslyze [@biaslyze_api] detect biased expressions, but they rely on encoder-only classifiers and do not generate explanations or rewrites. NeuTral Rewriter [@vanmassenhove2021neutral] rewrites gendered English sentences into gender-neutral alternatives, but addresses a single bias dimension and does not detect or explain bias. Lastly, LangFair [@bouchard2025langfair] and FairLangProc [@manerba2025fairlangproc] use LLM prompting for individual evaluation, without released fine-tuned models. To our knowledge, no existing tool combines granular detection, explanation, rewriting, open trained models, and retraining code. Extending any one of them would have meant replacing its core model, so we built UnBias-Plus as a unified pipeline instead.

# Software Design

\autoref{fig:pipeline} shows the inference pipeline. Given input text, the pipeline builds a prompt, runs the fine-tuned LLM to produce a structured analysis, and parses its JSON output. For each biased segment, the output contains the original text, bias type, severity, explanation, and replacement, together with an overall label, an article-level severity score, and a full neutral rewrite. Character-level offsets are computed deterministically after parsing, by locating each segment in the source text rather than relying on a separate span-tagging model. This keeps each replacement anchored to its source position regardless of tokenization. 

![Pipeline architecture of UnBias-Plus. Input text flows through prompt construction, LLM inference, JSON parsing, offset computation, and result assembly.\label{fig:pipeline}](UnBias.drawio.png)

```python
from unbias_plus import UnBiasPlus

pipe   = UnBiasPlus()
result = pipe.analyze(
    "The aggressive expansion strategy championed by "
    "our male executives has driven record profits."
)

print(result.binary_label)       # "biased"
print(result.severity)           # 3

for seg in result.biased_segments:
    print(seg.original, seg.bias_type, seg.replacement, seg.start, seg.end)
# aggressive expansion strategy  opinion_as_fact
#     -> expansion strategy  (4, 33)
# championed by our male executives  stereotypical_association
#     -> advocated by executives  (34, 67)

print(result.unbiased_text)
# "The expansion strategy advocated by executives has driven record profits."
```

To serve both developers and non-technical users, we expose a Python API, a CLI, and a FastAPI REST service over one shared pipeline. Running `unbias-plus --serve` launches the `/analyze` endpoint together with an interactive web interface, with no additional configuration (\autoref{fig:ui_demo}). The interface highlights each biased segment by severity and displays its reasoning, so reviewers can interpret the output without programming expertise. Results can also be exported as JSON for annotation, evaluation, or dataset construction.

![The UnBias-Plus web interface highlights biased segments by severity, displays per-segment bias types and reasoning, and generates a full neutral rewrite.\label{fig:ui_demo}](UnBias.png)

We release two models fine-tuned with low-rank adaptation (LoRA): the default Qwen3-8B [@unbias-plus-8b] and a compact Qwen3.5-4B [@unbias-plus-4b] for limited hardware. Both support 4-bit quantization for consumer-grade GPUs, and any Hugging Face–compatible checkpoint can be substituted via a custom model path.

## Fine-Tuning Recipes

With plain supervised fine-tuning, a debiasing model can launder bias, retaining a stereotype while hedging it, for example by rewriting “women overthink decisions” as “some people believe women may overthink decisions”, because most of the cross-entropy loss falls on copied tokens and JSON syntax rather than on the rewrite itself.

To address this, the training code provides a weighted trainer configured through YAML recipes, which upweight the `replacement`, `reasoning`, and `rewrite` fields, apply an unlikelihood penalty [@welleck2020unlikelihood] to hedging phrases within rewrites, and curate hard cases. The default model was trained with the bias_weighted recipe, and new recipes can be defined without changing any Python code.

# Research Impact Statement

UnBias-Plus is released on PyPI as `unbias-plus` under the Apache 2.0 license and is maintained on GitHub with continuous integration and versioned releases. For researchers, it provides a unified framework for bias detection, classification, explanation, and rewriting, together with released models, datasets, and training recipes that support reproducible development of fairness-aware language systems.

For practitioners, editorial and content moderation teams can use the toolkit to screen drafts for biased or non-neutral language before publication. In high-stakes domains such as hiring and insurance, it can flag potentially biased wording before documents are released or decisions are communicated.

More broadly, UnBias-Plus can help reduce the spread of biased language in public-facing systems, and in educational settings it can help students identify and revise biased expressions in their own writing.

## Validation

To validate the released models, we evaluated them on an out-of-distribution test set from BABE [@spinde-etal-2021-neural-media], a news media dataset, taking a stratified of 175 biased and 175 unbiased samples. GPT-4o-mini was used as a judge on a 0–5 scale, alongside automatic metrics including recall at biased words, ROUGE-L, length ratio, hallucination rate, and duplicate rate. The prompts for the judge metrics are given in our codebase.

: Biased samples. $\uparrow$ = higher is better, $\approx 1$ = closest to 1.0 is best. Judge scores on a 0–5 scale. \label{tab:biased}

| Model | Bias red. (%) $\uparrow$ | Bias red. (mean) $\uparrow$ | Relevance $\uparrow$ | Global rewrite $\uparrow$ | ROUGE-L $\uparrow$ | Length ratio $\approx 1$ |
|:----------------|-----:|-----:|-----:|-----:|-----:|-----:|
| Qwen3.5-4B UnBias-Plus | **57.5** | **1.91** | **4.09** | **3.15** | 0.71 | **1.00** |
| Qwen3-8B UnBias-Plus | 56.4 | 1.88 | 4.02 | 2.94 | **0.73** | 0.94 |

: Unbiased samples. Judge scores on a 0–5 scale. \label{tab:unbiased}

| Model | Correct ID (mean) $\uparrow$ | Correct ID (median) $\uparrow$ | No needless rewrite (mean) $\uparrow$ | No needless rewrite (median) $\uparrow$ |
|:----------------|-----:|-----:|-----:|-----:|
| Qwen3.5-4B UnBias-Plus | **3.44** | 2.0 | **3.97** | 3.0 |
| Qwen3-8B UnBias-Plus | 3.31 | 2.0 | 3.91 | 3.0 |

: Segment level. $\uparrow$ = higher is better, $\downarrow$ = lower is better. \label{tab:segment}

| Model | Recall at biased words $\uparrow$ | Segment edit quality $\uparrow$ | Hallucination (%) $\downarrow$ | Duplicate (%) $\downarrow$ |
|:----------------|-----:|-----:|-----:|-----:|
| Qwen3.5-4B UnBias-Plus | **0.85** | **3.88** | 5.3 | **0.0** |
| Qwen3-8B UnBias-Plus | 0.84 | 3.77 | **4.5** | 0.2 |

Results in \autoref{tab:biased}, \autoref{tab:unbiased}, and \autoref{tab:segment} show that the two variants perform closely. The compact Qwen3.5-4B UnBias-Plus is marginally stronger on most metrics, including bias reduction, relevance, global rewrite quality, recall at biased words, and segment edit quality, with a zero duplicate rate, while the default Qwen3-8B UnBias-Plus achieves higher lexical overlap with the reference rewrites and a lower hallucination rate. Both models score a median of 2.0 for correct identification of unbiased samples, indicating a tendency to flag unbiased text as biased; this motivates the human-in-the-loop review described in the limitations. These results support the design of UnBias-Plus as a configurable toolkit: users can select the smaller variant for limited hardware without a meaningful loss in quality, or the default 8B model where minimizing hallucinated content is the priority.

# Limitations and Future Development

UnBias-Plus should be used as a bias review assistant rather than as a final authority on neutrality or fairness. Bias is context-dependent: the same phrase may be interpreted differently across domains, communities, and cultural settings. For this reason, the toolkit is designed to support human-in-the-loop review, where model outputs can be inspected, contested, and revised before use in editorial, research, or decision-support workflows.

From a practical perspective, the amount of context that can be processed depends on the selected model and available hardware. While the released 4B and 8B variants make the toolkit easier to deploy, smaller models may still miss subtle framing patterns or produce incomplete reasoning for complex articles. Domain sensitivity is another limitation: adapting the toolkit to a custom domain requires retraining the models, for which we provide the training code.

Future development will focus on expanding domain coverage, improving multilingual support, and extending UnBias-Plus beyond text toward image and multimodal analysis [@raza2025humanibench; @radwan2026sonico1], where bias, framing, and manipulation can appear across visual as well as linguistic content.

# AI Usage Disclosure

ChatGPT was used for grammar and readability improvements in the paper text, and for AI-assisted code suggestions during development. All outputs were reviewed, edited, and validated by the human authors, who made all core design and architectural decisions.

# Acknowledgements

Resources used in preparing this research were provided, in part, by the Province of Ontario, the Government of Canada through CIFAR, and companies sponsoring the Vector Institute (http://www.vectorinstitute.ai/#partners). This research was funded by the European Union's Horizon Europe research and innovation programme under the AIXPERT project (Grant Agreement No. 101214389), which aims to develop an agentic, multi-layered, GenAI-powered framework for creating explainable, accountable, and transparent AI systems.

We thank Orli Namian, Natalie Richard, Marcie De Cesare, Kylie Williams, Kari Clarke-Zemnickis, Sindhuja Chaduvula, Aravind Narayanan, and Kathryn Hume for their valuable support to the UnBias-Plus project.

# References