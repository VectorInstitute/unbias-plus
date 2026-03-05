# What is unbias-plus?

**unbias-plus** is an AI-driven toolkit for bias detection and debiasing in text. It identifies biased segments, classifies severity, suggests neutral replacements with reasoning, and produces a full neutral rewrite—supporting risk identification, mitigation, and more trustworthy text systems.

[![PyPI version](https://img.shields.io/pypi/v/unbias-plus.svg)](https://pypi.org/project/unbias-plus/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FlUSeFFXr3VeADiStOIzX7hOV6IlliC8?usp=sharing)
[![GitHub](https://img.shields.io/badge/GitHub-repo-blue?logo=github)](https://github.com/VectorInstitute/unbias-plus) — *Setup, usage, CLI, and API.*

---

## Highlights

- **Bias detection** — Pinpoints biased phrases in text and returns them as structured segments with character-level offsets for highlighting.
- **Classification** — Overall binary label (biased/unbiased), per-segment severity (low/medium/high), and bias type (e.g. loaded language, framing).
- **Reasoning** — Each segment includes an explanation of why it is considered biased, so you can review and act with context.
- **Debiasing** — Per-segment neutral replacements plus a full rewritten `unbiased_text` of the input.
- **Structured output** — Pydantic-validated `BiasResult`: `binary_label`, `severity` (1–5), `biased_segments`, and `unbiased_text`. Use via **CLI** (`unbias-plus --text "..."`), **REST API** (FastAPI + demo UI), or **Python API** (`UnBiasPlus().analyze()`).
- **Production-ready** — Optional 4-bit quantization and default fine-tuned model [vector-institute/Unbias-plus-Qwen2.5](https://huggingface.co/vector-institute/Unbias-plus-Qwen2.5).

---

## Video tutorials

Videos show how to use the **demo UI** when you run the API with `unbias-plus --serve`:

- [Demo UI (silent)](https://drive.google.com/file/d/1aNh0bqeA2rTZ-uKi_cfrljo_UHP1M4Uq/view?usp=sharing)
- [Demo UI (voiced)](https://drive.google.com/file/d/1uPiLQ5GZKQH7cBeuV2QQeituxFPC6zTK/view?usp=sharing)

---

## Quick start

```bash
uv sync
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
unbias-plus --text "Women are too emotional to lead."
```

Or in Python:

```python
from unbias_plus import UnBiasPlus
pipe = UnBiasPlus()  # uses default model, or pass model_name_or_path=
result = pipe.analyze("Women are too emotional to lead.")
print(result.binary_label, result.unbiased_text)
```

---

## Contents

- [User Guide](user_guide.md) — Installation, CLI, REST API, Python API, and development.
- [API Reference](api.md) — Module and class reference for `unbias_plus`.
- [Team](team.md) — Contributors, acknowledgement, and support.
- [License](license.md) — Apache License 2.0.
