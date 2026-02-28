# unbias-plus

Bias detection and debiasing using a single LLM. Analyze text for biased language, get structured results (binary label, severity, biased segments with replacements and reasoning), and a full neutral rewrite—all via one fine-tuned causal language model.

## What it does

- **Single-model pipeline**: One HuggingFace causal LM handles both detection and debiasing (no separate classifier + generator).
- **Structured output**: Pydantic-validated `BiasResult` with `binary_label` (biased/unbiased), overall `severity` (1–5), `biased_segments` (original phrase, replacement, severity, bias type, reasoning), and `unbiased_text`.
- **Multiple entry points**: CLI (`unbias-plus`), REST API (FastAPI), or Python API (`UnBiasPlus`).

## Quick start

```bash
uv sync
source .venv/bin/activate
unbias-plus --text "Women are too emotional to lead."
```

Or in Python:

```python
from unbias_plus import UnBiasPlus
pipe = UnBiasPlus("meta-llama/Llama-3.2-3B")
result = pipe.analyze("Women are too emotional to lead.")
print(result.binary_label, result.unbiased_text)
```

## Documentation

- [User Guide](user_guide.md) — Installation, CLI, REST API, Python API, and development.
- [API Reference](api.md) — Module and class reference for `unbias_plus`.
