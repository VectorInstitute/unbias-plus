# unbias-plus
[![website](https://img.shields.io/badge/website-ff00ff)](https://vectorinstitute.github.io/unbias-plus/)
[![code checks](https://github.com/VectorInstitute/unbias-plus/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/unbias-plus/actions/workflows/code_checks.yml)
[![unit tests](https://github.com/VectorInstitute/unbias-plus/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/VectorInstitute/unbias-plus/actions/workflows/unit_tests.yml)
[![integration tests](https://github.com/VectorInstitute/unbias-plus/actions/workflows/integration_tests.yml/badge.svg)](https://github.com/VectorInstitute/unbias-plus/actions/workflows/integration_tests.yml)
[![docs](https://github.com/VectorInstitute/unbias-plus/actions/workflows/docs.yml/badge.svg)](https://github.com/VectorInstitute/unbias-plus/actions/workflows/docs.yml)
[![codecov](https://codecov.io/github/VectorInstitute/unbias-plus/graph/badge.svg?token=83MYFZ3UPA)](https://codecov.io/github/VectorInstitute/unbias-plus)
[![License](https://img.shields.io/github/license/VectorInstitute/unbias-plus)](https://github.com/VectorInstitute/unbias-plus/blob/main/LICENSE.md)
[![Contact](https://img.shields.io/badge/Contact-shaina.raza%40vectorinstitute.ai-green)](mailto:shaina.raza@vectorinstitute.ai)

Bias detection and debiasing in text: identify biased segments, classify severity, get reasoning and neutral replacements per segment, and a full neutral rewrite. Structured output (binary label, severity, biased segments with offsets) via CLI, REST API, or Python.

## Overview

Input text → analysis → validated `BiasResult`: binary label (biased/unbiased), overall severity (1–5), `biased_segments` (original phrase, replacement, severity, bias type, reasoning, character offsets), and full `unbiased_text`. Entry points: CLI (`unbias-plus`), REST API (FastAPI + demo UI), or Python (`UnBiasPlus`).

**Project structure:**
```
unbias-plus/
├── src/unbias_plus/
│   ├── __init__.py      # UnBiasPlus, BiasResult, BiasedSegment, serve
│   ├── cli.py           # unbias-plus entry point (--text, --file, --serve)
│   ├── api.py           # FastAPI app, /health, /analyze, serve()
│   ├── pipeline.py      # UnBiasPlus: prompt → model → parse → result
│   ├── model.py         # UnBiasModel: load LM, generate(), 4-bit optional
│   ├── prompt.py        # build_prompt(text), system prompt
│   ├── parser.py        # parse_llm_output() → BiasResult
│   ├── schema.py        # BiasResult, BiasedSegment (Pydantic)
│   ├── formatter.py     # format_cli, format_dict, format_json
│   └── demo/            # bundled web UI (served at / when using --serve)
│       ├── static/      # script.js, style.css
│       └── templates/   # index.html
├── tests/
│   ├── conftest.py      # fixtures (sample_result, sample_json, …)
│   └── unbias_plus/     # test_api, test_pipeline, test_parser, …
├── pyproject.toml
└── README.md
```

## Features

- **Bias detection**: Identifies biased phrases in text and returns them as segments with character-level offsets for highlighting.
- **Classification**: Binary label (biased/unbiased), per-segment severity (low/medium/high), and bias type (e.g. loaded language, framing).
- **Reasoning**: Each segment includes an explanation of why it is considered biased.
- **Debiasing**: Per-segment neutral replacements and a full rewritten `unbiased_text`.
- **Structured output**: Pydantic-validated `BiasResult` with `binary_label`, `severity` (1–5), `biased_segments`, and `unbiased_text`.
- **Demo UI**: `--serve` launches a FastAPI server that also serves a visual web interface at `http://localhost:8000`.
- **CLI**: Analyze from command line with `--text`, `--file`, or start the API + UI with `--serve`. Optional 4-bit quantization and JSON output.
- **REST API**: FastAPI server with `/health` and `/analyze` (POST JSON `{"text": "..."}`). Model loaded at startup via lifespan.
- **Python API**: Use `UnBiasPlus` in code; call `analyze()`, `analyze_to_cli()`, `analyze_to_dict()`, or `analyze_to_json()`.

## Requirements

- Python ≥3.10, <3.12
- CUDA 12.4 recommended (PyTorch + CUDA deps in `pyproject.toml`). CPU is supported with `device="cpu"`.

## Installation

The project uses [uv](https://github.com/astral-sh/uv) for dependency management. Install uv, then from the project root:

```bash
uv sync
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
```

For development (tests, linting, type checking):
```bash
uv sync --dev
source .venv/bin/activate
```

**Optional: flash-attn (GPU only)**
For training or faster inference with flash attention, install the `train` extra (requires CUDA/nvcc to build):
```bash
uv sync --extra train
# On HPC: load CUDA first, e.g. module load cuda/12.4.0
```
Default `uv sync` does **not** install flash-attn, so CI and CPU-only setups work without it.

## Usage

### Command line

```bash
# Analyze a string
unbias-plus --text "Women are too emotional to lead."

# Analyze a file, output JSON
unbias-plus --file article.txt --json

# Start API server + demo UI (default model, port 8000)
unbias-plus --serve
unbias-plus --serve --model path/to/model --port 8000
unbias-plus --serve --load-in-4bit   # reduce VRAM
```

Options: `--model`, `--load-in-4bit`, `--max-new-tokens`, `--host`, `--port`, `--json`.

### Test the model (CLI)

After `uv sync` (and optionally `uv sync --extra train` on a GPU machine), verify the pipeline with:

```bash
# Default install (no flash-attn); use a small model or --load-in-4bit on GPU
uv run unbias-plus --text "Women are too emotional to lead."

# With your own model path
uv run unbias-plus --text "Some biased sentence." --model path/to/your/model

# JSON output
uv run unbias-plus --text "Test." --json
```

Or in Python (same env):

```bash
uv run python -c "
from unbias_plus import UnBiasPlus
pipe = UnBiasPlus()  # or UnBiasPlus('your-model-id', load_in_4bit=True)
text = 'Women are too emotional to lead.'
print(pipe.analyze_to_cli(text))
"
```

### REST API + Demo UI

Start the server with `unbias-plus --serve` (or `serve()` in Python). This starts a single FastAPI server that:

- Serves the visual demo UI at **`http://localhost:8000/`**
- Exposes **`GET /health`** → `{"status": "ok", "model": "<model_name_or_path>"}`
- Exposes **`POST /analyze`** → Body: `{"text": "Your text here"}`. Returns JSON matching `BiasResult`.

Programmatic start:
```python
from unbias_plus import serve
serve("your-hf-model-id", port=8000, load_in_4bit=False)
```

> **Running on a remote server or HPC node:** If the server is running on a remote machine, use SSH port forwarding to access the UI in your browser:
> ```bash
> ssh -L 8000:localhost:8000 user@your-server.com
> # or through a login node to a compute node:
> ssh -L 8000:gpu-node-hostname:8000 user@login-node.com
> ```
> Then open `http://localhost:8000`. If port 8000 is already in use locally, use a different local port (e.g. `-L 8001:...`) and open `http://localhost:8001`.
>
> If you're using VS Code remote SSH, port forwarding is handled automatically via the **Ports** tab.

### Python API

```python
from unbias_plus import UnBiasPlus, BiasResult, BiasedSegment

pipe = UnBiasPlus("your-hf-model-id", load_in_4bit=False)
result = pipe.analyze("Women are too emotional to lead.")

print(result.binary_label)   # "biased" | "unbiased"
print(result.severity)       # 1–5
print(result.bias_found)     # bool

for seg in result.biased_segments:
    print(seg.original, seg.replacement, seg.severity, seg.bias_type, seg.reasoning)
    print(seg.start, seg.end)  # character offsets in original text

print(result.unbiased_text)  # full neutral rewrite

# Formatted outputs
cli_str  = pipe.analyze_to_cli("...")    # human-readable colored terminal output
d        = pipe.analyze_to_dict("...")   # plain dict
json_str = pipe.analyze_to_json("...")   # pretty-printed JSON string
```

## Development

- **Tests**: `pytest` (see `pyproject.toml` for markers). Run from repo root: `uv run pytest tests/`.
- **Linting / formatting**: `ruff` (format + lint), config in `pyproject.toml`.
- **Type checking**: `mypy` with strict options, `mypy_path = "src"`.


## 👥 Team

Developed by the **AI Engineering** team at the [Vector Institute](https://vectorinstitute.ai).
Ahmed Y. Radwan , Sindhuja Chaduvula, Shaina Raza, PhD


## Acknowledgement

Resources used in preparing this research are provided, in part, by the Province of Ontario, the Government of Canada through CIFAR, and companies sponsoring the Vector Institute.

This research is also supported by the European Union's Horizon Europe research and innovation programme under the AIXPERT project (Grant Agreement No. 101214389).



## License

Licensed under the **Apache License 2.0**. See [LICENSE](https://github.com/VectorInstitute/unbias-plus/blob/main/LICENSE.md) in the repository.

## Support

* Open an issue on GitHub: [https://github.com/VectorInstitute/unbias-plus/issues](https://github.com/VectorInstitute/unbias-plus/issues)
