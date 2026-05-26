# User Guide

This guide covers installation, the three usage modes (CLI, REST API, Python), and the development workflow. For the underlying pipeline and model behavior, see [How it works](how_it_works.md). For privacy and scope questions, see the [FAQ](faq.md).

---

## Installation

The simplest way to install UnBias-Plus is from PyPI:

```bash
pip install unbias-plus
```

For development, or to use [uv](https://docs.astral.sh/uv/) for dependency management, clone the repository and run from the project root:

```bash
uv sync
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

For development extras (tests, linting, type checking):

```bash
uv sync --dev
source .venv/bin/activate
```

To skip optional groups, for example documentation:

```bash
uv sync --no-group docs
```

For training or faster GPU inference with flash attention, install the `train` extra (requires CUDA and `nvcc` to build):

```bash
uv sync --extra train
# On HPC: load CUDA first, e.g. module load cuda/12.4.0
```

The default install does not include `flash-attn`, so CI and CPU-only setups work without it.

**Requirements:** Python `>= 3.10, < 3.12`. CUDA 12.4 is recommended for GPU; CPU is supported but slow.

!!! note "Project status"

    UnBias-Plus is at **alpha** status (PyPI 0.1.5, March 2026). The API is stable but may evolve as the model and pipeline are refined. Pin to a specific version for production use.

---

## Command line

Analyze a string:

```bash
unbias-plus --text "Women are too emotional to lead."
```

Analyze a file and emit JSON:

```bash
unbias-plus --file article.txt --json
```

Start the API server and demo UI. The default model is `vector-institute/Qwen3-8B-UnBias-Plus-SFT-Instruct` and the default port is `8000`. The demo UI is served at the same host and port:

```bash
unbias-plus --serve
unbias-plus --serve --model path/to/model --port 8000
unbias-plus --serve --load-in-4bit   # reduce VRAM (optional)
```

**Available options:** `--model`, `--load-in-4bit`, `--max-new-tokens`, `--host`, `--port`, `--json`.

---

## REST API

Start the server with `unbias-plus --serve`. The demo web UI is at `http://localhost:8000/`; the same host and port serve the API.

| Endpoint | Description |
|----------|-------------|
| **GET /health** | Returns `{"status": "ok", "model": "<model_name_or_path>"}`. |
| **POST /analyze** | Body: `{"text": "Your text here"}`. Returns a JSON object matching the `BiasResult` schema. |

Example with `curl`:

```bash
curl -X POST http://localhost:8000/analyze \
     -H "Content-Type: application/json" \
     -d '{"text": "Women are too emotional to lead."}'
```

Programmatic server start:

```python
from unbias_plus.api import serve

serve()  # default model, port 8000
# Or: serve("path/to/model", port=8000, load_in_4bit=False)
```

---

## Python API

```python
from unbias_plus import UnBiasPlus, BiasResult, BiasedSegment

pipe = UnBiasPlus()  # default: vector-institute/Qwen3-8B-UnBias-Plus-SFT-Instruct
result = pipe.analyze("Women are too emotional to lead.")

# Result fields
print(result.binary_label)    # "biased" | "unbiased"
print(result.severity)         # 1-5 (article-level)
print(result.bias_found)       # bool
print(result.unbiased_text)   # full neutral rewrite

for seg in result.biased_segments:
    print(seg.original, seg.replacement, seg.severity, seg.bias_type, seg.reasoning)
    # seg.start, seg.end are character offsets in the original text

# Formatted outputs
cli_str  = pipe.analyze_to_cli("...")    # human-readable terminal output
data     = pipe.analyze_to_dict("...")   # plain dict
json_str = pipe.analyze_to_json("...")   # pretty-printed JSON string
```

For the full `BiasResult` schema, see [How it works](how_it_works.md#the-structured-output-is-the-point) or the [API Reference](api.md).

---

## Development

- **Tests**: run from the repository root with `uv run pytest tests/`.
- **Linting and formatting**: `ruff` (format and lint), configured in `pyproject.toml`.
- **Type checking**: `mypy` with strict options, `mypy_path = "src"`.

Build documentation locally:

```bash
uv sync --group docs
mkdocs serve
```

Then open `http://127.0.0.1:8000` in your browser. If that port is already in use, run `mkdocs serve -a 127.0.0.1:8001` and use port `8001` instead.
