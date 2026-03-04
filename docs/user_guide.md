# User Guide

## Installation

The project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install uv, then from the project root:

```bash
uv sync
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

For development (tests, linting, type checking):

```bash
uv sync --dev
source .venv/bin/activate
```

To skip optional groups (e.g. docs):

```bash
uv sync --no-group docs
```

**Requirements:** Python ≥3.10, &lt;3.12. CUDA 12.4 is recommended for GPU; CPU is supported.

---

## Command line

Analyze a string:

```bash
unbias-plus --text "Women are too emotional to lead."
```

Analyze a file and output JSON:

```bash
unbias-plus --file article.txt --json
```

Start the API server and demo UI (default model `vector-institute/Unbias-plus-Qwen2.5`, port 8000). The demo UI is served at the same host/port:

```bash
unbias-plus --serve
unbias-plus --serve --model path/to/model --port 8000
unbias-plus --serve --load-in-4bit   # reduce VRAM (default for the bundled model)
```

**Options:** `--model`, `--load-in-4bit`, `--max-new-tokens`, `--host`, `--port`, `--json`.

---

## REST API

Start the server with `unbias-plus --serve`. The demo web UI is at `http://localhost:8000/`; use the same host/port for the API:

| Endpoint | Description |
|----------|-------------|
| **GET /health** | Returns `{"status": "ok", "model": "<model_name_or_path>"}`. |
| **POST /analyze** | Body: `{"text": "Your text here"}`. Returns a JSON object matching the `BiasResult` schema. |

Example with `curl`:

```bash
curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d '{"text": "Women are too emotional to lead."}'
```

Programmatic server start:

```python
from unbias_plus.api import serve
serve()  # default model vector-institute/Unbias-plus-Qwen2.5, port 8000
# Or: serve("path/to/model", port=8000, load_in_4bit=False)
```

---

## Python API

```python
from unbias_plus import UnBiasPlus, BiasResult, BiasedSegment

pipe = UnBiasPlus()  # default: vector-institute/Unbias-plus-Qwen2.5 (loads in 4-bit by default)
result = pipe.analyze("Women are too emotional to lead.")

# Result fields
print(result.binary_label)    # "biased" | "unbiased"
print(result.severity)         # 1–5
print(result.bias_found)       # bool
print(result.unbiased_text)   # full neutral rewrite

for seg in result.biased_segments:
    print(seg.original, seg.replacement, seg.severity, seg.bias_type, seg.reasoning)
    # seg.start, seg.end are character offsets in the original text

# Formatted outputs
cli_str = pipe.analyze_to_cli("...")   # human-readable terminal output
d = pipe.analyze_to_dict("...")         # plain dict
json_str = pipe.analyze_to_json("...")  # pretty-printed JSON string
```

---

## Development

- **Tests:** Run from repo root: `uv run pytest tests/`.
- **Linting / formatting:** `ruff` (format + lint), configured in `pyproject.toml`.
- **Type checking:** `mypy` with strict options, `mypy_path = "src"`.

Build documentation locally:

```bash
uv sync --group docs
mkdocs serve
```

Then open `http://127.0.0.1:8000` in your browser (if that port is already in use by another app, run `mkdocs serve -a 127.0.0.1:8001` and use port 8001).
