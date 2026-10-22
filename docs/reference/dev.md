# Development

- **Tests**: run from the repository root with `uv run pytest tests/`.
- **Linting and formatting**: `ruff` (format and lint), configured in `pyproject.toml`.
- **Type checking**: `mypy` with strict options, `mypy_path = "src"`.

Build documentation locally:

```bash
uv sync --group docs
mkdocs serve
```

Then open `http://127.0.0.1:8000` in your browser. If that port is already in use, run `mkdocs serve -a 127.0.0.1:8001` and use port `8001` instead.
