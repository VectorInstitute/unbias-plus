# Contributing to UnBias-Plus

Thanks for your interest in contributing to [UnBias-Plus](https://github.com/VectorInstitute/unbias-plus)!

This guide covers how to set up a development environment, the coding standards
we enforce, and how to run the checks and tests locally so your pull request
passes CI. To submit a PR, please fill out the
[pull request template](.github/pull_request_template.md); if the PR fixes an
issue, link it (e.g. `Closes #123`).

- [Development Practices](#development-practices)
- [Development Requirements](#development-requirements)
- [Coding Guidelines, Formatters, and Checks](#coding-guidelines-formatters-and-checks)
- [Code Documentation](#code-documentation)
- [Tests](#tests)
- [Security](#security)
- [Documentation Site](#documentation-site)

## Development Practices

We use the standard git branch-and-merge flow with pull requests on GitHub.
Branch off `main`, push your feature branch, and open a PR targeting `main`.
At least one core-team review is required before merging.

Every PR triggers automated checks that **must pass** before it is eligible to
merge:

| Workflow | What it runs |
| -------- | ------------ |
| `code checks` | `pre-commit run --all-files` (ruff, mypy, typos, doctest, pytest, …) + `pip-audit` |
| `unit tests` | `pytest -m "not integration_test"` with coverage → Codecov |
| `integration tests` | `pytest` suite |
| `docs` | `mkdocs build` (and deploy to GitHub Pages on `main`) |

The [demo/model deployment](.github/workflows/deploy-demo.yml) and
[publish](.github/workflows/publish.yml) workflows run separately on `main`.

## Development Requirements

We use [uv](https://docs.astral.sh/uv/) for dependency management. All
dependencies (runtime, dev, docs, and the optional `train` extra) are declared
in [`pyproject.toml`](pyproject.toml) and pinned in `uv.lock`. The project
targets Python **3.11** (see `.python-version`; `requires-python` is `>=3.10`).

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/VectorInstitute/unbias-plus.git
cd unbias-plus

# Install the project with dev tooling (matches CI)
uv sync --group dev
```

`uv sync` creates a `.venv/` and installs the `dev` and `docs` groups by default.
Optional installs:

```bash
uv sync --group docs          # documentation tooling only
uv sync --extra train         # add training extras (trl, unsloth, flash-attn, wandb)
```

> **Note:** CI installs with `uv sync --group dev --no-extra train`, so the
> `train` extra is intentionally absent there. `mypy` is configured to treat
> the train-only imports (`unsloth`, `trl`, `wandb`) as `Any` whether or not
> they are installed, so type checking behaves identically in CI and local
> clones.

If you change dependencies, edit `pyproject.toml` and refresh the lockfile with
`uv lock`. A `uv lock --check` pre-commit hook fails if the lockfile is stale.

## Coding Guidelines, Formatters, and Checks

For code style we follow [PEP 8](https://peps.python.org/pep-0008/). Formatting
and static analysis are handled by [ruff](https://docs.astral.sh/ruff/), whose
configuration lives under the `[tool.ruff]` sections of `pyproject.toml`
(line length 88, double quotes, and a broad lint rule set including pyflakes,
pycodestyle, bugbear, isort, pydocstyle, and pylint). Type hints are checked
with [mypy](https://mypy.readthedocs.io/en/stable/) using the settings under
`[tool.mypy]`.

We use the modern Python 3.10+ typing style — built-in generics
(`list[str]`, `dict[str, int]`) and `X | None` / `X | Y` instead of
`Optional[...]` / `Union[...]`.

Run the tools directly:

```bash
uv run ruff check .        # lint (add --fix to auto-fix)
uv run ruff format .       # format
uv run mypy src training   # type check
```

### Pre-commit hooks

All checks are wired into [pre-commit](https://pre-commit.com/). Install the
hooks once so they run on every commit:

```bash
uv run pre-commit install
```

Run them against the whole repository at any time:

```bash
uv run pre-commit run --all-files
```

The hook suite (see `.pre-commit-config.yaml`) runs the standard hygiene hooks,
`ruff` (check + format), `mypy`, [`typos`](https://github.com/crate-ci/typos),
`nbqa-ruff` for notebooks, `uv lock --check`, **doctest** over
`src/unbias_plus/`, and the **pytest** suite (`-m "not integration_test"`).
Because `doctest` runs in pre-commit, keep the runnable examples in docstrings
accurate.

## Code Documentation

We use the [numpy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html)
(enforced via ruff's `pydocstyle` with `convention = "numpy"`). Any non-trivial
public function, class, or module added under `src/unbias_plus/` (and
`training/`) should have a docstring. Tests are exempt from the strictest
docstring rules (see the per-file ignores in `pyproject.toml`).

Where practical, include a short runnable example in the docstring — these are
executed by the `doctest` pre-commit hook, so mark non-runnable snippets with
`# doctest: +SKIP`.

## Tests

Tests live in the `tests/` folder and run with
[pytest](https://docs.pytest.org/). Test paths and markers are configured under
`[tool.pytest.ini_options]` in `pyproject.toml`:

```
tests/unbias_plus/   # core package (schema, parser, pipeline, api, prompt, …)
tests/training/      # training scripts
tests/data/          # dataset-building utilities
```

Run the suite the same way CI does:

```bash
# Unit tests with coverage (skips integration tests)
uv run pytest tests/ -m "not integration_test" --cov=src/unbias_plus --cov-report=term-missing

# A single file
uv run pytest tests/unbias_plus/test_pipeline.py

# A single test
uv run pytest tests/unbias_plus/test_parser.py::test_parse_valid_json
```

Two markers are available: `slow` and `integration_test` (deselect with
`-m "not slow"` / `-m "not integration_test"`). New tests that require external
services or GPUs should be marked `integration_test`. Coverage is reported to
[Codecov](https://about.codecov.io/); please add tests for new functionality so
coverage does not regress.

## Security

Dependency vulnerabilities are scanned in CI with
[`pip-audit`](https://pypi.org/project/pip-audit/). You can run it locally:

```bash
uv run pip-audit
```

If a finding is a documented false positive or an accepted risk, it is added to
the `ignore-vulns` list in `.github/workflows/code_checks.yml` with a comment
explaining why. Never commit secrets (API keys, credentials); the
`detect-private-key` pre-commit hook guards against common cases.

## Documentation Site

User docs are built with [MkDocs](https://www.mkdocs.org/) (Material theme).
Preview locally and build:

```bash
uv run mkdocs serve    # live preview at http://127.0.0.1:8000
uv run mkdocs build    # static build into site/
```

The `docs` workflow builds on every PR and deploys to GitHub Pages from `main`.
