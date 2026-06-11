# Installation

This page covers installation. For the underlying pipeline and model behavior, see [How it works](../how_it_works.md). For privacy and scope questions, see the [FAQ](../faq.md).

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
