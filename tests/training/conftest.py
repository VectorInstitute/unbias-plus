"""Test setup for the standalone training scripts.

These scripts ``import unsloth`` at module level and live in ``training/``,
which is not a Python package. CI installs the project without the ``[train]``
extra, so ``unsloth`` is unavailable and ``training/`` is not on ``sys.path``.

This conftest:
  1. Stubs ``unsloth`` in ``sys.modules`` so the training modules import cleanly
     under CI. Tests target only pure-Python helpers, so a no-op stub is enough.
  2. Adds the ``training/`` directory to ``sys.path`` so test files can do
     ``from sanity_check import extract_json``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock


# 1. Stub heavy training-only deps before any test module imports a script.
for _name in ("unsloth", "trl"):
    if _name not in sys.modules:
        sys.modules[_name] = MagicMock()

# 2. Make ``training/`` importable.
_TRAINING_DIR = Path(__file__).resolve().parent.parent.parent / "training"
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))
