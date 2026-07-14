"""Test setup for the standalone data pipeline scripts.

The scripts live in ``data/``, which is not a Python package, so add it to
``sys.path`` for ``from build_vldbench_10k import ...``. They also need
``langdetect``, which is intentionally not a project dependency (data prep
runs offline, not in the deployed service) — when it is unavailable, tell
pytest not to collect the test modules at all. A module-level skip is not
an option here: raising ``Skipped`` while initial conftests load aborts
the whole session instead of skipping.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


collect_ignore: list[str] = []
if importlib.util.find_spec("langdetect") is None:
    collect_ignore = ["test_build_vldbench_10k.py"]

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))
