"""Test setup for the standalone data pipeline scripts.

The scripts live in ``data/``, which is not a Python package, so add it to
``sys.path`` for ``from build_vldbench_10k import ...``. They also need
``langdetect``, which is intentionally not a project dependency (data prep
runs offline, not in the deployed service) — skip the whole directory when
it is unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


pytest.importorskip("langdetect", reason="data pipeline scripts need langdetect")

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))
