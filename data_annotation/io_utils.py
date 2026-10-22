"""JSONL reading and atomic writing helpers."""

import json
import os
from typing import Any


def load_jsonl(path: str) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts, skipping blank lines."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl_atomic(rows: list[dict[str, Any]], path: str) -> None:
    """Write ``rows`` to ``path`` as JSONL via a temp file + atomic replace."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    temp_path = f"{path}.tmp.{os.getpid()}"

    with open(temp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    os.replace(temp_path, path)
