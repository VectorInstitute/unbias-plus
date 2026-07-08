"""JSONL reading and atomic writing helpers."""

import json
import os


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl_atomic(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    temp_path = f"{path}.tmp.{os.getpid()}"

    with open(temp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    os.replace(temp_path, path)
