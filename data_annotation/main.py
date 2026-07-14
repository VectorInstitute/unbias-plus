"""
Bias annotation pipeline with checkpointing and resume.

Runs are cumulative: results are written to a single growing output file
(annotated_final.jsonl). On each run, indices that were already annotated
successfully are skipped, previously failed rows are re-processed, and progress
is checkpointed atomically so an interrupted run can resume cleanly.

Usage:
    python main.py --num-samples 100     # first 100
    python main.py --num-samples 1000    # fills 100..999 + any failures
    python main.py                        # rest (full input)
"""

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from annotator import annotate_one
from client import get_client
from config import (
    DEFAULT_CALL_SLEEP,
    DEFAULT_CHECKPOINT_EVERY,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    MAX_WORKERS,
)
from io_utils import load_jsonl, write_jsonl_atomic


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("annotate")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the annotation pipeline."""
    parser = argparse.ArgumentParser(
        description="Bias annotation pipeline with checkpointing and resume."
    )

    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_CALL_SLEEP,
        help="seconds to sleep before each API call (with jitter)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help="write an atomic checkpoint after this many completions",
    )

    return parser.parse_args()


def main() -> None:
    """Run the annotation pipeline: resume, annotate, checkpoint, summarize."""
    args = parse_args()
    client = get_client()

    rows = load_jsonl(args.input)

    target_count = args.num_samples if args.num_samples is not None else len(rows)
    target_count = min(target_count, len(rows))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    records_by_index: dict[int, dict[str, Any]] = {}
    if os.path.exists(args.output):
        for record in load_jsonl(args.output):
            records_by_index[record["index"]] = record

    done = {
        idx
        for idx, record in records_by_index.items()
        if "annotation_error" not in record
    }

    todo = [(idx, rows[idx]) for idx in range(target_count) if idx not in done]

    log.info(
        f"Input rows: {len(rows)} | target: {target_count} | "
        f"already done: {len(done & set(range(target_count)))} | "
        f"to process: {len(todo)} -> {args.output}"
    )

    if not todo:
        log.info("Nothing to process. All target indices already annotated.")
        write_jsonl_atomic(
            [records_by_index[idx] for idx in sorted(records_by_index)],
            args.output,
        )
        return

    completed = 0
    failed = 0
    start_time = time.monotonic()

    def checkpoint() -> None:
        write_jsonl_atomic(
            [records_by_index[idx] for idx in sorted(records_by_index)],
            args.output,
        )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(annotate_one, client, idx, row, args.sleep)
            for idx, row in todo
        ]

        for future in as_completed(futures):
            result = future.result()
            records_by_index[result["index"]] = result

            completed += 1
            failed += "annotation_error" in result

            if completed % args.checkpoint_every == 0:
                checkpoint()
                log.info(f"checkpoint written at {completed}/{len(todo)}")

            elapsed = time.monotonic() - start_time
            log.info(
                f"progress: {completed}/{len(todo)} "
                f"| failed={failed} | elapsed={elapsed:.0f}s"
            )

    checkpoint()

    total_failed = sum(
        "annotation_error" in records_by_index[idx]
        for idx in range(target_count)
        if idx in records_by_index
    )

    log.info(
        f"Done. Processed {completed} rows this run ({failed} failed). "
        f"Total in target range still failing: {total_failed}. "
        f"Saved {len(records_by_index)} rows to {args.output}"
    )


if __name__ == "__main__":
    main()
