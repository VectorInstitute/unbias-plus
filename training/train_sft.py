"""Fine-tune Qwen3-8B using SFT with completion-only loss on debiasing data.

Training  : bf16 full precision on single A100
Export    : merged 16-bit model — load as bf16 or 4-bit at deploy time
"""

# ruff: noqa: E402, I001 — unsloth must be imported before torch/datasets/trl,
# and TORCHDYNAMO_DISABLE must be set before importing unsloth, which forces
# a non-standard import order that ruff's isort + E402 would otherwise reject.
from __future__ import annotations

import os

# Disable torch dynamo before any heavy imports (unsloth/torch read this at import).
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import unsloth  # noqa: F401 — must precede torch / datasets / trl

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, cast

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

from prompts import SFT_SYSTEM_PROMPT as SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_BASE_MODEL = "unsloth/Qwen3-8B"
DEFAULT_MAX_SEQ_LENGTH = 8192
DEFAULT_TRAIN_SAMPLES = 5000
DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the SFT fine-tuning run."""
    parser = argparse.ArgumentParser(
        description="SFT fine-tune Qwen3-8B for bias detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to the training data JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save the LoRA adapter and merged 16-bit model.",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="HuggingFace model ID of the base model.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help="Maximum tokens per sample. Samples exceeding this are dropped.",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=DEFAULT_TRAIN_SAMPLES,
        help="Truncate training data to this many samples (after filtering).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_raw_data(path: Path, limit: int) -> list[dict[str, Any]]:
    """Load and truncate raw JSON training data."""
    with path.open(encoding="utf-8") as f:
        data = cast(list[dict[str, Any]], json.load(f))
    return data[:limit]


def is_valid_sample(sample: dict[str, Any]) -> bool:
    """Filter out malformed or incomplete samples that would corrupt training.

    Accepts both biased and unbiased labeled samples.
    """
    article = sample.get("article_text", "")
    unbiased = sample.get("unbiased_text", "")
    segments = sample.get("biased_segments")

    if not article or len(article) < 200:
        return False
    if not unbiased or len(unbiased) < 100:
        return False
    if segments is None or not isinstance(segments, list):
        return False
    return not (sample.get("binary_label") == "biased" and len(segments) == 0)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def format_sample(sample: dict[str, Any], tokenizer: Any) -> dict[str, str]:
    """Convert a raw sample into completion-only chat training format.

    Structure:
      [system]
      [user]      → article text
      [assistant] → JSON output

    enable_thinking=False: model goes straight to structured JSON output.
    No synthetic <think> blocks — avoids train/inference mismatch.
    Output JSON schema is identical to Qwen2.5 for downstream compatibility.
    """
    completion = {
        "binary_label": sample["binary_label"],
        "severity": sample["severity"],
        "bias_found": sample["bias_found"],
        "biased_segments": sample["biased_segments"],
        "unbiased_text": sample["unbiased_text"],
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Analyze the following article for bias and return the result "
                "in the required JSON format.\n\n"
                f"ARTICLE:\n{sample['article_text']}"
            ),
        },
    ]

    # enable_thinking=False — no <think> tokens injected at train or inference.
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    assistant_turn = json.dumps(completion, ensure_ascii=False, indent=2)

    return {"text": prompt + assistant_turn}


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def build_dataset(raw_data: list[dict[str, Any]], tokenizer: Any) -> Dataset:
    """Filter invalid samples, format, and return a HuggingFace Dataset."""
    valid = [s for s in raw_data if is_valid_sample(s)]
    logger.info("  Valid samples after filtering: %d / %d", len(valid), len(raw_data))

    dataset = Dataset.from_list(valid)
    return dataset.map(
        lambda x: format_sample(x, tokenizer),
        remove_columns=dataset.column_names,
        desc="Formatting samples",
    )


def filter_by_token_length(
    dataset: Dataset,
    tokenizer: Any,
    max_length: int,
) -> Dataset:
    """Drop samples that exceed the model context window."""

    def is_within_length(example: dict[str, str]) -> bool:
        ids = tokenizer(example["text"], add_special_tokens=False)["input_ids"]
        return len(ids) <= max_length

    before = len(dataset)
    dataset = dataset.filter(is_within_length, desc="Length filter")
    logger.info("  Dropped %d overlength samples", before - len(dataset))
    return dataset


def print_token_stats(dataset: Dataset, tokenizer: Any, max_seq_length: int) -> None:
    """Log a quick summary of token-length distribution."""
    lengths = [
        len(tokenizer(dataset[i]["text"], add_special_tokens=False)["input_ids"])
        for i in range(len(dataset))
    ]
    logger.info("  Samples       : %d", len(lengths))
    logger.info("  Max tokens    : %d", max(lengths))
    logger.info("  Avg tokens    : %.0f", sum(lengths) / len(lengths))
    logger.info(
        "  > %d tokens : %d",
        max_seq_length,
        sum(length > max_seq_length for length in lengths),
    )


# ---------------------------------------------------------------------------
# Model & LoRA
# ---------------------------------------------------------------------------


def load_base_model(model_name: str, max_seq_length: int) -> tuple[Any, Any]:
    """Load Qwen3 in full bf16 precision.

    Tokenizer padding fix:
      Qwen3 uses its own special tokens — we must NOT override pad_token
      with a hardcoded string like '<EOS_TOKEN>' which doesn't exist in
      Qwen3's vocabulary. Instead we use the tokenizer's actual eos_token
      directly, which is already set correctly by Qwen3's tokenizer config.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,  # bf16 full precision
        dtype=torch.bfloat16,
        use_gradient_checkpointing="unsloth",
        device_map={"": 0},  # single GPU
    )

    # Fix: use tokenizer's own eos_token (Qwen3-specific), not a hardcoded string.
    # If pad_token is already set by the tokenizer config, don't override it.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"
    return model, tokenizer


def configure_lora(model: Any, seed: int) -> Any:
    """Configure LoRA tuned for Qwen3 / single A100 / ~5K samples.

    r=16          : sufficient rank for structured task, reduces overfit.
    lora_alpha=32 : alpha = 2×r with rslora.
    lora_dropout=0: set to 0 so Unsloth can apply its fastest kernel
                    patches (dropout > 0 disables fast patching).
                    Regularization handled by weight_decay + small dataset.
    use_rslora    : rank-stabilised LoRA for better gradient stability.
    """
    return FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0,  # 0 = Unsloth fast patching enabled
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        use_rslora=True,
        loftq_config=None,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


def build_training_args(
    output_dir: str,
    max_seq_length: int,
    seed: int,
) -> SFTConfig:
    """Build the SFT config for Qwen3 / single A100 / 5K samples / bf16.

    batch=4 + grad_accum=4 : effective batch=16.
    epochs=3               : avoids overfit on 5K samples.
    lr=1e-4                : conservative with rslora + r=16.
    save_total_limit=1     : single clean best adapter on disk.
    load_best_model_at_end : restores best checkpoint by eval loss.
    """
    return SFTConfig(
        output_dir=output_dir,
        # --- Batch / steps ---
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # effective batch = 16
        num_train_epochs=3,
        # --- Learning rate ---
        learning_rate=1e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        # --- Precision ---
        bf16=True,
        fp16=False,
        # --- Optimiser ---
        optim="paged_adamw_8bit",
        # --- Logging ---
        logging_steps=10,
        report_to="none",
        seed=seed,
        # --- Checkpoint: single best adapter by eval loss ---
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        # --- Dataset / sequence ---
        dataset_text_field="text",
        max_length=max_seq_length,
        completion_only_loss=True,
        remove_unused_columns=False,
        dataset_num_proc=2,
        # --- Misc ---
        ddp_find_unused_parameters=False,
        torch_compile=False,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
    training_args: SFTConfig,
    seed: int,
) -> None:
    """Train with TRL SFTTrainer.

    Splits 5% for eval so load_best_model_at_end works correctly.
    """
    split = dataset.train_test_split(test_size=0.05, seed=seed)
    train_ds = split["train"]
    eval_ds = split["test"]
    logger.info("  Train: %d | Eval: %d", len(train_ds), len(eval_ds))

    try:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=training_args,
            processing_class=tokenizer,  # TRL >= 0.12
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=training_args,
            tokenizer=tokenizer,  # TRL < 0.12 fallback
        )

    logger.info("Starting Qwen3 bf16 SFT training on single A100...")
    trainer.train()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_model(model: Any, tokenizer: Any, output_dir: Path) -> None:
    """Merge LoRA adapter into base model and save as merged 16-bit.

    Deployment options from merged_16bit:
      load_in_4bit=False  → full bf16, best quality  (~16GB VRAM)
      load_in_4bit=True   → 4-bit quantized on load  (~5GB VRAM)
      llama.cpp convert   → GGUF for Ollama/LM Studio (any laptop)
    """
    merged_path = output_dir / "merged_16bit"
    logger.info("\n  Merging adapter and saving to: %s", merged_path)

    model.save_pretrained_merged(
        str(merged_path),
        tokenizer,
        save_method="merged_16bit",
    )

    logger.info("  Export complete.")
    logger.info("")
    logger.info("  Deployment options:")
    logger.info("  ┌─ Full quality  (server / high-end machine)")
    logger.info("  │    load_in_4bit=False, dtype=torch.bfloat16")
    logger.info("  ├─ Lightweight   (commercial laptop, 4-8GB VRAM)")
    logger.info("  │    load_in_4bit=True")
    logger.info("  └─ CPU / Ollama  (any laptop)")
    logger.info("       convert merged_16bit → GGUF q4_k_m with llama.cpp")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full SFT fine-tuning pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n[1/8] Loading raw data...")
    raw_data = load_raw_data(args.input_path, args.train_samples)
    logger.info("  Loaded %d samples", len(raw_data))

    logger.info("\n[2/8] Loading base model (bf16, no quantization)...")
    model, tokenizer = load_base_model(args.base_model, args.max_seq_length)

    logger.info("\n[3/8] Building & filtering dataset...")
    dataset = build_dataset(raw_data, tokenizer)

    logger.info("\n[4/8] Token length statistics...")
    print_token_stats(dataset, tokenizer, args.max_seq_length)

    logger.info("\n[5/8] Filtering overlength samples...")
    dataset = filter_by_token_length(dataset, tokenizer, args.max_seq_length)
    logger.info("  Final dataset size: %d", len(dataset))

    logger.info("\n[6/8] Configuring LoRA...")
    model = configure_lora(model, args.seed)

    logger.info("\n[7/8] Training...")
    training_args = build_training_args(
        str(args.output_dir),
        args.max_seq_length,
        args.seed,
    )
    train_model(model, tokenizer, dataset, training_args, args.seed)

    logger.info("\n[8/8] Saving best adapter + exporting merged 16-bit model...")
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    logger.info("  LoRA adapter saved to : %s", args.output_dir)

    export_model(model, tokenizer, args.output_dir)

    logger.info("\nDone.")
    logger.info("  LoRA adapter   : %s", args.output_dir)
    logger.info("  Merged 16-bit  : %s/merged_16bit/", args.output_dir)


if __name__ == "__main__":
    main()
