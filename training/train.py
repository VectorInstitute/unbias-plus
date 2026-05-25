"""Fine-tune Qwen3-8B using SFT with completion-only loss on debiasing data.

Training  : bf16 LoRA on single A100 (or QLoRA 4-bit via --load-in-4bit)
Export    : merged 16-bit model — load as bf16 or 4-bit at deploy time

python train.py --input-path data/Unbias-plus-clean.json --output-dir outputs/run5
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
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
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
DEFAULT_EPOCHS = 3
DEFAULT_LORA_R = 16
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
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=DEFAULT_LORA_R,
        help="LoRA rank. lora_alpha is set to 2x this value (rslora).",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="QLoRA: load the base in 4-bit. Much lighter VRAM, slight quality cost.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log metrics to Weights & Biases (offline-safe; see --wandb-project).",
    )
    parser.add_argument(
        "--wandb-project",
        default="unbias-plus",
        help="W&B project name (used when --wandb / --wandb-log-model is set).",
    )
    parser.add_argument(
        "--wandb-log-model",
        action="store_true",
        help="After export, log the LoRA adapter as a W&B artifact (implies --wandb).",
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

    Accepts both biased and unbiased labeled samples. Verified against the
    Unbias-plus data (4867 rows): biased rows carry a length-preserved neutral
    rewrite, and unbiased rows carry ``unbiased_text`` identical to
    ``article_text`` (no bias -> nothing to neutralize, so the article is echoed
    back unchanged).

    Biased rows MUST have at least one biased segment; unbiased rows have an
    empty ``biased_segments`` list by construction. On the current data these
    length floors and the empty-segment rule drop 0 rows — they are guards
    against a future malformed regeneration, not active filters today.
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


def format_sample(sample: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Convert a raw sample into completion-only chat training format.

    Structure:
      [system]
      [user]      -> article text
      [assistant] -> JSON output

    enable_thinking=False: model goes straight to structured JSON output.
    No synthetic <think> blocks — avoids train/inference mismatch.

    EOS terminator: the assistant turn is included in `messages` so that
    apply_chat_template closes it with <|im_end|>. The previous version
    used add_generation_prompt=True and string-concatenated the JSON,
    leaving the sequence open — the model would never learn to stop.
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
        {
            "role": "assistant",
            "content": json.dumps(completion, ensure_ascii=False, indent=2),
        },
    ]

    # No add_generation_prompt: assistant turn is in messages, template
    # closes it properly with <|im_end|>.
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        enable_thinking=False,
    )

    # Cache the token count here so stats + the length filter don't each
    # re-tokenize the whole dataset again downstream.
    n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return {"text": text, "n_tokens": n_tokens}


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


def filter_by_token_length(dataset: Dataset, max_length: int) -> Dataset:
    """Drop overlength samples using the cached ``n_tokens`` column.

    Reuses the token counts computed in ``format_sample`` instead of
    re-tokenizing the whole dataset, then drops the helper column so the
    SFT collator only sees the ``text`` field.
    """
    before = len(dataset)
    dataset = dataset.filter(
        lambda ex: ex["n_tokens"] <= max_length, desc="Length filter"
    )
    logger.info("  Dropped %d overlength samples", before - len(dataset))
    return dataset.remove_columns("n_tokens")


def print_token_stats(dataset: Dataset, max_seq_length: int) -> None:
    """Log token-length distribution using the cached ``n_tokens`` column."""
    lengths = dataset["n_tokens"]
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


def _prepare_qwen35_tokenizer(tokenizer: Any, model_name: str) -> Any:
    """Qwen3.5-only tokenizer setup — no-op for Qwen3-8B and other models.

    Qwen3.5 Base checkpoints differ from Qwen3-8B in two ways:
      1. Unsloth may return a VLM *processor* instead of a plain tokenizer.
      2. ``-Base`` weights ship without a chat template, but our SFT pipeline
         relies on ``apply_chat_template`` (same as the Qwen3-8B path).

    This helper is called only when the model name looks like Qwen3.5, so the
    existing Qwen3-8B load path is untouched.
    """
    name = model_name.lower()
    if "qwen3.5" not in name and "qwen3_5" not in name:
        return tokenizer

    logger.info("  Qwen3.5 detected — preparing tokenizer for chat-template SFT")

    # Unsloth returns a multimodal processor for Qwen3.5; unwrap to text tokenizer.
    if hasattr(tokenizer, "tokenizer") and tokenizer.tokenizer is not None:
        tokenizer = tokenizer.tokenizer

    # Base checkpoints have no chat template; attach the Qwen3 instruct template.
    if not getattr(tokenizer, "chat_template", None):
        from unsloth.chat_templates import get_chat_template  # noqa: PLC0415

        tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")
        logger.info("  Attached qwen3-instruct chat template (Qwen3.5 Base)")

    return tokenizer


def load_base_model(
    model_name: str, max_seq_length: int, load_in_4bit: bool
) -> tuple[Any, Any]:
    """Load Qwen3 for LoRA SFT.

    load_in_4bit=False : full bf16 base — best quality, heavier VRAM.
    load_in_4bit=True  : QLoRA — base weights quantized to 4-bit (NF4) while
                         the LoRA adapters stay bf16. Much lighter VRAM at a
                         small quality cost; still merges back to 16-bit on
                         export, so the deployed model is unchanged in format.

    Tokenizer padding fix:
      Qwen3 uses its own special tokens — we must NOT override pad_token
      with a hardcoded string like '<EOS_TOKEN>' which doesn't exist in
      Qwen3's vocabulary. Instead we use the tokenizer's actual eos_token
      directly, which is already set correctly by Qwen3's tokenizer config.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=torch.bfloat16,
        use_gradient_checkpointing="unsloth",
        device_map={"": 0},  # single GPU
    )

    # --- Qwen3.5 only (skipped entirely for unsloth/Qwen3-8B etc.) ---
    tokenizer = _prepare_qwen35_tokenizer(tokenizer, model_name)

    # Fix: use tokenizer's own eos_token (Qwen3-specific), not a hardcoded string.
    # If pad_token is already set by the tokenizer config, don't override it.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"
    return model, tokenizer


def configure_lora(model: Any, seed: int, lora_r: int) -> Any:
    """Configure LoRA tuned for Qwen3 / single A100 / ~5K samples.

    r (default 16): sufficient rank for the structured-detection task and
                    cheap to train. If neutral-rewrite quality is the
                    bottleneck, try --lora-r 32 (alpha auto-scales to 64) for
                    extra generation capacity at modest overfit risk on 5K.
    lora_alpha   : fixed at 2xr with rslora.
    lora_dropout=0: set to 0 so Unsloth can apply its fastest kernel
                    patches (dropout > 0 disables fast patching).
                    Regularization handled by weight_decay + small dataset.
    use_rslora   : rank-stabilised LoRA for better gradient stability.
    """
    return FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_r * 2,
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
    epochs: int,
    use_wandb: bool,
) -> SFTConfig:
    """Build the SFT config for Qwen3 / single A100 / bf16.

    batch=4 + grad_accum=4 : effective batch=16.
    epochs (default 3)     : avoids overfit.
    lr=1e-4                : conservative with rslora + r=16.

    At ~5K samples and effective batch 16 you get ~300 steps/epoch, so
    per-epoch eval/save fires cleanly and load_best_model_at_end has real
    checkpoints to choose from. (Per-step schedules were dropped because at
    smaller dataset sizes they barely fire.)
    save_total_limit=2 keeps best + latest, which some TRL versions
    require when load_best_model_at_end is on.

    neftune_noise_alpha=5: injects uniform noise into input embeddings during
    training only (removed automatically at inference). A cheap, well-
    established bump for instruction/rewrite quality — targets weak rewrites
    at zero memory cost.
    """
    run_name = Path(output_dir).name if use_wandb else None
    return SFTConfig(
        output_dir=output_dir,
        # --- Batch / steps ---
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # effective batch = 16
        num_train_epochs=epochs,
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
        logging_first_step=True,
        report_to="wandb" if use_wandb else "none",
        run_name=run_name,
        seed=seed,
        # --- Generation-quality regularizer ---
        neftune_noise_alpha=5,
        # --- Checkpoint: per-epoch, keep best + a couple recent ---
        # save_total_limit=3 (not 2): with load_best_model_at_end the best
        # checkpoint is always retained, leaving room for the latest 2 so a
        # walltime kill always has a clean, recent checkpoint to resume from.
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
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


class EpochReporter(TrainerCallback):
    """Emit clean per-epoch lines to stdout so SLURM .out logs are readable.

    The HF Trainer already logs train loss every ``logging_steps`` (each line
    carries the fractional epoch), and runs eval + writes a checkpoint at each
    epoch boundary. This callback surfaces those events as tidy, greppable
    summary lines in the cluster job log.
    """

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log eval loss at the end of each epoch."""
        if metrics and "eval_loss" in metrics:
            logger.info(
                "  [epoch %.2f] eval_loss = %.4f",
                state.epoch or 0.0,
                metrics["eval_loss"],
            )

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Log checkpoint save events with step and output directory."""
        logger.info(
            "  [epoch %.2f] checkpoint saved -> %s/checkpoint-%d (keep best + latest)",
            state.epoch or 0.0,
            args.output_dir,
            state.global_step,
        )


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

    callbacks = [EpochReporter()]
    try:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=training_args,
            processing_class=tokenizer,  # TRL >= 0.12
            callbacks=callbacks,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=training_args,
            tokenizer=tokenizer,  # TRL < 0.12 fallback
            callbacks=callbacks,
        )

    # Auto-resume: if a checkpoint already exists in output_dir (e.g. a prior
    # run was killed at the walltime ceiling), pick up where it left off.
    # Otherwise start fresh. This makes a walltime kill fully recoverable —
    # just re-submit the same job.
    from transformers.trainer_utils import get_last_checkpoint  # noqa: PLC0415

    last_ckpt = None
    if os.path.isdir(training_args.output_dir):
        last_ckpt = get_last_checkpoint(training_args.output_dir)

    if last_ckpt is not None:
        logger.info("  Resuming from checkpoint: %s", last_ckpt)
    else:
        logger.info("  No checkpoint found — starting from scratch.")

    logger.info("Starting Qwen3 SFT training on single A100...")
    trainer.train(resume_from_checkpoint=last_ckpt)

    # load_best_model_at_end=True -> the in-memory model is now the best
    # (lowest eval_loss) checkpoint, which is what gets saved/exported next.
    best_metric = trainer.state.best_metric
    logger.info(
        "  Training complete. Best eval_loss = %s | best checkpoint = %s",
        f"{best_metric:.4f}" if best_metric is not None else "n/a",
        trainer.state.best_model_checkpoint,
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_model(model: Any, tokenizer: Any, output_dir: Path) -> None:
    """Merge LoRA adapter into base model and save as merged 16-bit.

    Deployment options from merged_16bit:
      load_in_4bit=False  -> full bf16, best quality  (~16GB VRAM)
      load_in_4bit=True   -> 4-bit quantized on load  (~5GB VRAM)
      llama.cpp convert   -> GGUF for Ollama/LM Studio (any laptop)
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
    logger.info("  +- Full quality  (server / high-end machine)")
    logger.info("  |    load_in_4bit=False, dtype=torch.bfloat16")
    logger.info("  +- Lightweight   (commercial laptop, 4-8GB VRAM)")
    logger.info("  |    load_in_4bit=True")
    logger.info("  +- CPU / Ollama  (any laptop)")
    logger.info("       convert merged_16bit -> GGUF q4_k_m with llama.cpp")


def log_adapter_artifact(output_dir: Path, project: str) -> None:
    """Log the trained LoRA adapter as a W&B artifact.

    Logs only the adapter (top-level files: adapter weights/config + tokenizer,
    ~tens of MB) — NOT merged_16bit/ or checkpoint-*/ subdirs. The merged 8B
    model is ~16GB and not worth versioning in W&B; the adapter plus the base
    model ID fully reconstructs it.

    Reuses the active training run if one is still open; otherwise opens a
    short export-only run (works offline — synced later with ``wandb sync``).
    """
    import wandb  # noqa: PLC0415

    created = False
    run = wandb.run
    if run is None:
        run = wandb.init(project=project, job_type="export", resume="allow")
        created = True

    artifact = wandb.Artifact(name="lora-adapter", type="model")
    for f in output_dir.iterdir():
        if f.is_file():
            artifact.add_file(str(f))
    run.log_artifact(artifact)
    logger.info("  Logged LoRA adapter to W&B as artifact 'lora-adapter'")

    if created:
        run.finish()


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

    # --wandb-log-model implies --wandb (need an active run to log the artifact).
    use_wandb = args.wandb or args.wandb_log_model
    if use_wandb:
        # Cluster compute nodes are usually offline — default to offline so the
        # run logs locally and can be synced later with `wandb sync`. Override
        # by exporting WANDB_MODE=online on a node with internet.
        os.environ.setdefault("WANDB_MODE", "offline")
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        logger.info(
            "  W&B enabled | project=%s | mode=%s",
            args.wandb_project,
            os.environ["WANDB_MODE"],
        )

    logger.info("\n[1/8] Loading raw data...")
    raw_data = load_raw_data(args.input_path, args.train_samples)
    logger.info("  Loaded %d samples", len(raw_data))

    logger.info("\n[2/8] Loading base model...")
    model, tokenizer = load_base_model(
        args.base_model, args.max_seq_length, args.load_in_4bit
    )
    logger.info("  Quantization: %s", "4-bit (QLoRA)" if args.load_in_4bit else "bf16")

    logger.info("\n[3/8] Building & filtering dataset...")
    dataset = build_dataset(raw_data, tokenizer)

    logger.info("\n[4/8] Token length statistics...")
    print_token_stats(dataset, args.max_seq_length)

    logger.info("\n[5/8] Filtering overlength samples...")
    dataset = filter_by_token_length(dataset, args.max_seq_length)
    logger.info("  Final dataset size: %d", len(dataset))

    logger.info("\n[6/8] Configuring LoRA...")
    model = configure_lora(model, args.seed, args.lora_r)

    logger.info("\n[7/8] Training...")
    training_args = build_training_args(
        str(args.output_dir),
        args.max_seq_length,
        args.seed,
        args.epochs,
        use_wandb,
    )
    train_model(model, tokenizer, dataset, training_args, args.seed)

    logger.info("\n[8/8] Saving best adapter + exporting merged 16-bit model...")
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    logger.info("  LoRA adapter saved to : %s", args.output_dir)

    export_model(model, tokenizer, args.output_dir)

    if args.wandb_log_model:
        logger.info("\n  Logging LoRA adapter artifact to W&B...")
        log_adapter_artifact(args.output_dir, args.wandb_project)

    logger.info("\nDone.")
    logger.info("  LoRA adapter   : %s", args.output_dir)
    logger.info("  Merged 16-bit  : %s/merged_16bit/", args.output_dir)


if __name__ == "__main__":
    main()
