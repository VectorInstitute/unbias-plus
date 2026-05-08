"""GRPO fine-tuning for bias detection with Unsloth, TRL, torchrun/DDP, and 4-bit QLoRA.

Key behaviors preserved:
  1) Patch Unsloth RL before importing TRL's GRPOTrainer
  2) Load 4-bit model on the correct local rank under DDP
  3) DDP passthrough properties on DistributedDataParallel (no cache edits)
  4) Raw ``article_text`` kept in the dataset for reward checks

Launch (example):
  torchrun --standalone --nproc_per_node=4 training/train_grpo.py \\
    --input-path data.json --output-dir ./out_grpo
"""

# ruff: noqa: E402, I001 — PatchFastRL must run before TRL GRPOTrainer import;
# import order differs from ruff isort expectations.
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, cast

# Optional NCCL workarounds for problematic clusters / networking.
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

# DDP env helpers — do not call dist.get_rank() before process group init.
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
IS_DDP = WORLD_SIZE > 1
IS_MAIN = RANK == 0

import torch
from datasets import Dataset  # type: ignore[import-untyped]
from torch.nn.parallel import DistributedDataParallel

if torch.cuda.is_available():
    torch.cuda.set_device(LOCAL_RANK)

from unsloth import (  # type: ignore[import-not-found]
    FastLanguageModel,
    PatchFastRL,
    is_bfloat16_supported,
)

PatchFastRL("GRPO", FastLanguageModel)

from trl import GRPOConfig, GRPOTrainer  # type: ignore[import-not-found]

from prompts import GRPO_SYSTEM_PROMPT as SYSTEM_PROMPT  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"
DEFAULT_MAX_SEQ_LENGTH = 8192
DEFAULT_MAX_PROMPT_LENGTH = 3072
DEFAULT_MAX_COMPLETION_LENGTH = 1024
DEFAULT_LORA_RANK = 16
DEFAULT_SEED = 3407

REQUIRED_KEYS = frozenset(
    {
        "binary_label",
        "bias_found",
        "severity",
        "biased_segments",
        "unbiased_text",
    }
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the GRPO fine-tuning run."""
    parser = argparse.ArgumentParser(
        description="GRPO fine-tune Qwen3-8B for bias detection (Unsloth + TRL).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to the training data JSON file (list of samples).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save the LoRA adapter and tokenizer.",
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
        help="Maximum sequence length passed to FastLanguageModel.",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=DEFAULT_MAX_PROMPT_LENGTH,
        help="Drop samples whose prompt token length exceeds this.",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=DEFAULT_MAX_COMPLETION_LENGTH,
        help="Maximum new tokens per GRPO completion.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=DEFAULT_LORA_RANK,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load base model in 4-bit (recommended for multi-GPU QLoRA).",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="If set, truncate training data to this many samples after load.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=2,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=2,
        help="GRPO num_generations (effective batch must be divisible by this).",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="Maximum training steps (-1 to use epochs only if supported).",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=50,
        help="Checkpoint save frequency in steps.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# DDP patch
# ---------------------------------------------------------------------------


def patch_ddp_passthrough() -> None:
    """Monkeypatch DDP passthrough properties (config, generation_config, adapters)."""
    try:
        ddp_cls = DistributedDataParallel

        if not hasattr(ddp_cls, "config"):
            ddp_cls.config = property(lambda self: self.module.config)  # type: ignore[attr-defined]

        if not hasattr(ddp_cls, "generation_config"):
            ddp_cls.generation_config = property(  # type: ignore[attr-defined]
                lambda self: self.module.generation_config
            )

        if not hasattr(ddp_cls, "active_adapters"):
            ddp_cls.active_adapters = property(  # type: ignore[attr-defined]
                lambda self: getattr(self.module, "active_adapters", None)
            )

        if IS_MAIN:
            logger.info("[patch] DDP passthrough properties installed")
    except Exception as exc:
        if IS_MAIN:
            logger.warning(
                "[patch] Failed to install DDP passthrough patch: %s",
                exc,
            )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_raw_data(path: Path, limit: int | None) -> list[dict[str, Any]]:
    """Load JSON training data (list of dicts); optionally truncate."""
    with path.open(encoding="utf-8") as f:
        data = cast(list[dict[str, Any]], json.load(f))
    if limit is not None:
        return data[:limit]
    return data


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Apply chat template; tolerate tokenizers without enable_thinking."""
    try:
        return cast(
            str,
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ),
        )
    except TypeError:
        return cast(
            str,
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            ),
        )


def format_sample(sample: dict[str, Any], tokenizer: Any) -> dict[str, str]:
    """Build prompt, ground-truth JSON string, and raw article for rewards."""
    article_text = sample["article_text"]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Analyze the following article:\n\n" + article_text,
        },
    ]
    prompt = apply_chat_template(tokenizer, messages)
    ground_truth = json.dumps(
        {
            "binary_label": sample["binary_label"],
            "severity": sample["severity"],
            "bias_found": sample["bias_found"],
            "biased_segments": sample["biased_segments"],
            "unbiased_text": sample["unbiased_text"],
        },
        ensure_ascii=False,
    )
    return {
        "prompt": prompt,
        "ground_truth": ground_truth,
        "article_text": article_text,
    }


def build_dataset(
    raw_data: list[dict[str, Any]],
    tokenizer: Any,
    max_prompt_length: int,
) -> Dataset:
    """Build HF dataset: prompt, ground-truth JSON, article; drop long prompts."""
    dataset = Dataset.from_list(raw_data)

    def _fmt(row: dict[str, Any]) -> dict[str, str]:
        return format_sample(row, tokenizer)

    dataset = dataset.map(
        _fmt,
        remove_columns=dataset.column_names,
        desc="Formatting samples",
    )

    def prompt_fits(example: dict[str, str]) -> bool:
        ids = tokenizer(
            example["prompt"],
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        return len(ids) <= max_prompt_length

    return dataset.filter(prompt_fits, desc="Prompt length filter")


# ---------------------------------------------------------------------------
# Reward functions (TRL passes dataset columns as kwargs)
# ---------------------------------------------------------------------------


def _safe_json_load(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    try:
        return cast(dict[str, Any], json.loads(text))
    except Exception:
        return None


def reward_json_valid(completions: list[str], **kwargs: Any) -> list[float]:
    """Reward valid JSON and presence of required top-level keys."""
    rewards = []
    for c in completions:
        parsed = _safe_json_load(c)
        if parsed is None:
            rewards.append(0.0)
        else:
            rewards.append(1.0 if REQUIRED_KEYS.issubset(parsed.keys()) else 0.2)
    return rewards


def reward_binary_label(
    completions: list[str],
    ground_truth: list[str],
    **kwargs: Any,
) -> list[float]:
    """Reward matching binary_label vs ground truth."""
    rewards = []
    for c, gt in zip(completions, ground_truth):
        pred = _safe_json_load(c)
        true = _safe_json_load(gt)
        if pred is None or true is None:
            rewards.append(0.0)
            continue
        rewards.append(
            1.0 if pred.get("binary_label") == true.get("binary_label") else 0.0
        )
    return rewards


def reward_severity(
    completions: list[str],
    ground_truth: list[str],
    **kwargs: Any,
) -> list[float]:
    """Reward closeness of numeric severity to ground truth."""
    rewards = []
    for c, gt in zip(completions, ground_truth):
        pred = _safe_json_load(c)
        true = _safe_json_load(gt)
        if pred is None or true is None:
            rewards.append(0.0)
            continue
        try:
            diff = abs(int(pred["severity"]) - int(true["severity"]))
            rewards.append(1.0 if diff == 0 else (0.5 if diff == 1 else 0.0))
        except Exception:
            rewards.append(0.0)
    return rewards


def reward_segment_f1(
    completions: list[str],
    ground_truth: list[str],
    **kwargs: Any,
) -> list[float]:
    """Reward F1 over ``original`` strings in biased_segments."""
    rewards = []
    for c, gt in zip(completions, ground_truth):
        pred = _safe_json_load(c)
        true = _safe_json_load(gt)
        if pred is None or true is None:
            rewards.append(0.0)
            continue
        try:
            pred_set = {
                s["original"]
                for s in pred.get("biased_segments", [])
                if isinstance(s, dict) and isinstance(s.get("original"), str)
            }
            true_set = {
                s["original"]
                for s in true.get("biased_segments", [])
                if isinstance(s, dict) and isinstance(s.get("original"), str)
            }
            if not true_set and not pred_set:
                rewards.append(1.0)
                continue
            if not true_set or not pred_set:
                rewards.append(0.0)
                continue
            inter = pred_set & true_set
            precision = len(inter) / max(len(pred_set), 1)
            recall = len(inter) / max(len(true_set), 1)
            f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
            rewards.append(float(f1))
        except Exception:
            rewards.append(0.0)
    return rewards


def reward_originals_in_article(
    completions: list[str],
    article_text: list[str],
    **kwargs: Any,
) -> list[float]:
    """Reward each segment original being a substring of the source article."""
    rewards = []
    for c, article in zip(completions, article_text):
        pred = _safe_json_load(c)
        if pred is None:
            rewards.append(0.0)
            continue
        segs = pred.get("biased_segments", [])
        if not segs:
            rewards.append(1.0)
            continue
        valid = 0
        total = 0
        for s in segs:
            if not isinstance(s, dict):
                continue
            original = s.get("original")
            if not isinstance(original, str):
                continue
            total += 1
            if original in article:
                valid += 1
        rewards.append(1.0 if total == 0 else valid / total)
    return rewards


def reward_no_duplicates(completions: list[str], **kwargs: Any) -> list[float]:
    """Reward non-duplicate ``original`` values within biased_segments."""
    rewards = []
    for c in completions:
        pred = _safe_json_load(c)
        if pred is None:
            rewards.append(0.0)
            continue
        try:
            segs = [
                s["original"]
                for s in pred.get("biased_segments", [])
                if isinstance(s, dict) and isinstance(s.get("original"), str)
            ]
            rewards.append(1.0 if len(segs) == len(set(segs)) else 0.3)
        except Exception:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# Model & LoRA
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool,
    local_rank: int,
    is_ddp: bool,
) -> tuple[Any, Any]:
    """Load base model; under DDP + 4-bit, pin to this rank's GPU."""
    device_map = {"": local_rank} if (is_ddp and load_in_4bit) else None
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        device_map=device_map,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def configure_lora(model: Any, lora_rank: int, seed: int) -> Any:
    """Attach QLoRA adapters for GRPO."""
    return FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        random_state=seed,
    )


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


def build_grpo_config(
    output_dir: str,
    max_prompt_length: int,
    max_completion_length: int,
    seed: int,
    learning_rate: float,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_generations: int,
    num_train_epochs: int,
    max_steps: int,
    save_steps: int,
) -> GRPOConfig:
    """Build GRPOConfig (effective batch must be divisible by num_generations)."""
    return GRPOConfig(
        learning_rate=learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        temperature=0.7,
        top_p=0.9,
        loss_type="dr_grpo",
        scale_rewards="group",
        mask_truncated_completions=False,
        beta=0.0,
        reward_weights=[0.5, 1.5, 1.0, 2.0, 1.5, 0.5],
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        save_steps=save_steps,
        save_total_limit=2,
        max_grad_norm=0.1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        dataloader_drop_last=True,
        dataloader_num_workers=2,
        logging_steps=1,
        report_to="none",
        output_dir=output_dir,
        seed=seed,
        remove_unused_columns=False,
        use_vllm=False,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_grpo(
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
    training_args: GRPOConfig,
) -> None:
    """Run GRPOTrainer.train() with TRL version fallbacks for tokenizer arg name."""
    reward_funcs = [
        reward_json_valid,
        reward_binary_label,
        reward_severity,
        reward_segment_f1,
        reward_originals_in_article,
        reward_no_duplicates,
    ]
    try:
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset,
        )
    except TypeError:
        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset,
        )

    if IS_MAIN:
        logger.info("Starting GRPO training...")
        logger.info(
            "Samples: %d | max_prompt: %d | max_completion: %d | world_size: %d",
            len(dataset),
            training_args.max_prompt_length,
            training_args.max_completion_length,
            WORLD_SIZE,
        )

    trainer.train()


def save_artifacts(model: Any, tokenizer: Any, output_dir: Path) -> None:
    """Save LoRA adapter and tokenizer on the main process only."""
    if not IS_MAIN:
        return
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Saved LoRA adapter + tokenizer to %s", output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full GRPO fine-tuning pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    args = parse_args()
    patch_ddp_passthrough()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n[1/6] Loading raw data...")
    raw_data = load_raw_data(args.input_path, args.train_samples)
    if IS_MAIN:
        logger.info("  Loaded %d samples", len(raw_data))

    logger.info("\n[2/6] Loading base model...")
    model, tokenizer = load_model_and_tokenizer(
        args.base_model,
        args.max_seq_length,
        args.load_in_4bit,
        LOCAL_RANK,
        IS_DDP,
    )

    logger.info("\n[3/6] Building dataset...")
    dataset = build_dataset(raw_data, tokenizer, args.max_prompt_length)
    if IS_MAIN:
        logger.info("  Dataset size after filter: %d", len(dataset))

    logger.info("\n[4/6] Configuring LoRA...")
    model = configure_lora(model, args.lora_rank, args.seed)

    logger.info("\n[5/6] Training...")
    training_args = build_grpo_config(
        str(args.output_dir),
        args.max_prompt_length,
        args.max_completion_length,
        args.seed,
        args.learning_rate,
        args.per_device_train_batch_size,
        args.gradient_accumulation_steps,
        args.num_generations,
        args.num_train_epochs,
        args.max_steps,
        args.save_steps,
    )
    train_grpo(model, tokenizer, dataset, training_args)

    logger.info("\n[6/6] Saving adapter...")
    save_artifacts(model, tokenizer, args.output_dir)

    if IS_MAIN:
        logger.info("\nDone.")
        logger.info("  LoRA adapter: %s", args.output_dir)


if __name__ == "__main__":
    main()
