# ruff: noqa: E402, I001 — unsloth must be imported before torch/datasets/trl,
# and TORCHDYNAMO_DISABLE must be set before importing unsloth, which forces a
# non-standard import order that ruff's isort + E402 would otherwise reject.
"""Single configurable SFT entry point for all recipes.

Loads a recipe by name (see ``recipes.config.REGISTRY``) or YAML path, pulls the
``train_4`` split from the Hub, curates and formats it per the recipe, and trains
a Qwen3-8B LoRA with the shared weighted trainer, exporting a merged 16-bit model.

Run from the ``training/`` directory::

    python -m recipes.train --recipe bias_weighted
    python -m recipes.train --recipe baseline --train-samples 20 --load-in-4bit
"""

from __future__ import annotations

import os


os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import unsloth  # noqa: F401 — must precede torch / datasets / trl

import argparse
from pathlib import Path
from typing import Any

import torch
from transformers import EarlyStoppingCallback, TrainingArguments
from unsloth import FastLanguageModel

from recipes import data
from recipes.config import RecipeConfig, load_recipe
from recipes.losses import WeightedDataCollator, WeightedLossTrainer


TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a recipe training run."""
    parser = argparse.ArgumentParser(description="Train an UnBias-Plus SFT recipe.")
    parser.add_argument(
        "--recipe",
        required=True,
        help="Registered recipe name (see recipes.config.REGISTRY) or a YAML path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: outputs/<recipe-name>).",
    )
    parser.add_argument(
        "--heldout-path",
        default=None,
        help="Where to write the held-out JSONL (default: <output-dir>/heldout.jsonl).",
    )
    parser.add_argument("--base-model", default=None, help="Override the base model.")
    parser.add_argument(
        "--train-samples", type=int, default=None, help="Override train_samples."
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    parser.add_argument(
        "--max-seq-length", type=int, default=None, help="Override max_seq_length."
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the base model in 4-bit (QLoRA) to reduce VRAM.",
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable Weights & Biases logging."
    )
    return parser.parse_args()


def apply_overrides(cfg: RecipeConfig, args: argparse.Namespace) -> RecipeConfig:
    """Apply CLI overrides onto a loaded recipe config in place."""
    if args.base_model is not None:
        cfg.base_model = args.base_model
    if args.train_samples is not None:
        cfg.train_samples = args.train_samples
    if args.epochs is not None:
        cfg.optim.epochs = args.epochs
    if args.max_seq_length is not None:
        cfg.max_seq_length = args.max_seq_length
    if args.no_wandb:
        cfg.wandb.enabled = False
    return cfg


def setup_wandb(cfg: RecipeConfig) -> None:
    """Configure W&B via env vars so the trainer logs to the right project."""
    if not cfg.wandb.enabled:
        os.environ["WANDB_DISABLED"] = "true"
        return
    os.environ.setdefault("WANDB_PROJECT", cfg.wandb.project)
    os.environ.setdefault("WANDB_LOG_MODEL", "false")
    print(f"  W&B project: {cfg.wandb.project} | run: {cfg.wandb.run_name}")


def load_base_model(cfg: RecipeConfig, load_in_4bit: bool) -> tuple[Any, Any]:
    """Load the base Qwen3 model and tokenizer via Unsloth."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=torch.bfloat16,
        use_gradient_checkpointing="unsloth",
        device_map={"": 0},
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return model, tokenizer


def configure_lora(model: Any, cfg: RecipeConfig) -> Any:
    """Attach a LoRA adapter with rank-stabilised scaling."""
    return FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
        use_rslora=True,
        loftq_config=None,
        target_modules=TARGET_MODULES,
    )


def build_training_args(cfg: RecipeConfig, output_dir: str) -> TrainingArguments:
    """Build HF ``TrainingArguments`` from the recipe's optimizer config."""
    optim = cfg.optim
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=optim.per_device_batch,
        gradient_accumulation_steps=optim.grad_accum,
        per_device_eval_batch_size=optim.per_device_batch,
        num_train_epochs=optim.epochs,
        learning_rate=optim.learning_rate,
        warmup_ratio=optim.warmup_ratio,
        lr_scheduler_type=optim.lr_scheduler,
        weight_decay=optim.weight_decay,
        max_grad_norm=optim.max_grad_norm,
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        logging_steps=10,
        report_to="wandb" if cfg.wandb.enabled else "none",
        run_name=cfg.wandb.run_name if cfg.wandb.enabled else None,
        seed=cfg.seed,
        eval_strategy="steps",
        eval_steps=optim.eval_steps,
        save_strategy="steps",
        save_steps=optim.save_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
        torch_compile=False,
    )


def export_model(model: Any, tokenizer: Any, output_dir: str) -> None:
    """Merge the LoRA adapter into the base model and save as merged 16-bit."""
    merged_path = os.path.join(output_dir, "merged_16bit")
    print(f"\n  Merging adapter and saving to: {merged_path}")
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
    print("  Export complete.")


def _preview_first_supervised_token(
    trainer: WeightedLossTrainer, tokenizer: Any
) -> None:
    """Print a preview around the first supervised token as a sanity check."""
    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch["labels"][0]
    first = next((i for i, x in enumerate(labels.tolist()) if x != -100), None)
    if first is not None:
        preview_ids = batch["input_ids"][0][max(0, first - 20) : first + 40]
        print("  First supervised-token preview:")
        print(tokenizer.decode(preview_ids, skip_special_tokens=False))


def main() -> None:
    """Run the full recipe training pipeline."""
    args = parse_args()
    cfg = apply_overrides(load_recipe(args.recipe), args)

    output_dir = args.output_dir or os.path.join("outputs", cfg.name)
    heldout_path = args.heldout_path or os.path.join(output_dir, "heldout.jsonl")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n=== Recipe: {cfg.name} ===")
    print(f"  {cfg.description.strip()}")
    print(f"  prompt={cfg.prompt} | lora r={cfg.lora.r} a={cfg.lora.alpha}")

    print("\n[1/8] Loading train_4 from the Hub...")
    raw = data.load_train4(cfg.train_samples)
    print(f"  Loaded {len(raw)} rows")

    print("\n[2/8] Validating + carving held-out test set...")
    valid = data.filter_valid(raw)
    # Scale the held-out size down for tiny/smoke runs so a train pool remains.
    heldout_n = min(cfg.heldout_size, max(1, len(valid) // 5))
    train_pool, heldout = data.carve_heldout(valid, heldout_n, cfg.seed)
    data.save_jsonl(heldout, heldout_path)

    print("\n[3/8] Applying recipe curation...")
    train_pool = data.apply_curation(train_pool, cfg)

    print("\n[4/8] Loading base model...")
    model, tokenizer = load_base_model(cfg, args.load_in_4bit)

    print("\n[5/8] Formatting weighted dataset...")
    phrase_seqs = (
        data.make_phrase_token_sequences(tokenizer, cfg.phrase_unlikelihood.phrases)
        if cfg.phrase_unlikelihood is not None
        else []
    )
    forbidden_ids = (
        data.make_forbidden_token_ids(
            tokenizer, cfg.token_unlikelihood.forbidden_strings
        )
        if cfg.token_unlikelihood is not None
        else []
    )
    dataset = data.format_dataset(train_pool, tokenizer, cfg, phrase_seqs)
    data.print_token_stats(dataset, cfg.max_seq_length)
    dataset = data.filter_by_token_length(dataset, cfg.max_seq_length)
    print(f"  Final dataset size: {len(dataset)}")

    print("\n[6/8] Configuring LoRA + W&B...")
    model = configure_lora(model, cfg)
    setup_wandb(cfg)

    print("\n[7/8] Training...")
    # Clamp the eval split for tiny/smoke runs (fixed 250 would exceed the pool).
    eval_n = min(cfg.eval_size, max(1, len(dataset) // 5))
    split = dataset.train_test_split(test_size=eval_n, seed=cfg.seed)
    training_args = build_training_args(cfg, output_dir)
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=WeightedDataCollator(
            tokenizer, phrase_width=data.phrase_width(cfg)
        ),
        forbidden_token_ids=forbidden_ids,
        token_unlikelihood_lambda=(
            cfg.token_unlikelihood.lambda_weight if cfg.token_unlikelihood else 0.0
        ),
        phrase_unlikelihood_lambda=(
            cfg.phrase_unlikelihood.lambda_weight if cfg.phrase_unlikelihood else 0.0
        ),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg.optim.early_stopping_patience
            )
        ],
    )
    print(f"  Train: {len(split['train'])} | Eval: {len(split['test'])}")
    _preview_first_supervised_token(trainer, tokenizer)
    trainer.train()

    print("\n[8/8] Saving adapter + exporting merged 16-bit model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    export_model(model, tokenizer, output_dir)

    print("\nDone.")
    print(f"  LoRA adapter  : {output_dir}")
    print(f"  Merged 16-bit : {output_dir}/merged_16bit/")
    print(f"  Held-out test : {heldout_path} ({len(heldout)} rows)")


if __name__ == "__main__":
    main()
