"""Fine-tune any registered model using SFT with completion-only loss.

Usage (via Slurm launcher):
    MODEL_KEY=llama31_8b sbatch train/launch_sft.sh

Usage (direct, for debugging):
    python train/train_sft.py --model-key qwen3_8b

Export    : merged 16-bit locally (HuggingFace push handled by push_to_hub.py)
Tracking  : Weights & Biases (set WANDB_API_KEY in .env)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

# ============================================================
# IMPORTANT: unsloth MUST be imported first before anything
# else to ensure all kernel optimizations are applied.
# ============================================================
import unsloth  # noqa: F401 — must be first
from dotenv import load_dotenv


load_dotenv()

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from model_configs import MODEL_REGISTRY, ModelConfig  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402
from unsloth import FastLanguageModel  # noqa: E402

import wandb  # noqa: E402


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH = "vector-institute/unbias-plus-dataset"
OUTPUT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models")

# ---------------------------------------------------------------------------
# System prompt (model-agnostic)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert linguist and bias detection specialist.
Your task is to carefully read a news article, detect ALL biased language,
and return a structured JSON response.

## BIAS TYPES
- loaded language             : words with strong emotional connotations
- dehumanizing framing        : language that strips dignity from groups
- false generalizations       : sweeping statements ("they always", "all of them")
- framing bias                : selective wording that implies a viewpoint
- euphemism/dysphemism        : softening or hardening language to manipulate perception
- politically charged terminology : labels used to provoke rather than describe
- sensationalism              : exaggerated language to evoke emotional responses

## SEGMENT RULES
- A segment is a consecutive sequence of words forming ONE biased idea.
- Prefer fewer, longer segments over many short overlapping ones.
- If two biased words are adjacent and part of the same biased idea → ONE segment.
- If biased words are separated by neutral words → SEPARATE segments.
- "original" MUST be the EXACT substring as it appears in the input (case-sensitive).
- Only modify phrases listed in biased_segments; preserve all factual content.
- Replacements must be similar in length to the original phrase. Do not use a long phrase to replace a short one.

## SEVERITY (per segment — string value)
- high   : dehumanizing, hateful, or strongly prejudiced language
- medium : framing bias, loaded terms, misleading generalizations
- low    : subtle word choice bias, mild framing issues

## GLOBAL SEVERITY (article-level — integer value)
- 0 : neutral / no bias
- 2 : recurring biased framing
- 3 : strong persuasive tone
- 4 : inflammatory rhetoric

## OUTPUT SCHEMA
Return ONLY a raw JSON object. No markdown, no code fences, no backticks.
The response must start with { and end with }.
{
  "binary_label": "biased" | "unbiased",
  "severity": 0 | 2 | 3 | 4,              // GLOBAL article-level integer
  "bias_found": true | false,
  "biased_segments": [
    {
      "original": "exact substring from input",
      "replacement": "neutral alternative phrase in the same language as original",
      "severity": "high" | "medium" | "low",   // SEGMENT-level string
      "bias_type": "loaded language | dehumanizing framing | false generalizations | framing bias | euphemism/dysphemism | politically charged terminology | sensationalism",
      "reasoning": "1-2 sentence explanation of why this is biased"
    }
  ],
  "unbiased_text": "Full rewritten neutral article in the same language as the input"
}

## REWRITE RULES
- Build unbiased_text by replacing each biased phrase with its neutral replacement from biased_segments.
- Only modify phrases listed in biased_segments — leave everything else unchanged.
- Preserve the original article's facts, structure, and length. The rewritten text must be as close in length as possible to the original. Do not add sentences, expand phrases, or elaborate. Only swap biased phrases with neutral alternatives of similar length.
- Do not add new information, opinions, or commentary.
- If the article is unbiased, return the original text exactly as-is.

## LANGUAGE HANDLING
- Always respond in the same language as the input article.
- All text fields (original, replacement, unbiased_text) must be in the article's original language.
- JSON keys must always remain in English.
- If the article's language is not well-supported, return unbiased_text in English and note the limitation in the reasoning field.

Rules:
- If no bias: severity=0, bias_found=false, biased_segments=[], unbiased_text=<original text unchanged>
- If biased: severity must be 2, 3, or 4 — never 0
- Return ONLY the JSON object. No preamble, no markdown fences.
""".strip()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> ModelConfig:
    """Resolve ModelConfig from --model-key CLI arg or MODEL_KEY env var."""
    parser = argparse.ArgumentParser(description="UnBias-Plus SFT trainer")
    parser.add_argument(
        "--model-key",
        type=str,
        default=os.environ.get("MODEL_KEY", "qwen3_8b"),
        choices=list(MODEL_REGISTRY.keys()),
        help="Key from MODEL_REGISTRY in model_configs.py",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to local training data JSON file.",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Limit to N samples after filtering (optional).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Override max_seq_length from ModelConfig (optional).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Minimal smoke test: 10 samples, 512 tokens, 1 epoch, no W&B.",
    )
    args = parser.parse_args()

    config = MODEL_REGISTRY[args.model_key]
    config._data_path = args.data_path
    config._train_samples = args.train_samples
    config._smoke_test = args.smoke_test

    # Smoke test overrides everything
    if args.smoke_test:
        config.max_seq_length = 2048
        config._train_samples = 10
    elif args.max_seq_length:
        config.max_seq_length = args.max_seq_length

    logger.info("\n%s", "=" * 60)
    logger.info("  Model key   : %s", config.key)
    logger.info("  Base model  : %s", config.base_model)
    logger.info("  HF repo     : %s", config.hf_repo_id)
    logger.info("  W&B run     : %s", config.run_name)
    logger.info("  Data path   : %s", args.data_path)
    logger.info(
        "  Batch size  : %d (accum %d)",
        config.per_device_train_batch_size,
        config.gradient_accumulation_steps,
    )
    if args.smoke_test:
        logger.info("  SMOKE TEST  : 10 samples, 512 tokens, 1 epoch")
    logger.info("%s\n", "=" * 60)
    return config


# ---------------------------------------------------------------------------
# W&B initialisation
# ---------------------------------------------------------------------------


def init_wandb(config: ModelConfig) -> None:
    """Login and initialise a W&B run. Falls back gracefully if key is missing."""
    api_key = os.environ.get("WANDB_API_KEY", "")
    if not api_key:
        logger.info("[W&B] WANDB_API_KEY not set — skipping W&B initialisation.")
        return

    wandb.login(key=api_key)
    wandb.init(
        project="unbias-plus",
        name=config.run_name,
        tags=config.tags,
        config={
            "base_model": config.base_model,
            "hf_repo_id": config.hf_repo_id,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "max_seq_length": config.max_seq_length,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "data_path": getattr(config, "_data_path", DEFAULT_DATA_PATH),
        },
    )
    logger.info("[W&B] Run initialised: %s", wandb.run.url if wandb.run else "None")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_raw_data(data_path: str) -> List[Dict[str, Any]]:
    """Load training data from local JSON file or HuggingFace dataset."""
    logger.info("  Loading dataset from: %s", data_path)
    if os.path.isfile(data_path):
        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)
    else:
        from datasets import load_dataset as hf_load  # noqa: PLC0415

        ds = hf_load(data_path, data_files="data/regenerated.json", split="train")
        data = [dict(row) for row in ds]

    # Normalize biased_segments: parse JSON string if needed
    for sample in data:
        segs = sample.get("biased_segments")
        if isinstance(segs, str):
            try:
                sample["biased_segments"] = json.loads(segs)
            except Exception:
                sample["biased_segments"] = []
        elif segs is None:
            sample["biased_segments"] = []

    return data  # type: ignore


def is_valid_sample(sample: Dict[str, Any]) -> bool:
    """Filter out malformed or incomplete samples."""
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


def format_sample(
    sample: Dict[str, Any],
    tokenizer: Any,
    config: ModelConfig,
) -> Dict[str, str]:
    """Convert a raw sample into completion-only chat training format.

    The assistant turn is included inside messages so apply_chat_template
    closes the sequence with <|im_end|>. The previous approach used
    add_generation_prompt=True and string-concatenated the JSON, leaving
    the sequence open — the model would never learn when to stop generating.
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
        **config.extra_template_kwargs,
    )

    return {"text": text}


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def build_dataset(
    raw_data: List[Dict[str, Any]],
    tokenizer: Any,
    config: ModelConfig,
) -> Dataset:
    """Filter invalid samples, format with the right template, return Dataset."""
    valid = [s for s in raw_data if is_valid_sample(s)]
    logger.info("  Valid samples after filtering: %d / %d", len(valid), len(raw_data))

    dataset = Dataset.from_list(valid)
    return dataset.map(
        lambda x: format_sample(x, tokenizer, config),
        remove_columns=dataset.column_names,
        desc="Formatting samples",
    )


# ---------------------------------------------------------------------------
# Token counting — safe for standard tokenizers and VLM processors
# ---------------------------------------------------------------------------


def _token_count(tokenizer: Any, text: str) -> int:
    """Count tokens safely for both standard tokenizers and VLM processors.

    VLM processors (Gemma4, Ministral/Pixtral, Qwen3.5/Qwen3VL) wrap a text
    tokenizer inside a processor object. Unsloth patches the outer __call__
    to handle images, which crashes on plain text. Unwrapping to the inner
    text tokenizer and calling encode() bypasses that path entirely.
    """
    tok = (
        getattr(tokenizer, "tokenizer", None)
        or getattr(tokenizer, "text_tokenizer", None)
        or tokenizer
    )
    try:
        return len(tok.encode(text, add_special_tokens=False))
    except Exception:
        try:
            return len(tok(text=text, add_special_tokens=False)["input_ids"])
        except Exception:
            return len(text.split()) * 2


def filter_by_token_length(
    dataset: Dataset,
    tokenizer: Any,
    max_length: int,
) -> Dataset:
    """Drop samples that exceed the model context window."""

    def is_within_length(example: Dict[str, str]) -> bool:
        return _token_count(tokenizer, example["text"]) <= max_length

    before = len(dataset)
    dataset = dataset.filter(is_within_length, desc="Length filter")
    logger.info("  Dropped %d overlength samples", before - len(dataset))
    return dataset


def print_token_stats(dataset: Dataset, tokenizer: Any, max_seq_length: int) -> None:
    """Log token-length distribution summary."""
    lengths = [_token_count(tokenizer, dataset[i]["text"]) for i in range(len(dataset))]
    logger.info("  Samples            : %d", len(lengths))
    logger.info("  Max tokens         : %d", max(lengths))
    logger.info("  Avg tokens         : %.0f", sum(lengths) / len(lengths))
    logger.info(
        "  > %d tokens : %d",
        max_seq_length,
        sum(length > max_seq_length for length in lengths),
    )


# ---------------------------------------------------------------------------
# Model & LoRA
# ---------------------------------------------------------------------------


def load_base_model(config: ModelConfig) -> Tuple[Any, Any]:
    """Load any registered model in full bf16 precision."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_seq_length,
        load_in_4bit=False,
        dtype=torch.bfloat16,
        use_gradient_checkpointing="unsloth",
        device_map={"": 0},
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"
    return model, tokenizer


def configure_lora(model: Any, config: ModelConfig) -> Any:
    """LoRA config — r/alpha sourced from ModelConfig."""
    return FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
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


def build_training_args(output_dir: str, config: ModelConfig) -> SFTConfig:
    """SFT config — batch size and grad accum read from ModelConfig."""
    smoke_test = getattr(config, "_smoke_test", False)
    num_epochs = 1 if smoke_test else 3
    report_to = "none" if smoke_test else ("wandb" if wandb.run is not None else "none")
    logger.info("  report_to                    = %s", report_to)
    logger.info(
        "  per_device_train_batch_size  = %d", config.per_device_train_batch_size
    )
    logger.info(
        "  gradient_accumulation_steps  = %d", config.gradient_accumulation_steps
    )

    return SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        logging_steps=1,
        report_to=report_to,
        seed=42,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        dataset_text_field="text",
        max_length=config.max_seq_length,
        completion_only_loss=True,
        remove_unused_columns=False,
        dataset_num_proc=2,
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
) -> None:
    """Train with TRL SFTTrainer, 5% eval split."""
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    logger.info("  Train: %d | Eval: %d", len(train_ds), len(eval_ds))

    try:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=training_args,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = SFTTrainer(  # type: ignore
            model=model,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=training_args,
            tokenizer=tokenizer,
        )

    # Prevent TRL from calling importlib.metadata.version("torch") on every
    # checkpoint save — crashes with +computecanada wheels that lack dist-info.
    trainer.create_model_card = lambda **kwargs: None  # type: ignore

    logger.info("Starting bf16 SFT training on single A100...")
    trainer.train()


# ---------------------------------------------------------------------------
# Export — merged 16-bit locally only
# HuggingFace push is handled separately by push_to_hub.py
# ---------------------------------------------------------------------------


def export_model(
    model: Any, tokenizer: Any, output_dir: str, config: ModelConfig
) -> None:
    """Save merged 16-bit model locally.

    HuggingFace push (16-bit and 4-bit) is intentionally handled by
    push_to_hub.py to keep the training pipeline independent of Hub availability.
    """
    merged_path = os.path.join(output_dir, "merged_16bit")

    logger.info("\n  Saving merged 16-bit to: %s", merged_path)
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
    logger.info("  Merged 16-bit saved.")

    # --- HuggingFace push disabled — run push_to_hub.py separately ---
    # model.push_to_hub_merged(
    #     config.hf_repo_id, tokenizer, save_method="merged_16bit", token=hf_token
    # )
    # del model
    # torch.cuda.empty_cache()
    # repo_4bit = config.hf_repo_id + "-4bit"
    # model_4bit, tokenizer_4bit = FastLanguageModel.from_pretrained(
    #     merged_path, load_in_4bit=True, ...
    # )
    # model_4bit.push_to_hub(repo_4bit, token=hf_token)
    # tokenizer_4bit.push_to_hub(repo_4bit, token=hf_token)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Full SFT fine-tuning pipeline for any model in MODEL_REGISTRY."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    config = parse_args()
    output_dir = os.path.join(OUTPUT_BASE, config.key)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("\n[1/9] Initialising W&B...")
    if not getattr(config, "_smoke_test", False):
        init_wandb(config)
    else:
        logger.info("  Smoke test — skipping W&B.")

    logger.info("\n[2/9] Loading raw data from local file...")
    raw_data = load_raw_data(getattr(config, "_data_path", DEFAULT_DATA_PATH))
    train_samples = getattr(config, "_train_samples", None)
    if train_samples:
        raw_data = raw_data[:train_samples]
        logger.info("  Loaded %d samples (limited to %d)", len(raw_data), train_samples)
    else:
        logger.info("  Loaded %d samples", len(raw_data))

    logger.info("\n[3/9] Loading base model (bf16, no quantization)...")
    model, tokenizer = load_base_model(config)

    logger.info("\n[4/9] Building & filtering dataset...")
    dataset = build_dataset(raw_data, tokenizer, config)

    logger.info("\n[5/9] Token length statistics...")
    print_token_stats(dataset, tokenizer, config.max_seq_length)

    logger.info("\n[6/9] Filtering overlength samples...")
    dataset = filter_by_token_length(dataset, tokenizer, config.max_seq_length)
    logger.info("  Final dataset size: %d", len(dataset))

    logger.info("\n[7/9] Configuring LoRA...")
    model = configure_lora(model, config)

    logger.info("\n[8/9] Training...")
    training_args = build_training_args(output_dir, config)
    train_model(model, tokenizer, dataset, training_args)

    logger.info("\n[9/9] Saving adapter + exporting merged 16-bit model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("  LoRA adapter saved to: %s", output_dir)

    export_model(model, tokenizer, output_dir, config)

    if wandb.run is not None:
        wandb.finish()

    logger.info("\nDone.")
    logger.info("  LoRA adapter   : %s", output_dir)
    logger.info("  Merged 16-bit  : %s/merged_16bit/", output_dir)
    logger.info("  HuggingFace    : run push_to_hub.py --model-key %s", config.key)


if __name__ == "__main__":
    main()
