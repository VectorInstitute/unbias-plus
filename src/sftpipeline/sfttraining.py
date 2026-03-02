"""Fine-tune Qwen2.5-7B using SFT with completion-only loss on VLDBench data."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple, cast


os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


INPUT_PATH = "/Unbias/checkpoints_gpt/checkpoint.json"
OUTPUT_DIR = "/Unbias/Models/Qwen25_Unbias_4k_5epoch"

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 8192
TRAIN_SAMPLES = 4000  # use first 4k samples


SYSTEM_PROMPT = """
You are an expert linguist and bias detection specialist. Your job is to:
1. Identify ALL biased, loaded, or prejudiced language in the given text
2. Rewrite the text in a neutral, factual, unbiased way
3. For each biased segment, explain why it is biased and how severe it is

## Global Severity level
0 = neutral
2 = recurring biased framing
3 = strong persuasive tone
4 = inflammatory rhetoric

## BIAS TYPES TO DETECT
- **Loaded language**: Words that carry strong emotional connotations (e.g. "flooding", "invasion", "destroying")
- **Dehumanizing framing**: Language that strips dignity from groups of people
- **False generalizations**: Sweeping statements about groups ("they always", "all of them")
- **Framing bias**: Selective word choices that imply a particular viewpoint
- **Euphemisms or dysphemisms**: Softening or hardening language to manipulate perception
- **Politically charged terminology**: Labels used to provoke rather than describe
- **Sensationalism**: Exaggerated language to evoke emotional responses

## SEVERITY SCALE
- **high**: Dehumanizing, hateful, or strongly prejudiced language
- **medium**: Framing bias, loaded terms, misleading generalizations
- **low**: Subtle word choice bias, mild framing issues

## SEGMENT RULES
- A segment is a consecutive sequence of words that form a single biased idea
- If two biased words are adjacent and part of the same biased idea, treat them as ONE segment
- If biased words are separated by neutral words, treat them as SEPARATE segments
- The "original" field must be the EXACT substring as it appears in the input text (case-sensitive, no changes)

OUTPUT SCHEMA (ALL KEYS REQUIRED):

{
  "binary_label": "biased" | "unbiased",
  "severity": 0 | 2 | 3 | 4,
  "bias_found": true | false,
  "biased_segments": [
    {
      "original": "exact substring from input text",
      "replacement": "neutral alternative phrase",
      "severity": "high" | "medium" | "low",
      "bias_type": "loaded language | dehumanizing framing | false generalizations | framing bias | euphemism/dysphemism | politically charged terminology | sensationalism",
      "reasoning": "1-2 sentence explanation"
    }
  ],
  "unbiased_text": "Full rewritten neutral article"
}

Rules:
- "original" MUST be exact case-sensitive substring.
- Only modify phrases listed in biased_segments.
- Preserve all factual information.
- If no bias: severity=0, bias_found=false, biased_segments=[]
- Return ONLY valid JSON.
""".strip()


def load_raw_data(path: str, limit: int) -> List[Dict[str, Any]]:
    """Load and optionally truncate raw JSON training data."""
    with open(path, "r", encoding="utf-8") as f:
        data = cast(List[Dict[str, Any]], json.load(f))
    return data[:limit]


def load_base_model(model_name: str, max_seq_length: int) -> Tuple[Any, Any]:
    """Load Qwen model in 4-bit mode using Unsloth."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def format_sample(sample: Dict[str, Any], tokenizer: Any) -> Dict[str, str]:
    """Convert raw sample into completion-only chat training format."""
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
            "content": f"Analyze the following article:\n\n{sample['article_text']}",
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_text = prompt + json.dumps(completion, ensure_ascii=False)
    return {"text": full_text}


def build_dataset(raw_data: List[Dict[str, Any]], tokenizer: Any) -> Dataset:
    """Build HuggingFace Dataset with formatted training text."""
    dataset = Dataset.from_list(raw_data)
    return dataset.map(
        lambda x: format_sample(x, tokenizer),
        remove_columns=dataset.column_names,
    )


def filter_by_token_length(
    dataset: Dataset, tokenizer: Any, max_length: int
) -> Dataset:
    """Filter dataset samples exceeding max token length."""

    def is_valid(example: Dict[str, str]) -> bool:
        tokens = tokenizer(example["text"], add_special_tokens=False)["input_ids"]
        return len(tokens) <= max_length

    return dataset.filter(is_valid)


def print_token_stats(dataset: Dataset, tokenizer: Any) -> None:
    """Print token statistics for dataset."""
    lengths: List[int] = []
    max_tokens = 0

    for idx in range(len(dataset)):
        tokens = tokenizer(dataset[idx]["text"], add_special_tokens=False)["input_ids"]
        token_length = len(tokens)
        lengths.append(token_length)
        max_tokens = max(max_tokens, token_length)

    print(f"Max token length: {max_tokens}")
    print(f"Average token length: {sum(lengths) / len(lengths):.2f}")
    print(
        f"Samples > {MAX_SEQ_LENGTH} tokens: "
        f"{sum(token_length > MAX_SEQ_LENGTH for token_length in lengths)}"
    )


def configure_lora(model: Any) -> Any:
    """Attach LoRA adapters to model."""
    return FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
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


def build_training_args(output_dir: str) -> SFTConfig:
    """Create SFT training configuration."""
    return SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=5,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=20,
        save_total_limit=3,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=1.0,
        report_to="none",
        seed=42,
        dataset_text_field="text",
        completion_only_loss=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        dataset_num_proc=4,
        max_length=MAX_SEQ_LENGTH,
        torch_compile=False,
    )


def train_model(
    model: Any, tokenizer: Any, dataset: Dataset, training_args: SFTConfig
) -> None:
    """Train model using TRL SFTTrainer."""
    try:
        # Newer TRL uses `processing_class` instead of `tokenizer`
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
            processing_class=tokenizer,
        )
    except TypeError:
        # Older TRL: fall back to minimal signature
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
        )

    print("Starting Qwen2.5-7B completion-only training...")
    trainer.train()


def main() -> None:
    """Run full SFT fine-tuning pipeline."""
    print("Loading raw data...")
    raw_data = load_raw_data(INPUT_PATH, TRAIN_SAMPLES)
    print(f"Loaded {len(raw_data)} samples")

    print("Loading base model...")
    model, tokenizer = load_base_model(BASE_MODEL, MAX_SEQ_LENGTH)

    print("Building dataset...")
    dataset = build_dataset(raw_data, tokenizer)

    print("Checking token lengths...")
    print_token_stats(dataset, tokenizer)

    print("Filtering overlength samples...")
    dataset = filter_by_token_length(dataset, tokenizer, MAX_SEQ_LENGTH)
    print("Dataset size after filtering:", len(dataset))

    print("Configuring LoRA...")
    model = configure_lora(model)

    print("Building training args...")
    training_args = build_training_args(OUTPUT_DIR)

    train_model(model, tokenizer, dataset, training_args)

    print("Saving model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
