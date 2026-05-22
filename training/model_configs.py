"""Registry and configurations for fine-tuning models."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ModelConfig:
    """Configuration dataclass for model training parameters."""

    key: str
    base_model: str
    hf_repo_id: str
    run_name: str
    tags: List[str]
    extra_template_kwargs: Dict[str, Any] = field(default_factory=dict)
    max_seq_length: int = 8192
    lora_r: int = 16
    lora_alpha: int = 32
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    per_device_eval_batch_size: int = 4

    # Runtime properties added via CLI
    _data_path: str | None = None
    _train_samples: int | None = None
    _smoke_test: bool = False


HF_USERNAME = os.environ.get("HF_USERNAME", "ahelkadyy")

MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "qwen3_8b": ModelConfig(
        key="qwen3_8b",
        base_model="unsloth/Qwen3-8B",
        hf_repo_id=f"{HF_USERNAME}/Qwen3-8B-UnBias-Plus-SFT-Instruct",
        run_name="qwen3-8b-unbias-sft",
        tags=["qwen3", "8b", "sft", "bias-detection", "unbias-plus"],
        extra_template_kwargs={"enable_thinking": False},
    ),
    "qwen35_4b": ModelConfig(
        key="qwen35_4b",
        base_model="unsloth/Qwen3.5-4B",
        hf_repo_id=f"{HF_USERNAME}/Qwen3.5-4B-UnBias-Plus-SFT-Instruct",
        run_name="qwen35-4b-unbias-sft",
        tags=["qwen3.5", "4b", "sft", "bias-detection", "unbias-plus"],
        extra_template_kwargs={"enable_thinking": False},
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        per_device_eval_batch_size=1,
        max_seq_length=4096,
    ),
    "gemma4_e4b": ModelConfig(
        key="gemma4_e4b",
        base_model="unsloth/gemma-4-E4B-it",
        hf_repo_id=f"{HF_USERNAME}/Gemma4-E4B-UnBias-Plus-SFT-Instruct",
        run_name="gemma4-e4b-unbias-sft",
        tags=["gemma4", "e4b", "sft", "bias-detection", "unbias-plus"],
        extra_template_kwargs={"enable_thinking": False},
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=1,
        max_seq_length=3072,
    ),
    "llama31_8b": ModelConfig(
        key="llama31_8b",
        base_model="unsloth/Meta-Llama-3.1-8B-Instruct",
        hf_repo_id=f"{HF_USERNAME}/Llama-3.1-8B-UnBias-Plus-SFT-Instruct",
        run_name="llama31-8b-unbias-sft",
        tags=["llama3.1", "8b", "sft", "bias-detection", "unbias-plus"],
    ),
    "ministral_8b": ModelConfig(
        key="ministral_8b",
        base_model="unsloth/Ministral-3-8B-Instruct-2512",
        hf_repo_id=f"{HF_USERNAME}/Ministral-3-8B-UnBias-Plus-SFT-Instruct",
        run_name="ministral-8b-unbias-sft",
        tags=["ministral3", "8b", "sft", "bias-detection", "unbias-plus"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        per_device_eval_batch_size=1,
        max_seq_length=4096,
    ),
}
