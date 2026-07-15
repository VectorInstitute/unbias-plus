"""Typed recipe configuration: dataclasses, a YAML loader, and a registry.

A "recipe" fully specifies one fine-tuning setup as data (a YAML file under
``configs/``): which prompt to use, LoRA rank, optimizer schedule, per-field loss
weights, optional anti-laundering unlikelihood terms, and dataset curation. Code
never branches on a recipe name; it reads a :class:`RecipeConfig`.

Add a new recipe by copying a YAML file and registering its name in
:data:`REGISTRY` (or pass an explicit path to :func:`load_recipe`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


CONFIG_DIR = Path(__file__).resolve().parent / "configs"

# Recipe name -> YAML filename (relative to CONFIG_DIR).
REGISTRY: dict[str, str] = {
    "baseline": "baseline.yaml",
    "field_weighted": "field_weighted.yaml",
    "token_unlikelihood": "token_unlikelihood.yaml",
    "phrase_unlikelihood": "phrase_unlikelihood.yaml",
    "bias_weighted": "bias_weighted.yaml",
}

# The canonical segment fields emitted in the training target, in order. This
# order is load-bearing: the weighted-span parser walks JSON keys in this order.
SEGMENT_FIELDS = ["original", "replacement", "severity", "bias_type", "reasoning"]

VALID_BIAS_TYPES = frozenset(
    {
        "loaded_language",
        "euphemism",
        "dehumanizing_language",
        "opinion_as_fact",
        "unsupported_generalization",
        "stereotypical_association",
        "sensationalism",
        "informational_bias",
    }
)
VALID_SEGMENT_SEVERITY = frozenset({"Low", "Medium", "High"})


@dataclass
class LoraConfig:
    """LoRA adapter hyperparameters."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.0


@dataclass
class OptimConfig:
    """Optimizer and schedule hyperparameters."""

    learning_rate: float = 1e-4
    epochs: int = 5
    per_device_batch: int = 1
    grad_accum: int = 16
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    lr_scheduler: str = "cosine"
    max_grad_norm: float = 1.0
    eval_steps: int = 100
    save_steps: int = 100
    early_stopping_patience: int = 3


@dataclass
class WeightsConfig:
    """Per-field token loss weights applied inside the assistant JSON.

    ``base_completion`` applies to all supervised (assistant) tokens; the field
    weights override it on the corresponding JSON value spans. All-ones reproduces
    plain completion-only cross-entropy.
    """

    base_completion: float = 1.0
    original: float = 1.0
    replacement: float = 1.0
    reasoning: float = 1.0
    unbiased_text: float = 1.0


@dataclass
class BiasTypeWeighting:
    """Optional per-bias-type multipliers on segment/rewrite weights (capped).

    When present, the replacement/reasoning weights become
    ``WeightsConfig.<field> * field_multipliers[bias_type][<field>]`` (capped),
    and the ``unbiased_text`` weight uses the hardest present bias type via
    ``rewrite_multipliers``.
    """

    field_multipliers: dict[str, dict[str, float]] = field(default_factory=dict)
    rewrite_multipliers: dict[str, float] = field(default_factory=dict)
    max_replacement: float = 1e9
    max_reasoning: float = 1e9
    max_unbiased_text: float = 1e9


@dataclass
class TokenUnlikelihood:
    """Token-level unlikelihood penalty on soft-debias words in hard cases."""

    lambda_weight: float = 0.0
    forbidden_strings: list[str] = field(default_factory=list)


@dataclass
class PhraseUnlikelihood:
    """Phrase-level unlikelihood penalty on exact laundering continuations."""

    lambda_weight: float = 0.0
    max_ids_per_pos: int = 8
    phrases: list[str] = field(default_factory=list)


@dataclass
class DataConfig:
    """Dataset curation applied to the train pool before formatting."""

    # Hard-case = a sample containing a segment whose bias_type is in
    # ``soft_debias_bias_types``. Only one of the two upsampling knobs is used:
    # an integer-style ``multiplier`` (>=1 duplicates) or an ``extra_fraction``.
    hard_upsample_multiplier: float = 1.0
    hard_upsample_extra_fraction: float = 0.0
    filter_soft_debias_targets: bool = False
    curated_hard_case_repeat: int = 0
    augment_reasoning_cue: bool = False
    soft_debias_bias_types: list[str] = field(
        default_factory=lambda: [
            "stereotypical_association",
            "unsupported_generalization",
        ]
    )


@dataclass
class WandbConfig:
    """Weights & Biases logging configuration."""

    enabled: bool = True
    project: str = "unbias-sft"
    run_name: str = "unbias-sft-run"


@dataclass
class RecipeConfig:
    """A complete, self-contained fine-tuning recipe."""

    name: str
    description: str
    prompt: str
    base_model: str = "unsloth/Qwen3-8B"
    max_seq_length: int = 8192
    train_samples: int = 5000
    seed: int = 42
    heldout_size: int = 250
    eval_size: int = 250
    lora: LoraConfig = field(default_factory=LoraConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    weights: WeightsConfig = field(default_factory=WeightsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    bias_type_weighting: BiasTypeWeighting | None = None
    token_unlikelihood: TokenUnlikelihood | None = None
    phrase_unlikelihood: PhraseUnlikelihood | None = None


def _build_nested(cls: type, payload: dict[str, Any] | None) -> Any:
    """Instantiate a dataclass from a mapping, ignoring unknown keys.

    Unknown keys are rejected loudly so config typos surface immediately.
    """
    if payload is None:
        return None
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(payload) - known
    if unknown:
        msg = f"Unknown keys for {cls.__name__}: {sorted(unknown)}"
        raise ValueError(msg)
    return cls(**payload)


def _from_dict(payload: dict[str, Any]) -> RecipeConfig:
    """Construct a :class:`RecipeConfig` from a parsed YAML mapping."""
    payload = dict(payload)  # shallow copy; we pop nested sections

    lora = _build_nested(LoraConfig, payload.pop("lora", None)) or LoraConfig()
    optim = _build_nested(OptimConfig, payload.pop("optim", None)) or OptimConfig()
    weights = (
        _build_nested(WeightsConfig, payload.pop("weights", None)) or WeightsConfig()
    )
    data = _build_nested(DataConfig, payload.pop("data", None)) or DataConfig()
    wandb_cfg = _build_nested(WandbConfig, payload.pop("wandb", None)) or WandbConfig()
    bias_type_weighting = _build_nested(
        BiasTypeWeighting, payload.pop("bias_type_weighting", None)
    )
    token_unlikelihood = _build_nested(
        TokenUnlikelihood, payload.pop("token_unlikelihood", None)
    )
    phrase_unlikelihood = _build_nested(
        PhraseUnlikelihood, payload.pop("phrase_unlikelihood", None)
    )

    top_known = {
        "name",
        "description",
        "prompt",
        "base_model",
        "max_seq_length",
        "train_samples",
        "seed",
        "heldout_size",
        "eval_size",
    }
    unknown = set(payload) - top_known
    if unknown:
        msg = f"Unknown top-level recipe keys: {sorted(unknown)}"
        raise ValueError(msg)

    return RecipeConfig(
        lora=lora,
        optim=optim,
        weights=weights,
        data=data,
        wandb=wandb_cfg,
        bias_type_weighting=bias_type_weighting,
        token_unlikelihood=token_unlikelihood,
        phrase_unlikelihood=phrase_unlikelihood,
        **payload,
    )


def resolve_config_path(recipe: str) -> Path:
    """Resolve a recipe name or path to a config file path.

    ``recipe`` may be a registered name (see :data:`REGISTRY`) or a direct path
    to a YAML file.
    """
    if recipe in REGISTRY:
        return CONFIG_DIR / REGISTRY[recipe]
    path = Path(recipe)
    if path.exists():
        return path
    valid = ", ".join(sorted(REGISTRY))
    msg = (
        f"Unknown recipe {recipe!r}. Registered recipes: {valid}. Or pass a YAML path."
    )
    raise ValueError(msg)


def load_recipe(recipe: str) -> RecipeConfig:
    """Load and validate a recipe by registered name or explicit YAML path.

    Parameters
    ----------
    recipe
        A registered recipe name (see :data:`REGISTRY`) or a path to a YAML file.

    Returns
    -------
    RecipeConfig
        The fully-populated, typed recipe.
    """
    path = resolve_config_path(recipe)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        msg = f"Recipe file {path} must contain a mapping at the top level."
        raise ValueError(msg)
    return _from_dict(payload)
