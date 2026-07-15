"""Weighted causal-LM trainer with optional anti-laundering unlikelihood.

All recipes share this single trainer. Uniform loss weights and no unlikelihood
reproduce standard completion-only cross-entropy; the field/bias-type weights and
the token/phrase unlikelihood terms are what differentiate the recipes.

Three loss components (the latter two are opt-in):

1. Token-weighted cross-entropy over supervised (assistant) tokens.
2. Token-level unlikelihood: penalize probability mass on a fixed set of
   soft-debias token ids at "anti-soft" positions (hard-case replacement /
   rewrite spans).
3. Phrase-level unlikelihood: penalize the specific next-token that would
   continue an exact laundering phrase, at anti-soft positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812 — conventional alias for functional
from transformers import Trainer


@dataclass
class WeightedDataCollator:
    """Pad a batch of formatted samples to a common length.

    Pads ``input_ids``/``attention_mask``/``labels``/``loss_weights`` and the
    per-position ``phrase_forbid_ids`` matrix (width fixed by the recipe).
    """

    tokenizer: Any
    phrase_width: int = 1
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate and pad a list of formatted samples into tensors."""
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_len = ((max_len + multiple - 1) // multiple) * multiple

        pad_id = self.tokenizer.pad_token_id
        width = self.phrase_width
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        loss_weights: list[list[float]] = []
        anti_soft_mask: list[list[float]] = []
        phrase_forbid_ids: list[list[list[int]]] = []

        empty_row = [-100] * width
        for feature in features:
            n = len(feature["input_ids"])
            pad = max_len - n
            input_ids.append(list(feature["input_ids"]) + [pad_id] * pad)
            attention_mask.append(list(feature["attention_mask"]) + [0] * pad)
            labels.append(list(feature["labels"]) + [-100] * pad)
            loss_weights.append(list(feature["loss_weights"]) + [0.0] * pad)
            anti_soft_mask.append(
                list(feature.get("anti_soft_mask", [0.0] * n)) + [0.0] * pad
            )
            rows = [list(r) for r in feature.get("phrase_forbid_ids", [])]
            rows = rows + [list(empty_row) for _ in range(max_len - len(rows))]
            phrase_forbid_ids.append(rows)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "loss_weights": torch.tensor(loss_weights, dtype=torch.float32),
            "anti_soft_mask": torch.tensor(anti_soft_mask, dtype=torch.float32),
            "phrase_forbid_ids": torch.tensor(phrase_forbid_ids, dtype=torch.long),
        }


class WeightedLossTrainer(Trainer):
    """Causal-LM trainer with token-weighted CE and unlikelihood penalties."""

    def __init__(
        self,
        *args: Any,
        forbidden_token_ids: list[int] | None = None,
        token_unlikelihood_lambda: float = 0.0,
        phrase_unlikelihood_lambda: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Store unlikelihood configuration alongside the base trainer state."""
        super().__init__(*args, **kwargs)
        self.forbidden_token_ids = sorted(set(forbidden_token_ids or []))
        self.token_unlikelihood_lambda = float(token_unlikelihood_lambda)
        self.phrase_unlikelihood_lambda = float(phrase_unlikelihood_lambda)

    def _token_unlikelihood(
        self,
        shift_logits: torch.Tensor,
        shift_labels: torch.Tensor,
        anti_soft_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        """Token-set unlikelihood on forbidden ids at anti-soft positions."""
        shift_anti = anti_soft_mask[..., 1:].contiguous().to(shift_logits.device)
        positions = (shift_anti > 0) & (shift_labels.to(shift_logits.device) != -100)
        if not bool(positions.any()):
            return None
        selected = shift_logits[positions].float()
        forbidden = torch.tensor(
            self.forbidden_token_ids, dtype=torch.long, device=selected.device
        )
        # p(forbidden set | prefix) without materializing a full softmax.
        log_p = torch.logsumexp(
            selected.index_select(-1, forbidden), dim=-1
        ) - torch.logsumexp(selected, dim=-1)
        p_forbidden = log_p.exp().clamp(max=1.0 - 1e-6)
        return -torch.log1p(-p_forbidden).mean()

    def _phrase_unlikelihood(
        self,
        shift_logits: torch.Tensor,
        shift_labels: torch.Tensor,
        phrase_forbid_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        """Phrase-level unlikelihood on continuation ids at anti-soft positions."""
        shift_phrase = (
            phrase_forbid_ids[..., 1:, :].contiguous().to(shift_logits.device)
        )
        valid = (shift_phrase >= 0).any(dim=-1) & (
            shift_labels.to(shift_logits.device) != -100
        )
        if not bool(valid.any()):
            return None
        selected = shift_logits[valid].float()  # [N, V]
        ids = shift_phrase[valid]  # [N, K]
        id_mask = ids >= 0
        forbidden_logits = selected.gather(1, ids.clamp_min(0))
        forbidden_logits = forbidden_logits.masked_fill(~id_mask, float("-inf"))
        log_p = torch.logsumexp(forbidden_logits, dim=-1) - torch.logsumexp(
            selected, dim=-1
        )
        p_forbidden = log_p.exp().clamp(max=1.0 - 1e-6)
        return -torch.log1p(-p_forbidden).mean()

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        """Compute token-weighted CE plus any configured unlikelihood terms."""
        labels = inputs.pop("labels")
        loss_weights = inputs.pop("loss_weights")
        anti_soft_mask = inputs.pop("anti_soft_mask", None)
        phrase_forbid_ids = inputs.pop("phrase_forbid_ids", None)

        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = loss_weights[..., 1:].contiguous().to(shift_logits.device)

        flat_logits = shift_logits.view(-1, shift_logits.size(-1)).float()
        flat_labels = shift_labels.view(-1).to(flat_logits.device)
        flat_weights = shift_weights.view(-1)

        token_loss = F.cross_entropy(
            flat_logits, flat_labels, reduction="none", ignore_index=-100
        )
        active = (flat_labels != -100).float()
        weighted = token_loss * flat_weights * active
        denom = (flat_weights * active).sum().clamp_min(1.0)
        loss = weighted.sum() / denom

        if (
            anti_soft_mask is not None
            and self.token_unlikelihood_lambda > 0
            and self.forbidden_token_ids
        ):
            term = self._token_unlikelihood(shift_logits, shift_labels, anti_soft_mask)
            if term is not None:
                loss = loss + self.token_unlikelihood_lambda * term

        if phrase_forbid_ids is not None and self.phrase_unlikelihood_lambda > 0:
            term = self._phrase_unlikelihood(
                shift_logits, shift_labels, phrase_forbid_ids
            )
            if term is not None:
                loss = loss + self.phrase_unlikelihood_lambda * term

        return (loss, outputs) if return_outputs else loss
