"""Configurable, train_4-driven SFT recipes for UnBias-Plus.

This package expresses several fine-tuning setups (baseline SFT, field-weighted
loss, token/phrase anti-laundering unlikelihood, and per-bias-type weighting) as
data-only recipe configs sharing one weighted trainer, one data module, and a
set of named prompts.

Run from the ``training/`` directory, for example::

    python -m recipes.train --recipe bias_weighted
    python -m recipes.eval  --recipe bias_weighted --jsonl heldout.jsonl --limit 10
"""
