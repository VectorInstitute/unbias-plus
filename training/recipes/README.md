# Configurable SFT recipes (train_4)

One configurable fine-tuning pipeline that expresses several debiasing setups as
data-only **recipe configs**. Every recipe shares the same weighted trainer, data
module, and prompt library — they differ only in their YAML config
(`configs/*.yaml`). Training data always comes from the Hub
[`vector-institute/unbias-plus-dataset`](https://huggingface.co/datasets/vector-institute/unbias-plus-dataset),
config `train_4` (the newest, highest-quality span-level split).

This lives alongside the older `training/` scripts; it does not replace them.

## Recipes

| Recipe | Idea | Prompt | Loss / data highlights |
|--------|------|--------|------------------------|
| `baseline` | Plain completion-only SFT | `conservative` | Uniform token loss (equivalent to standard CE) |
| `field_weighted` | Emphasize rewrite fields | `rewrite_editor` | Upweight replacement/reasoning/rewrite, downweight copied spans |
| `token_unlikelihood` | Discourage soft-debias words | `hard_neutralization` | Field weights + token-level unlikelihood + x2 hard-case upsampling |
| `phrase_unlikelihood` | Kill laundering phrases | `concise_hard` | Phrase-level unlikelihood + curated exemplars + target filtering + r=32 |
| `bias_weighted` | Balanced default | `concise_balanced` | Per-bias-type weighting + mild phrase penalty + 0.5x upsampling |

`bias_weighted` is the recommended default: it removes laundering without the
over-deletion tendency of the more aggressive recipes.

## Layout

```
recipes/
  config.py    # typed RecipeConfig + YAML loader + REGISTRY
  configs/     # one YAML per recipe (all hyperparameters live here)
  prompts.py   # named system prompts + build_messages(prompt_id, article)
  data.py      # load_train4, validation, curation, weighted formatting
  losses.py    # WeightedDataCollator + WeightedLossTrainer (unlikelihood)
  train.py     # single entry point (python -m recipes.train)
  eval.py      # standalone eval CLI (python -m recipes.eval)
  launch.slurm # generic SLURM launcher taking a recipe name
```

## Data

These recipes always use the `train_4` config (the newest split, with the
expanded span-level schema). The dataset also ships earlier splits — `train_1`,
`train_2`, `train_3`, and `test_set` — under the `other_splits` config; they use
an older schema and are not used here.

`train_4` is loaded automatically; you never point at a local JSON file:

```python
from datasets import load_dataset

train_4 = load_dataset("vector-institute/unbias-plus-dataset", "train_4", split="train")
```

`train.py` validates the rows, deterministically carves a 250-row held-out test
set (seed 42, never trained on) written to `<output-dir>/heldout.jsonl`, applies
the recipe's curation, and formats the rest for training.

## Environment

Set these before running (the training venv must have the `train` extra
installed: `uv sync --extra train`):

```bash
export HF_TOKEN=...            # required to pull the dataset / base model
export HF_HOME=/path/to/hf_cache   # point at storage with quota (home is often small)
export WANDB_API_KEY=...       # optional; omit or pass --no-wandb to disable
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Train

Run from the `training/` directory so `recipes` resolves as a package:

```bash
cd training

# smoke test (fast: 20 samples, 1 epoch, 4-bit)
python -m recipes.train --recipe bias_weighted \
    --train-samples 20 --epochs 1 --max-seq-length 4096 --load-in-4bit

# full run
python -m recipes.train --recipe bias_weighted

# on a cluster
sbatch recipes/launch.slurm bias_weighted
```

Outputs land in `outputs/<recipe>/` (LoRA adapter + `merged_16bit/` +
`heldout.jsonl`). List recipes with the keys in `recipes/config.py::REGISTRY`.

Useful overrides: `--output-dir`, `--base-model`, `--train-samples`, `--epochs`,
`--max-seq-length`, `--load-in-4bit`, `--no-wandb`. You can also pass a direct
YAML path instead of a registered name to `--recipe`.

## Evaluate

`eval.py` embeds the recipe's prompt and parses the model's JSON directly:

```bash
cd training

# single string / file
python -m recipes.eval --recipe bias_weighted --text "The stakes are extremely high."
python -m recipes.eval --recipe bias_weighted --file article.txt

# held-out set: per-row output + severity MAE, mean |segment delta|, laundering hits
python -m recipes.eval --recipe bias_weighted \
    --jsonl outputs/bias_weighted/heldout.jsonl --limit 20
```

By default the model is read from `outputs/<recipe>/merged_16bit`; override with
`--model-path`. Add `--load-in-4bit` for a smaller footprint or `--raw` to dump
the model's JSON verbatim.

## Add a new recipe

1. Copy a YAML in `configs/` and adjust weights / prompt / data ops.
2. Register its name in `recipes/config.py::REGISTRY` (or pass the YAML path
   directly to `--recipe`).

No Python changes are needed for a new hyperparameter combination.
