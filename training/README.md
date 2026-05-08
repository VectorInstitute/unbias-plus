# Training scripts

Code used to fine-tune the Qwen3 model that powers the `unbias-plus` demo.
These are **standalone scripts**, not part of the installable `unbias_plus`
Python package. They are kept here for reproducibility and reference.

The trained model itself is **not** in this repo — see "Model artifacts" below.

## Files

| File | Purpose |
|---|---|
| `train_sft.py` | SFT fine-tune of Qwen3-8B with completion-only loss, exports merged bf16 weights |
| `train_grpo.py` | GRPO post-training pass (DDP across 4 GPUs) |
| `sanity_check.py` | Smoke-test inference against a trained checkpoint on a single article |

## Environment

- A machine with NVIDIA A100s (or comparable) and CUDA 12.4
- Python 3.11 (matches `.python-version`)
- The `[train]` optional extra of this project supplies `peft`, `trl`,
  `unsloth`, `unsloth-zoo`, `flash-attn`. Install with:
  ```bash
  uv sync --extra train
  source .venv/bin/activate
  ```

## Running

### SFT (single A100)

```bash
python training/train_sft.py \
    --input-path /path/to/training_data.json \
    --output-dir /path/to/output/qwen3_sft
```

Other flags: `--base-model`, `--max-seq-length`, `--train-samples`, `--seed`.
See `python training/train_sft.py --help`.

For reasonable resource sizing: 1× A100 80 GB, ~8 hours for the default
~5 K-sample run at `max_seq_length=8192`.

### GRPO (4× A100, DDP)

```bash
accelerate launch \
    --num_processes=4 \
    --mixed_precision=bf16 \
    --main_process_port=29500 \
    training/train_grpo.py \
        --input-path /path/to/training_data.json \
        --output-dir /path/to/output/qwen3_grpo
```

Useful environment variables:
```bash
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Do not set CUDA_VISIBLE_DEVICES — let accelerate manage all 4 GPUs.
```

The Python entrypoint exposes the full hyperparameter surface
(`--load-in-4bit/--no-load-in-4bit`, `--max-completion-length`, `--lora-rank`,
`--learning-rate`, etc.) — see `python training/train_grpo.py --help`.
GRPO at default settings: 4× A100 80 GB, ~72 hours.

### Sanity check

Smoke-test a trained checkpoint against a single article in a text file:

```bash
python training/sanity_check.py \
    --model-path /path/to/output/qwen3_sft/merged_16bit \
    --article-file my_article.txt
```

`--thinking` / `--no-thinking` toggles Qwen's `<think>` block at inference
(default `--no-thinking`, matching the SFT training-time setting).

## Model artifacts

Trained checkpoints live outside the repo:

- SFT merged 16-bit: produced under `OUTPUT_DIR/merged_16bit/` after a successful
  `train_sft.py` run.
- (Add HuggingFace Hub repo here once published.)

The repo's `.gitignore` excludes `Models/`, `*.safetensors`, `unsloth_compiled_cache/`,
etc., so running these scripts in-place will not accidentally commit weights.
