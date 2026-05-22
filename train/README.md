# UnBias-Plus Training Pipeline

Fine-tuning pipeline for bias detection and debiasing on news articles. Trains models to identify biased language, annotate segments, and produce neutral rewrites.

---

## Models on HuggingFace

> These models are currently being improved. Links will be updated when final versions are released.

| Model | 16-bit | 4-bit |
|-------|--------|-------|
| Qwen3-8B | [ahelkadyy/Qwen3-8B-UnBias-Plus-SFT-Instruct](https://huggingface.co/ahelkadyy/Qwen3-8B-UnBias-Plus-SFT-Instruct) | [4-bit](https://huggingface.co/ahelkadyy/Qwen3-8B-UnBias-Plus-SFT-Instruct-4bit) |
| Qwen3.5-4B | [ahelkadyy/Qwen3.5-4B-UnBias-Plus-SFT-Instruct](https://huggingface.co/ahelkadyy/Qwen3.5-4B-UnBias-Plus-SFT-Instruct) | [4-bit](https://huggingface.co/ahelkadyy/Qwen3.5-4B-UnBias-Plus-SFT-Instruct-4bit) |
| Llama-3.1-8B | [ahelkadyy/Llama-3.1-8B-UnBias-Plus-SFT-Instruct](https://huggingface.co/ahelkadyy/Llama-3.1-8B-UnBias-Plus-SFT-Instruct) | [4-bit](https://huggingface.co/ahelkadyy/Llama-3.1-8B-UnBias-Plus-SFT-Instruct-4bit) |
| Ministral-3-8B | [ahelkadyy/Ministral-3-8B-UnBias-Plus-SFT-Instruct](https://huggingface.co/ahelkadyy/Ministral-3-8B-UnBias-Plus-SFT-Instruct) | [4-bit](https://huggingface.co/ahelkadyy/Ministral-3-8B-UnBias-Plus-SFT-Instruct-4bit) |
| Gemma4-E4B | [ahelkadyy/Gemma4-E4B-UnBias-Plus-SFT-Instruct](https://huggingface.co/ahelkadyy/Gemma4-E4B-UnBias-Plus-SFT-Instruct) | [4-bit](https://huggingface.co/ahelkadyy/Gemma4-E4B-UnBias-Plus-SFT-Instruct-4bit) |

---

## Directory Structure

```
train/
├── train_sft.py        — main SFT training script
├── model_configs.py    — model registry (5 models + per-model configs)
├── quick_test.py       — inference sanity check on merged model
├── run_inference.py    — GPU inference pipeline (saves JSONL for judge)
├── run_judge.py        — GPT-4o-mini judge evaluation (CPU)
├── push_to_hub.py      — push merged_16bit and 4-bit models to HuggingFace
├── Models/             — trained model outputs per model key
│   └── <model_key>/
│       ├── merged_16bit/
│       └── checkpoint-*/
└── inference_results/  — inference JSONL + judge metrics + usage logs
```

---

## Environment Setup (Compute Canada A100)

Compute Canada provides precompiled wheels for PyTorch and Flash Attention. Standard PyPI packages conflict with the cluster native NCCL and CUDA modules. Follow this setup exactly.

### 1. Load modules

```bash
module purge
module load StdEnv python/3.11 cuda/12.6 nccl/2.27.7
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install packages

Verified working versions:

```
torch==2.9.1+computecanada
flash_attn==2.8.3+torch29.computecanada
transformers==5.5.0
unsloth==2026.5.2
```

```bash
uv pip install unsloth==2026.5.2 unsloth-zoo trl transformers==5.5.0 \
    datasets peft accelerate bitsandbytes python-dotenv wandb openai
```

### 4. Restore Compute Canada wheels

Installing unsloth pulls standard PyPI torch which conflicts with cluster CUDA. Restore the precompiled wheels:

```bash
uv pip uninstall torch torchvision torchaudio torchao xformers \
  nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
  nvidia-cufile-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 \
  nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 \
  nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12 nvidia-nvtx-cu12

uv pip install \
  "torch==2.9.1+computecanada" \
  "torchvision==0.24.0+computecanada" \
  "torchaudio==2.9.1+computecanada" \
  "flash_attn==2.8.3+torch29.computecanada" \
  --find-links /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic/ \
  --find-links /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3/ \
  --no-index --no-deps
```

### 5. Verify installation

```bash
python -c "import torch; print(torch.__version__)"
python -c "import flash_attn; print(flash_attn.__version__)"
python -c "import unsloth; print('ok')"
```

### 6. Required linker exports

Add these to your Slurm scripts or shell before running any training or inference:

```bash
export LD_PRELOAD=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.6.2/lib64/libnvJitLink.so.12
export LD_LIBRARY_PATH=$EBROOTCUDA/lib64:$LD_LIBRARY_PATH
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/path/to/hf_cache"
```

### 7. Environment variables (.env)

Create a `.env` file in the project root:

```
HF_TOKEN=your_huggingface_token
HF_USERNAME=your_hf_username
WANDB_API_KEY=your_wandb_key
VECTOR_API_KEY=your_vector_proxy_key
```

---

## Full Pipeline: Replicating a Run

### Step 1 — Train

Via Slurm:
```bash
MODEL_KEY=qwen3_8b sbatch train/launch_sft.sh
```

Directly:
```bash
python train/train_sft.py --model-key qwen3_8b
```

All 5 models in parallel via Slurm:
```bash
for MODEL_KEY in qwen3_8b qwen35_4b gemma4_e4b llama31_8b ministral_8b; do
    MODEL_KEY=$MODEL_KEY sbatch train/launch_sft.sh
done
```

Smoke test (10 samples, no W&B, ~2 min):
```bash
python train/train_sft.py --model-key qwen3_8b --smoke-test
```

Output saved to `train/Models/<model_key>/merged_16bit/`.

---

### Step 2 — Sanity Check

```bash
python train/quick_test.py --model-path train/Models/qwen3_8b/merged_16bit
```

---

### Step 3 — Inference

Via Slurm:
```bash
MODEL_PATH=train/Models/qwen3_8b/merged_16bit sbatch train/launch_inference.sh
```

Directly:
```bash
python train/run_inference.py \
    --model-path     train/Models/qwen3_8b/merged_16bit \
    --test-path      evaluation/babe_golden_500.json \
    --output-dir     train/inference_results \
    --max-samples    100 \
    --seed           42 \
    --load-4bit \
    --max-new-tokens 2048
```

For models with verbose output (e.g. Llama):
```bash
MAX_NEW_TOKENS=4096 MODEL_PATH=train/Models/llama31_8b/merged_16bit sbatch train/launch_inference.sh
```

Output saved to `train/inference_results/inference_<model>.jsonl`.

---

### Step 4 — Judge Evaluation

CPU only, no GPU needed. Requires `VECTOR_API_KEY` in `.env`:

```bash
python train/run_judge.py \
    --inference-file train/inference_results/inference_qwen3_8b.jsonl \
    --output-dir train/inference_results
```

Output per model:
- `metrics_<model>.json` — aggregate scores (mean/median/min/max)
- `predictions_<model>.jsonl` — per-sample scores
- `usage_<model>.json` — token usage and cost per call

---

### Step 5 — Push to HuggingFace

Via Slurm:
```bash
MODEL_KEY=qwen3_8b sbatch train/push_to_hub.sh
```

Directly:
```bash
python train/push_to_hub.py --model-key qwen3_8b
```

Requires `HF_TOKEN` and `HF_USERNAME` in `.env`.

---

## Troubleshooting

**NCCL symbol crashes (`ncclCommWindowDeregister`):** Standard PyPI torch overrode the cluster NCCL module. Purge and reinstall with `--no-index` as described in Step 4 of environment setup.

**HuggingFace 403 Forbidden:** Ensure `HF_TOKEN` in `.env` has Write permissions for your namespace.

**JSON parse failures on inference:** Model output was truncated at `max_new_tokens`. Resubmit with `MAX_NEW_TOKENS=4096`.

**Ministral loading errors:** Requires Unsloth — `AutoModelForCausalLM` does not support `Mistral3Config`. `run_inference.py` handles this automatically via fallback.
