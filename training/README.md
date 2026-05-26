# Training

Standalone scripts for fine-tuning and evaluating the UnBias-Plus model.
Not part of the installable `unbias_plus` package (`src/unbias_plus/`).

## Scripts

| File | Purpose |
|------|---------|
| `prompts.py` | Shared system prompts (imported by the other scripts) |
| `train.py` | SFT fine-tuning (main entry, 1× A100) |
| `clean_dataset.py` | Clean / prepare the training JSON |
| `eval_judge.py` | Run the model on a test set + LLM-judge metrics |
| `quick_test.py` | Fast local inference smoke test |
| `smoke_test_inference.py` | Inference check via the merged model / adapter |

## Environment

### Standard setup

```bash
# create + activate the training venv
python -m venv .venv-train
source .venv-train/bin/activate

# install deps (torch, transformers, trl, unsloth, ...)
uv sync --extra train      # or: pip install -r requirements.txt

# HF cache (home quota is small — point it at the project space)
export HF_HOME=/projects/aixpert/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Compute Canada A100 (flash attention)

Compute Canada provides precompiled wheels for PyTorch and Flash Attention. Standard PyPI packages conflict with the cluster's native NCCL and CUDA modules. Follow this setup exactly.

**1. Load modules**

```bash
module purge
module load StdEnv python/3.11 cuda/12.6 nccl/2.27.7
```

**2. Create virtual environment**

```bash
python -m venv .venv-train
source .venv-train/bin/activate
```

**3. Install packages**

```bash
uv pip install unsloth==2026.5.2 unsloth-zoo trl transformers==5.5.0 \
    datasets peft accelerate bitsandbytes python-dotenv wandb openai
```

**4. Restore Compute Canada wheels**

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

**5. Verify installation**

```bash
python -c "import torch; print(torch.__version__)"
python -c "import flash_attn; print(flash_attn.__version__)"
python -c "import unsloth; print('ok')"
```

**6. Required linker exports**

Add these to your Slurm scripts or shell before running any training or inference:

```bash
export LD_PRELOAD=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.6.2/lib64/libnvJitLink.so.12
export LD_LIBRARY_PATH=$EBROOTCUDA/lib64:$LD_LIBRARY_PATH
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/path/to/hf_cache"
```

**7. Environment variables (.env)**

Create a `.env` file in the project root:

```
HF_TOKEN=your_huggingface_token
HF_USERNAME=your_hf_username
WANDB_API_KEY=your_wandb_key
VECTOR_API_KEY=your_vector_proxy_key
```

## Data

Source: https://huggingface.co/datasets/vector-institute/unbias-plus-dataset

```python
from datasets import load_dataset

train = load_dataset("vector-institute/unbias-plus-dataset", name="unbias-plus_train", split="train")
test  = load_dataset("vector-institute/unbias-plus-dataset", name="unbias-plus_test",  split="test")
```

Save a split to local JSON, then pass it to `train.py` / `eval_judge.py` via
`--input-path` / `--test-path`. The data folder itself is gitignored.

## Train

```bash
# smoke test (fast — 20 samples, 1 epoch)
python train.py \
  --input-path data/train.json \
  --output-dir outputs/smoke_test \
  --train-samples 20 --epochs 1 --max-seq-length 4096 --load-in-4bit

# full run
python train.py \
  --input-path data/train.json \
  --output-dir outputs/run5 --epochs 5
```

Output lands in `outputs/<run>/merged_16bit/` (gitignored).

## Inference check

```bash
# defaults to outputs/vldbench_1k/merged_16bit; override with the env var or an arg
UNBIAS_MODEL_PATH=outputs/run5/merged_16bit python quick_test.py
# or
python quick_test.py outputs/run5/merged_16bit
```

## Eval

```bash
python eval_judge.py \
  --model-path outputs/run5/merged_16bit \
  --test-path data/test.json \
  --output-dir eval_results \
  --env-path /path/to/.env \
  --load-4bit
```
