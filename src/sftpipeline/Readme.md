# Unbias: Bias Detection & Neutralization Pipeline (SFT + Evaluation)

This repository implements a full supervised fine-tuning (SFT) pipeline
for training a large language model (Qwen2.5-7B-Instruct) to:

-   Detect biased or loaded language in news articles\
-   Identify exact biased segments\
-   Assign severity labels\
-   Generate a fully neutral rewritten version\
-   Preserve factual accuracy and structure

The system includes:

1.  Dataset Generation (Gemini structured JSON generation)
2.  Supervised Fine-Tuning (QLoRA using Unsloth + TRL)
3.  Structured Inference (Outlines + Pydantic validation)
4.  Model Evaluation (LLM-as-a-Judge using GPT-4o)

------------------------------------------------------------------------

## Updating Unsloth (Without Updating Dependencies)

If you need to reinstall **Unsloth** and **Unsloth Zoo** without modifying other dependency versions (recommended for stable training environments), run or if you face any issue with xformers:

```bash
pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth
pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth_zoo
```

------------------------------------------------------------------------

# Repository Structure
``` text
├── datageneration_async.py
├── sft_training.py
├── inference_final.py
├── evaluate_models.py
└── README.md
```

------------------------------------------------------------------------

# 1️⃣ datageneration_async.py

## Dataset Construction (Gemini → Structured JSON)

### Purpose

Generates a structured supervision dataset from labeled articles.

-   Input: VLDBench_5k_60_40_balanced.csv\
-   Output: Unbias-plus-dataset.json\
-   Checkpointing: checkpoints_gpt/checkpoint.json

Uses Gemini 3 Flash Preview to produce structured JSON with strict
schema enforcement.

------------------------------------------------------------------------

## Output Schema

``` json
{
  "index": int,
  "binary_label": "biased" | "unbiased",
  "article_text": "...",
  "severity": 0 | 2 | 3 | 4,
  "bias_found": true | false,
  "biased_segments": [
    {
      "original": "...",
      "replacement": "...",
      "severity": "high|medium|low",
      "bias_type": "...",
      "reasoning": "..."
    }
  ],
  "unbiased_text": "..."
}
```

------------------------------------------------------------------------

# 2️⃣ sft_training.py

## Supervised Fine-Tuning (QLoRA)

Fine-tunes Qwen2.5-7B-Instruct using:

-   4-bit quantization
-   LoRA adapters
-   Completion-only loss
-   Structured JSON supervision

### Key Training Settings

-   LoRA Rank: 32\
-   LoRA Alpha: 64\
-   Epochs: 5\
-   Max Length: 8192\
-   Optimizer: paged_adamw_8bit

Model saved to:

/Unbias/Models/Unbias-plus-Qwen2.5

------------------------------------------------------------------------

# 3️⃣ inference_final.py

## Structured Inference

Uses:

-   Fine-tuned Qwen model
-   Outlines structured generation
-   Pydantic validation
-   Dynamic context window control

Handles truncation safely and validates substring integrity.

Output saved to:

ft_model_inference_resultsfinal.json

------------------------------------------------------------------------

# 4️⃣ evaluate_models.py

## Model Evaluation (LLM-as-a-Judge)

Compares base vs fine-tuned model using GPT-4o.

### Metrics (0--5 each):

-   Bias Reduction\
-   Faithfulness\
-   Fluency

Results saved to:

model_comparison_results_final.csv

------------------------------------------------------------------------

# Required Environment Variables

GEMINI_API_KEY=...\
OPENAI_API_KEY=...

------------------------------------------------------------------------

# How To Run

uv run datageneration_async.py\
uv run sft_training.py\
uv run inference_final.py\
uv run evaluate_models.py
