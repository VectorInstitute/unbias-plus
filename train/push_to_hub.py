"""Push a locally saved merged_16bit model to HuggingFace in both 16-bit and 4-bit.

Usage:
    python push_to_hub.py --model-key qwen3_8b
    python push_to_hub.py --model-key llama31_8b
"""

import argparse
import os
import sys


sys.path.insert(0, os.path.dirname(__file__))

import torch
from dotenv import load_dotenv
from model_configs import MODEL_REGISTRY
from unsloth import FastLanguageModel


load_dotenv()


OUTPUT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models")


def main() -> None:
    """Parse arguments and push models to the Hugging Face Hub."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-key", required=True, choices=list(MODEL_REGISTRY.keys())
    )
    args = parser.parse_args()

    config = MODEL_REGISTRY[args.model_key]
    merged_path = os.path.join(OUTPUT_BASE, config.key, "merged_16bit")
    hf_token = os.environ.get("HF_TOKEN", "")
    repo_4bit = config.hf_repo_id + "-4bit"

    if not os.path.isdir(merged_path):
        raise FileNotFoundError(f"merged_16bit not found at: {merged_path}")

    # --- Push 16-bit ---
    print(f"\n  Loading merged 16-bit from: {merged_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        merged_path,
        max_seq_length=config.max_seq_length,
        load_in_4bit=False,
        dtype=torch.bfloat16,
    )
    print(f"  Pushing 16-bit to: {config.hf_repo_id}")
    model.push_to_hub(config.hf_repo_id, token=hf_token)
    tokenizer.push_to_hub(config.hf_repo_id, token=hf_token)
    print(f"  16-bit live: https://huggingface.co/{config.hf_repo_id}")

    # --- Free VRAM, reload in 4-bit, push ---
    del model
    torch.cuda.empty_cache()

    print("\n  Loading merged model in 4-bit...")
    model_4bit, tokenizer_4bit = FastLanguageModel.from_pretrained(
        merged_path,
        max_seq_length=config.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    print(f"  Pushing 4-bit to: {repo_4bit}")
    model_4bit.push_to_hub(repo_4bit, token=hf_token)
    tokenizer_4bit.push_to_hub(repo_4bit, token=hf_token)
    print(f"  4-bit live: https://huggingface.co/{repo_4bit}")


if __name__ == "__main__":
    main()
