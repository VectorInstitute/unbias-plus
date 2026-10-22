"""Quick inference check — verify the merged model actually works.

Usage:
  python test_inference.py                       # default path below
  python test_inference.py outputs/smoke_test/merged_16bit
"""

import os
import sys
from typing import Any

import torch
from prompts import SFT_SYSTEM_PROMPT as SYSTEM_PROMPT
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = os.environ.get(
    "UNBIAS_MODEL_PATH",
    "outputs/vldbench_1k/merged_16bit",
)
# Two test articles — one clearly biased, one neutral.
ARTICLES = {
    "biased": (
        "The radical extremist senator pushed his dangerous agenda through "
        "the corrupt establishment, ignoring the will of real Americans."
    ),
    "neutral": (
        "The senator introduced new legislation on healthcare reform this week. "
        "The bill passed committee review with support from members of both parties "
        "and is scheduled for a floor vote next month."
    ),
}


def run(model: Any, tok: Any, label: str, article: str) -> None:
    """Run one inference sample and print raw + clean output."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Analyze the following article for bias and return the result "
                f"in the required JSON format.\n\nARTICLE:\n{article}"
            ),
        },
    ]
    # add_generation_prompt=True: leave the assistant turn OPEN for the model
    # to complete. This is the inference counterpart to training, where the
    # assistant turn is closed so the model learns to stop.
    prompt = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            temperature=None,  # silence Qwen3 sampling-default warnings
            top_p=None,
            top_k=None,
            pad_token_id=tok.eos_token_id,
        )

    new_tokens = out[0][inputs.input_ids.shape[1] :]
    print(f"\n{'=' * 60}\n  TEST: {label.upper()}\n{'=' * 60}")
    print(f"\n--- Input article ---\n{article}")
    print("\n--- Raw output (special tokens visible) ---")
    print(tok.decode(new_tokens, skip_special_tokens=False))
    print("\n--- Clean output ---")
    print(tok.decode(new_tokens, skip_special_tokens=True))


def main() -> None:
    """Load model and run inference smoke test on biased and neutral articles."""
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH

    print(f"Loading model from: {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Loaded.\n")

    for label, article in ARTICLES.items():
        run(model, tok, label, article)


if __name__ == "__main__":
    main()
