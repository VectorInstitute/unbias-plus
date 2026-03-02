import json
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from accelerate import Accelerator

accelerator = Accelerator()


input_path = "/projects/aixpert/users/sindhu/Unbias/checkpoints_gpt/checkpoint.json"
output_dir = "/projects/aixpert/users/sindhu/Unbias/Models/Qwen25_Unbias_4k_5epoch"


SYSTEM_PROMPT = """
You are an expert linguist and bias detection specialist. Your job is to:
1. Identify ALL biased, loaded, or prejudiced language in the given text
2. Rewrite the text in a neutral, factual, unbiased way
3. For each biased segment, explain why it is biased and how severe it is

## Global Severity level
0 = neutral
2 = recurring biased framing
3 = strong persuasive tone
4 = inflammatory rhetoric

## BIAS TYPES TO DETECT
- **Loaded language**: Words that carry strong emotional connotations (e.g. "flooding", "invasion", "destroying")
- **Dehumanizing framing**: Language that strips dignity from groups of people
- **False generalizations**: Sweeping statements about groups ("they always", "all of them")
- **Framing bias**: Selective word choices that imply a particular viewpoint
- **Euphemisms or dysphemisms**: Softening or hardening language to manipulate perception
- **Politically charged terminology**: Labels used to provoke rather than describe
- **Sensationalism**: Exaggerated language to evoke emotional responses

## SEVERITY SCALE
- **high**: Dehumanizing, hateful, or strongly prejudiced language
- **medium**: Framing bias, loaded terms, misleading generalizations  
- **low**: Subtle word choice bias, mild framing issues

## SEGMENT RULES
- A segment is a consecutive sequence of words that form a single biased idea
- If two biased words are adjacent and part of the same biased idea, treat them as ONE segment
- If biased words are separated by neutral words, treat them as SEPARATE segments
- The "original" field must be the EXACT substring as it appears in the input text (case-sensitive, no changes)

OUTPUT SCHEMA (ALL KEYS REQUIRED):

{
  "binary_label": "biased" | "unbiased",
  "severity": 0 | 2 | 3 | 4,
  "bias_found": true | false,
  "biased_segments": [
    {
      "original": "exact substring from input text",
      "replacement": "neutral alternative phrase",
      "severity": "high" | "medium" | "low",
      "bias_type": "loaded language | dehumanizing framing | false generalizations | framing bias | euphemism/dysphemism | politically charged terminology | sensationalism",
      "reasoning": "1-2 sentence explanation"
    }
  ],
  "unbiased_text": "Full rewritten neutral article"
}

Rules:
- "original" MUST be exact case-sensitive substring.
- Only modify phrases listed in biased_segments.
- Preserve all factual information.
- If no bias: severity=0, bias_found=false, biased_segments=[]
- Return ONLY valid JSON.
""".strip()

with open(input_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

raw_data = raw_data[:4000]

print(f"Loaded {len(raw_data)} samples (first 1000 only)")

max_seq_length = 8192

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def format_sample(sample):
    completion = {
        "binary_label": sample["binary_label"],
        "severity": sample["severity"],
        "bias_found": sample["bias_found"],
        "biased_segments": sample["biased_segments"],
        "unbiased_text": sample["unbiased_text"],
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze the following article:\n\n{sample['article_text']}"},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    text += json.dumps(completion, ensure_ascii=False)

    return {"text": text}
dataset = Dataset.from_list(raw_data)
dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

print("Dataset ready:", len(dataset))
print(dataset[0]["text"])

print("Checking token lengths...")

max_tokens = 0
lengths = []

for i in range(len(dataset)):
    tokens = tokenizer(dataset[i]["text"], add_special_tokens=False)["input_ids"]
    length = len(tokens)
    lengths.append(length)
    if length > max_tokens:
        max_tokens = length

print(f"Max token length: {max_tokens}")
print(f"Average token length: {sum(lengths) / len(lengths):.2f}")
print(f"Samples > {max_seq_length} tokens: {sum(l > max_seq_length for l in lengths)}")

print("Filtering samples exceeding max_seq_length...")

dataset = dataset.filter(
    lambda x: len(tokenizer(x["text"], add_special_tokens=False)["input_ids"]) <= max_seq_length
)

print("Dataset size after filtering:", len(dataset))

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=5,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=20,
    save_total_limit=3,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    max_grad_norm=1.0,
    report_to="none",
    seed=42,
    dataset_text_field="text",
    completion_only_loss=True,
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
    dataset_num_proc=4,
    max_length=8192,
    torch_compile=False,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    packing=False,
    args=training_args,
)

print("Starting Qwen2.5-7B Completion-Only Training...")
trainer.train()

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model saved to {output_dir}")