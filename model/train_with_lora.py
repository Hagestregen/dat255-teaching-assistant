"""
train_with_lora.py — QLoRA fine-tuning of Qwen2.5-7B-Instruct on the DAT255 dataset
Uses the same JSONL data as train.py but with ChatML format for the instruct model.
"""
import json
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset

# MODEL_ID  = "Qwen/Qwen2.5-7B-Instruct"
MODEL_ID  = "Qwen/Qwen2.5-3B-Instruct"
# MODEL_ID  = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = "data/train.jsonl"
# OUT_DIR   = "checkpoints/qwen_7b_lora"
OUT_DIR   = "checkpoints/qwen_3b_lora"

# ── 1. Load model in 4-bit (QLoRA) ──────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# ── 2. Attach LoRA adapters ──────────────────────────────────────────────────
lora_config = LoraConfig(
    r=16,                  # rank — higher = more capacity, more VRAM
    lora_alpha=32,         # scaling factor (typically 2× r)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 35,127,296 || all params: 7,650,000,000 || 0.46%

# ── 3. Convert your JSONL to ChatML format ───────────────────────────────────
# Instruct models MUST be fine-tuned in their native chat format.
# Qwen2.5 uses ChatML: <|im_start|>role\ncontent<|im_end|>
TASK_TO_SYSTEM = {
    "explanation": "You are a teaching assistant. Explain the concept clearly.",
    "quiz":        "You are a teaching assistant. Generate a multiple-choice quiz question as JSON.",
    "review":      "You are a teaching assistant. Evaluate the student's answer constructively.",
    "flashcard":   "You are a teaching assistant. Create a concise flashcard.",
}

def jsonl_to_chatml(jsonl_path: str) -> list[dict]:
    examples = []
    with open(jsonl_path) as f:
        for line in f:
            ex = json.loads(line)
            task   = ex.get("type", "explanation")
            system = TASK_TO_SYSTEM.get(task, TASK_TO_SYSTEM["explanation"])
            # Re-use the existing text field but strip the task prefix token
            # since ChatML handles role separation differently
            text = ex["text"]
            for prefix in ["<|explain|>", "<|quiz|>", "<|review|>", "<|flashcard|>"]:
                text = text.replace(prefix, "").strip()
            text = text.rstrip("<|endoftext|>").strip()

            chatml = (
                f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{text.split('Response:')[0].strip()}<|im_end|>\n"
                f"<|im_start|>assistant\n"
                + (text.split("Response:")[-1].strip() if "Response:" in text else text)
                + "<|im_end|>"
            )
            examples.append({"text": chatml})
    return examples

raw = jsonl_to_chatml(DATA_PATH)
ds  = Dataset.from_list(raw).train_test_split(test_size=0.1, seed=42)

def tokenize(batch):
    return tokenizer(
        batch["text"], truncation=True, max_length=512, padding=False
    )

ds = ds.map(tokenize, batched=True, remove_columns=["text"])

# ── 4. Train ─────────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,     # effective batch = 16
    learning_rate=2e-4,                # higher than full fine-tune is fine for LoRA
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=400,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="wandb",                 # same W&B project as your custom model
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()
model.save_pretrained(OUT_DIR)