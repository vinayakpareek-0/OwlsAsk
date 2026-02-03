# -*- coding: utf-8 -*-


# 1. Environment Setup Function
def setup_environment():
    """Programmatic equivalent of !pip install"""
    packages = [
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        "xformers<0.0.28",
        "trl<0.9.0",
        "peft",
        "accelerate",
        "bitsandbytes"
    ]
    # Use sys.executable to ensure we install to the current python environment
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps"] + packages)

# Run setup only if needed (e.g., first run)
# setup_environment() 

import os
import sys
import subprocess
import torch
from packaging import version
import trl
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import SFTTrainer, DPOTrainer
from transformers import TrainingArguments, TextStreamer
from datasets import load_dataset

# 2. Path Logic for Lightning AI
# In Lightning AI Studio, /teamspace/studios/this_studio is your home
SAVE_PATH = "/teamspace/studios/this_studio/results"
os.makedirs(SAVE_PATH, exist_ok=True)

# 3. Model & Tokenizer Loading
MODEL_ID = "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_ID,
    max_seq_length = 2048,
    load_in_4bit = True,
)

# 4. SFT Setup
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
)

# 5. Data Prep (SFT)
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")

def format_sft(batch):
    texts = [f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{ins}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{res}<|eot_id|>"
             for ins, res in zip(batch["instruction"], batch["response"])]
    return {"text": texts}

dataset = dataset.map(format_sft, batched=True)

# 6. Training Configuration
sft_args = TrainingArguments(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_ratio = 0.1,
    num_train_epochs = 1,
    learning_rate = 2e-4,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 1,
    max_steps = 500, 
    optim = "adamw_8bit",
    save_strategy= "steps",
    save_steps = 250,
    output_dir = os.path.join(SAVE_PATH, "sft_outputs"),
    save_total_limit = 2,
)

trainer = SFTTrainer(
    model = model, 
    tokenizer = tokenizer, 
    train_dataset = dataset,
    dataset_text_field = "text", 
    max_seq_length = 2048, 
    args = sft_args
)

trainer.train()

# Save SFT results
model.save_pretrained("/teamspace/studios/this_studio/results/sft_adapter")
tokenizer.save_pretrained("/teamspace/studios/this_studio/results/sft_adapter")

# 7. DPO Phase
PatchDPOTrainer()

def prepare_dpo_data(example):
    if isinstance(example["chosen"], list):
        example["chosen"] = example["chosen"][-1]["content"]
    if isinstance(example["rejected"], list):
        example["rejected"] = example["rejected"][-1]["content"]
    if isinstance(example["prompt"], list):
        example["prompt"] = example["prompt"][0]["content"]
    return example

dpo_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_gen")
dpo_dataset = dpo_dataset.map(prepare_dpo_data)

dpo_args = TrainingArguments(
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,
    learning_rate = 5e-7,
    max_steps = 50,
    output_dir = os.path.join(SAVE_PATH, "dpo_outputs"),
    optim = "adamw_8bit",
)

trainer_kwargs = {
    "model": model,
    "ref_model": None,
    "args": dpo_args,
    "train_dataset": dpo_dataset,
    "beta": 0.1,
    "max_prompt_length": 512,
    "max_length": 1024,
}

# Version-safe keyword selection for DPO
if version.parse(trl.__version__) >= version.parse("0.12.0"):
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["tokenizer"] = tokenizer

dpo_trainer = DPOTrainer(**trainer_kwargs)
dpo_trainer.train()

# Final Save
model.save_pretrained(os.path.join(SAVE_PATH, "final_adapter"))