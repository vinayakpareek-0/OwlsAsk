import os
import torch
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import DPOTrainer, DPOConfig 
from datasets import load_dataset

# 1. Path Logic
SAVE_PATH = "/teamspace/studios/this_studio/results"
SFT_ADAPTER_PATH = os.path.join(SAVE_PATH, "sft_adapter")

# 2. Load Base Model and Tokenizer
MODEL_ID = "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_ID,
    max_seq_length = 2048,
    load_in_4bit = True,
)

# 3. Attach the SFT Adapter
model.load_adapter(SFT_ADAPTER_PATH)

# 4. Memory Optimization
PatchDPOTrainer()

# 5. Data Preparation
def prepare_dpo_data(example):
    if isinstance(example["chosen"], list):
        example["chosen"] = example["chosen"][-1]["content"]
    if isinstance(example["rejected"], list):
        example["rejected"] = example["rejected"][-1]["content"]
    if isinstance(example["prompt"], list):
        example["prompt"] = example["prompt"][0]["content"]
    return example

dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_gen")
dataset = dataset.map(prepare_dpo_data)
# Use DPOConfig instead of TrainingArguments
dpo_args = DPOConfig(
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,
    learning_rate = 5e-7,
    max_steps = 50,
    output_dir = os.path.join(SAVE_PATH, "dpo_outputs"),
    optim = "adamw_8bit",
    bf16 = torch.cuda.is_bf16_supported(),
    fp16 = not torch.cuda.is_bf16_supported(),
    beta = 0.1,             # Now passed directly in the config
    max_prompt_length = 512,
    max_length = 1024,
)

# Initialize Trainer
dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None,       # Unsloth handles this
    args = dpo_args,        # Pass the new DPOConfig object
    train_dataset = dataset,
    processing_class = tokenizer,
)

# 8. Train and Save
dpo_trainer.train()
model.save_pretrained(os.path.join(SAVE_PATH, "final_adapter"))
tokenizer.save_pretrained(os.path.join(SAVE_PATH, "final_adapter"))