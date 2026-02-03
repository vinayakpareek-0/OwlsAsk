import torch
import re
import uuid
import os
from unsloth import FastLanguageModel
from transformers import TextStreamer
from context_eng import ContextManager
from audit_logger import AuditLogger

# --- 1. CONFIGURATION ---
MODEL_ID = "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit"
ADAPTER_PATH = "/teamspace/studios/this_studio/results/final_adapter"
ATLAS_URI = "mongodb+srv://name:password@cluster0.tooqapc.mongodb.net/?appName=Cluster0"

# Metadata for Regex Cleaner
USER_DATA = {
    "name": "Vinayak",
    "order": "#32323"
}

# --- 2. INITIALIZATION ---
print("Loading model and adapters...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_PATH,
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

ctx = ContextManager(model_id=MODEL_ID, anchor_turns=2, recent_turns=4)
logger = AuditLogger(atlas_uri=ATLAS_URI)

# --- 3. UTILITY FUNCTIONS ---
def clean_placeholders(text, data):
    """Hard-swaps any leaked dataset templates with real user data."""
    replacements = {
        r"\{\{Order Number\}\}": data.get("order", "your order"),
        r"\{\{Customer Name\}\}": data.get("name", "Customer"),
        r"\{\{.*\}\}": "the details" 
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    return text

# --- 4. MAIN CHAT LOOP ---
def run_production_chat():
    session_id = str(uuid.uuid4())
    user_id = "vinay"
    
    # Powerful System Instruction to override template bias
    system_instruction = f"""You are a professional customer support assistant for Ecommerce's Store.
RULES:
1. Use the REAL name ({USER_DATA['name']}) and order number ({USER_DATA['order']}).
2. NEVER use double curly braces {{{{ }}}} or placeholder text.
3. If the user mentions an issue, acknowledge it specifically.
"""

    print("\n" + "="*50)
    print(f"SUPPORT BOT ONLINE | Session: {session_id}")
    print("="*50 + "\n")

    while True:
        user_query = input("\033[1;34mYou: \033[0m").strip()
        if user_query.lower() in ["exit", "quit"]: break
        if not user_query: continue

        # Context Engineering
        base_prompt = ctx.get_working_context()
        
        full_prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_instruction}<|eot_id|>"
            f"{base_prompt}"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user_query}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
        
        # We don't use the streamer here because we need to clean the text BEFORE showing it
        with torch.no_grad():
            output = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=512,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 1. Decode raw response
        raw_response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # 2. CLEAN THE TEXT (The Safety Net)
        cleaned_response = clean_placeholders(raw_response, USER_DATA)
        
        # 3. Display cleaned response
        print(f"\033[1;32mBot: \033[0m{cleaned_response}")
        
        # 4. Update memory and logs with CLEANED text
        ctx.add_message("user", user_query)
        ctx.add_message("assistant", cleaned_response, model=model)
        
        logger.log_interaction(
            session_id=session_id,
            user_id=user_id,
            user_query=user_query,
            bot_response=cleaned_response,
            metadata={"cleaner_applied": True}
        )
        print()

if __name__ == "__main__":
    run_production_chat()
