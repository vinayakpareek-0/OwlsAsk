import torch
from unsloth import FastLanguageModel
from context_eng import ContextManager

# 1. Setup
MODEL_ID = "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit"
# Load model once for both chat and summarization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# Initialize Manager: Short window (e.g., 6 messages) to trigger compression quickly
ctx = ContextManager(model_id=MODEL_ID, anchor_turns=1, recent_turns=2)

def simulate_chat(query):
    # Prepare current context
    prompt = ctx.get_working_context() + f"\nUSER: {query}\nASSISTANT:"
    
    # Tokenize and Generate
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    
    # Update Context Manager
    ctx.add_message("user", query)
    ctx.add_message("assistant", response, model=model)
    
    print(f"\n[USER]: {query}")
    print(f"[BOT]: {response}")

# --- THE TEST SEQUENCE ---

print("--- PHASE 1: THE ANCHOR ---")
simulate_chat("Hi, my name is Alex and my order number is #99988. I received the wrong item.")

print("\n--- PHASE 2: THE FILLER (Triggering Compression) ---")
filler_queries = [
    "What's the weather like?",
    "Do you like robots?",
    "What is 2+2?",
    "Tell me a joke.",
    "Is the moon made of cheese?",
    "Can you sing?"
]
for q in filler_queries:
    simulate_chat(q)

print("\n--- PHASE 3: THE PROBE ---")
# This tests if Anchor (Name/Order) and Middle (The fact that it was a 'wrong item') survived
simulate_chat("Can you confirm my name and exactly why I contacted you originally?")

# Final Debug: See what the internal 'Sandwich' looks like
print("\n" + "="*50)
print("FINAL INTERNAL CONTEXT STRUCTURE:")
print(ctx.get_working_context())