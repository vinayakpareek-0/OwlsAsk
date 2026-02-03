import torch
import gc
import os
from unsloth import FastLanguageModel
from transformers import TextStreamer

# --- Step 0: Force Clear GPU Memory ---
def clear_gpu():
    torch.cuda.empty_cache()
    gc.collect()

clear_gpu()

# 1. Configuration
SAVE_PATH = "/teamspace/studios/this_studio/results"
FINAL_MODEL_PATH = os.path.join(SAVE_PATH, "final_adapter")

# 2. Load the Final Aligned Model
# We set device_map="cuda" to force it onto the GPU and avoid CPU offloading errors
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = FINAL_MODEL_PATH,
    max_seq_length = 1024, # Reduced from 2048 to save 1.5GB of VRAM
    load_in_4bit = True,
    device_map = "cuda", 
)

# 3. Enable 2x Faster Inference
FastLanguageModel.for_inference(model)

# 4. Terminal Chat Loop
def run_terminal_chat():
    system_prompt = "You are a professional customer support assistant. Use the history to be helpful."
    history = [] 
    MAX_HISTORY = 3 # Reduced slightly to prevent context bloat and OOM

    print("\n" + "="*50 + "\nBOT ONLINE (Memory Optimized)\n" + "="*50)

    while True:
        try:
            user_query = input("\033[1;34mYou: \033[0m").strip()
            if user_query.lower() in ["exit", "quit"]: break
            if not user_query: continue
            
            # Build the full conversation context
            messages = [{"role": "system", "content": system_prompt}]
            for turn in history:
                messages.append(turn)
            messages.append({"role": "user", "content": user_query})

            # Generate
            inputs = tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                add_generation_prompt=True, 
                return_tensors="pt"
            ).to("cuda")

            text_streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            print("\033[1;32mBot: \033[0m", end="")
            
            # Inference settings optimized for stability
            with torch.no_grad():
                output = model.generate(
                    input_ids = inputs, 
                    streamer = text_streamer, 
                    max_new_tokens = 256,
                    use_cache = True
                )
            
            # Update History
            response_text = tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)
            history.append({"role": "user", "content": user_query})
            history.append({"role": "assistant", "content": response_text})
            
            if len(history) > MAX_HISTORY * 2:
                history = history[-MAX_HISTORY * 2:]
            print()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}")
            clear_gpu()

if __name__ == "__main__":
    run_terminal_chat()