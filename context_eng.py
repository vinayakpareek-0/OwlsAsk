import torch
from transformers import AutoTokenizer

class ContextManager:
    def __init__(self, model_id, max_tokens=2048, anchor_turns=2, recent_turns=4):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.max_tokens = max_tokens
        self.anchor_turns = anchor_turns * 2  # Convert turns to messages (User+Bot)
        self.recent_turns = recent_turns * 2
        
        self.system_prompt = "You are a professional customer support assistant."
        self.anchor_history = []  # The critical first messages
        self.middle_summary = ""  # The compressed middle
        self.recent_history = []  # The sliding window
        self.full_history_for_audit = [] # For your company database

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text))

    def get_working_context(self):
        """Builds the 'Sandwich' prompt for the LLM."""
        context = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{self.system_prompt}"
        
        # Add Initial Anchor
        if self.anchor_history:
            context += "\n\n[Initial Context]:\n"
            for msg in self.anchor_history:
                context += f"{msg['role']}: {msg['content']}\n"

        # Add Compressed Middle
        if self.middle_summary:
            context += f"\n\n[Summary of previous discussion]:\n{self.middle_summary}"

        # Add Recent Working Memory
        context += "\n\n[Recent Conversation]:\n"
        for msg in self.recent_history:
            context += f"{msg['role']}: {msg['content']}\n"
            
        return context

    def compress_middle(self, model):
        """Moves messages from Recent to Middle Summary."""
        # If history gets too long, we take the OLDEST of the 'recent' 
        # and fold them into the summary.
        to_summarize = self.recent_history[:-self.recent_turns]
        self.recent_history = self.recent_history[-self.recent_turns:]

        text_to_compress = "\n".join([f"{m['role']}: {m['content']}" for m in to_summarize])
        
        prompt = f"Update the following summary with these new details. Keep it brief:\nExisting Summary: {self.middle_summary}\nNew details: {text_to_compress}"
        
        # Summarization Inference
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            summary_ids = model.generate(**inputs, max_new_tokens=150)
        
        self.middle_summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print("--- Middle Context Compressed ---")

    def add_message(self, role, content, model=None):
        msg = {"role": role, "content": content}
        self.full_history_for_audit.append(msg)

        # 1. Fill Anchor first (the very start of the conversation)
        if len(self.anchor_history) < self.anchor_turns:
            self.anchor_history.append(msg)
        else:
            # 2. Otherwise, add to Recent
            self.recent_history.append(msg)

        # 3. Check for compression trigger
        # If 'recent' exceeds our threshold, compress the middle
        if len(self.recent_history) > self.recent_turns + 2 and model:
            self.compress_middle(model)