import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
from unsloth import FastLanguageModel
from context_eng import ContextManager
from audit_logger import AuditLogger
import re
from fastapi.middleware.cors import CORSMiddleware




# --- CONFIGURATION ---
ADAPTER_PATH = "/teamspace/studios/this_studio/results/final_adapter"
ATLAS_URI = "mongodb+srv://vinay:Lpp38xBlQ2hTrWiH@cluster0.tooqapc.mongodb.net/?appName=Cluster0"
USER_DATA = {"name": "Vinayak", "order": "#32323"}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- INITIALIZE AI ENGINE ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_PATH,
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

ctx = ContextManager(model_id="unsloth/meta-llama-3.1-8b-instruct-bnb-4bit")
logger = AuditLogger(atlas_uri=ATLAS_URI)

# --- UTILS ---
class ChatRequest(BaseModel):
    message: str
    user_id: str = "vinay"

def clean_placeholders(text, data):
    replacements = {
        r"\{\{Order Number\}\}": data.get("order", "your order"),
        r"\{\{Customer Name\}\}": data.get("name", "Customer"),
        r"\{\{.*\}\}": "the details" 
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    return text

# --- API ENDPOINT ---
@app.get("/")
async def root():
    return {"status": "Online", "message": "Owls Ask API is running. Use /docs for testing."}
    

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # 1. Prepare Prompt
        base_prompt = ctx.get_working_context()
        full_prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"You are a professional support assistant. Use Name: {USER_DATA['name']} and Order: {USER_DATA['order']}.<|eot_id|>"
            f"{base_prompt}"
            f"<|start_header_id|>user<|end_header_id|>\n\n{request.message}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        # 2. Generate
        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=256)
        
        # 3. Clean
        raw_text = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        cleaned_text = clean_placeholders(raw_text, USER_DATA)

        # 4. Update Context & Log
        ctx.add_message("user", request.message)
        ctx.add_message("assistant", cleaned_text, model=model)
        logger.log_interaction(str(uuid.uuid4()), request.user_id, request.message, cleaned_text)

        return {"response": cleaned_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))