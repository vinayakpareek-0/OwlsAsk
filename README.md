# SupportBot: Production-Grade NLP Customer Service Agent

[![Llama 3.1](https://img.shields.io/badge/Model-Llama%203.1%208B-blue)](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)
[![Unsloth](https://img.shields.io/badge/Optimization-Unsloth-green)](https://github.com/unslothai/unsloth)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SupportBot is a high-performance customer support agent built on **Llama 3.1 8B**. Unlike basic RAG setups, this project implements a custom **Sandwich Memory** architecture for long-form dialogue and a cloud-native audit pipeline via **MongoDB Atlas**.

## 🚀 Key Features

- **Optimized Inference:** Fine-tuned using **QLoRA and Unsloth**, resulting in a 2x faster inference speed and a significantly reduced VRAM footprint.
- **Tiered Context Management (Sandwich Memory):** A custom Python engine that preserves initial user anchors (names/orders), summarizes middle-turn discussions, and maintains recent turns for high coherence.
- **Template Hallucination Recovery:** Implements a **Regex-based post-processing layer** to catch and correct "template leaks" (e.g., `{{Order Number}}`), ensuring 100% data accuracy for the end user.
- **Cloud Audit Pipeline:** Real-time session logging to **MongoDB Atlas** for enterprise-level tracking, compliance, and performance monitoring.
- **Deployment Ready:** Orchestrated via **FastAPI** and containerized for serverless deployment on **RunPod** or **Lightning AI**.

## 🏗️ Architecture

The system follows a modular "Request-Process-Clean" pipeline:

1.  **Context Engineering:** Prepends the system prompt and historical summary to the user query.
2.  **Inference:** Llama 3.1 8B generates a response based on the "Sandwich" prompt.
3.  **Refinement:** Regex filters verify the output for leaked training placeholders.
4.  **Persistence:** The final interaction is logged to the cloud and the local memory window is updated.

## 🛠️ Tech Stack

- **LLM:** Llama 3.1 8B (4-bit quantized)
- **Optimization:** Unsloth / QLoRA
- **Backend:** FastAPI / Uvicorn
- **Database:** MongoDB Atlas (NoSQL)
- **Infrastructure:** Lightning AI / Docker / RunPod
- **Frontend:** Vanilla JS / HTML5 (CSS3 Flexbox)

## 📥 Installation

1. Clone the repository:

```bash
git clone [https://github.com/Vinayak-0/SupportBot.git](https://github.com/Vinayak-0/SupportBot.git)
```

2. Install dependencies:

```bash
pip install unsloth torch fastapi pymongo[srv] uvicorn

```

3. Set your environment variables:

```bash
export MONGO_URI="your_mongodb_atlas_string"

```

## 📈 Performance Metrics

- **Latency:** ~35ms per token (on L4 GPU).
- **Post-Processing:** 100% removal of template placeholders via post-processing.

---

Developed by **Vinayak**
