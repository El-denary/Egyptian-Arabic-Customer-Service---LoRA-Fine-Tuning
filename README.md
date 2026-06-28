<div align="center">

# 🤖 Qwen3 Egyptian Arabic Customer Service — LoRA Fine-Tuning

**Fine-tuning Qwen3-1.7B on Egyptian Arabic customer service conversations using LoRA.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-FF6B35?style=flat)](https://github.com/huggingface/peft)
[![HuggingFace Model](https://img.shields.io/badge/🤗%20Model-Eldenary%2Fqwen--Customer--Service--lora-FFD21E?style=flat)](https://huggingface.co/Eldenary/qwen-Customer-Service-lora)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

*Teaching a language model to speak Egyptian Arabic — naturally, contextually, helpfully.*

</div>

---

## What Is This?

This project fine-tunes **Qwen/Qwen3-1.7B** on a custom dataset of **257 Egyptian Arabic customer service conversations** using **LoRA (Low-Rank Adaptation)**. The result is a lightweight, efficient model that can handle real-world customer inquiries in Egyptian Arabic dialect — resolving order issues, answering product questions, and providing natural, context-aware support responses.

---

## 🤗 Model on HuggingFace Hub

The fine-tuned LoRA adapter is publicly available on HuggingFace:

**[Eldenary/qwen-Customer-Service-lora](https://huggingface.co/Eldenary/qwen-Customer-Service-lora)**

You can load it directly without cloning this repo or running the notebook:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model = "Qwen/Qwen3-1.7B"
adapter    = "Eldenary/qwen-Customer-Service-lora"

model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, adapter)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(adapter)
```

---

## Dataset

`CustomerData.json` contains **257 multi-turn conversations** in Egyptian Arabic dialect, formatted as chat templates:

```json
{
  "conversations": [
    { "role": "user",      "content": "السلام عليكم، عندي مشكلة في الأوردر." },
    { "role": "assistant", "content": "وعليكم السلام، تحت أمرك. ممكن تقولي رقم الأوردر؟" }
  ]
}
```

The conversations cover common customer service scenarios including order tracking, delivery issues, product inquiries, and complaint resolution — all in colloquial Egyptian Arabic.

---

## How It Works

```
CustomerData.json
      │
      ▼
Chat Template Formatting  (Qwen3 chat format)
      │
      ▼
Tokenization + Padding    (max_length = 512)
      │
      ▼
LoRA Adapter Training     (r=8, alpha=16, 5 epochs)
      │
      ▼
Fine-tuned Model          (saved locally + HuggingFace Hub)
      │
      ▼
Inference                 (Egyptian Arabic customer queries)
```

---

## LoRA Configuration

| Parameter | Value | Notes |
|---|---|---|
| Base model | `Qwen/Qwen3-1.7B` | 1.7B parameter causal LM |
| LoRA rank (`r`) | `8` | Adapter matrix rank |
| LoRA alpha | `16` | Scaling factor |
| Dropout | `0.05` | Regularization |
| Task type | `CAUSAL_LM` | Autoregressive generation |
| Bias | `none` | No bias terms in adapters |

---

## Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 5 |
| Batch size | 2 (per device) |
| Gradient accumulation | 2 steps (effective batch = 4) |
| Learning rate | 2e-4 |
| Precision | `fp16` mixed precision |
| Max sequence length | 512 tokens |
| Checkpointing | Every 50 steps, keep last 2 |

---

## Project Structure

```
.
├── Fine_tuning.ipynb    # Full training pipeline — load, preprocess, train, infer
├── CustomerData.json    # 257 Egyptian Arabic customer service conversations
└── README.md
```

---

## Quickstart

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended — fp16 training)
- HuggingFace account (for model download + optional Hub push)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies

```bash
pip install transformers datasets peft torch accelerate huggingface_hub
```

### 3. Run the notebook

Open `Fine_tuning.ipynb` and run all cells in order:

| Step | What it does |
|---|---|
| **1. Load & Preprocess** | Loads `CustomerData.json`, applies Qwen3 chat template, tokenizes |
| **2. LoRA Config** | Wraps the base model with LoRA adapters via PEFT |
| **3. Training** | Trains for 5 epochs with the HuggingFace `Trainer` |
| **4. Save** | Saves adapter weights and tokenizer locally and pushes to `Eldenary/qwen-Customer-Service-lora` on the Hub |
| **5. Inference** | Reloads the merged model and runs a sample query |

---

## Inference Example

You can run inference by loading the adapter directly from the Hub — no local training needed:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model = "Qwen/Qwen3-1.7B"
adapter    = "Eldenary/qwen-Customer-Service-lora"

model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, adapter)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(adapter)

prompt = "الأوردر لسه موصلش."

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Base Model | Qwen/Qwen3-1.7B |
| Fine-tuning Method | LoRA via PEFT |
| Training Framework | HuggingFace Transformers + Trainer |
| Dataset Format | HuggingFace `datasets.Dataset` |
| Precision | `fp16` mixed precision |
| Model Hub | [Eldenary/qwen-Customer-Service-lora](https://huggingface.co/Eldenary/qwen-Customer-Service-lora) |

---

## Design Decisions

**Why LoRA instead of full fine-tuning?**  
Full fine-tuning a 1.7B model requires significant GPU memory and compute. LoRA injects small trainable adapter matrices (rank 8) into the attention layers, reducing trainable parameters by ~99% while preserving most of the base model's general capabilities. The result is fast, memory-efficient training that still achieves strong task-specific performance.

**Why Qwen3-1.7B?**  
Qwen3-1.7B is a compact but capable multilingual model with strong Arabic language understanding out of the box. Its small size makes it practical for LoRA fine-tuning on a single GPU, and its chat template format aligns naturally with the conversation structure of the dataset.

**Why fp16 mixed precision?**  
fp16 halves GPU memory usage versus fp32, enabling larger effective batch sizes on consumer hardware. Combined with gradient accumulation (2 steps), this gives a stable, memory-efficient training loop.

---

## License

MIT
