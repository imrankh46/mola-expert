# MoLA: Mixture of LoRA Experts

Implementation of **MoLA (Mixture of LoRA Experts)** for efficient LLM fine-tuning.  
Extends LoRA by attaching multiple experts per layer and routing tokens dynamically.  
Built on **Hugging Face Transformers** + **TRL SFTTrainer**.

---

## 🚀 Features
- Multiple LoRA experts per layer (`--experts 2,4,6,8` or fixed).
- Token routing with configurable `--top_k`.
- Supports **fp16/bf16** training (Colab/GPUs).
- ChatML dataset formatting for instruction tuning.
- Example: **Qwen2.5-Instruct** on financial translation dataset.

---

## 📦 Installation
```bash
git clone https://github.com/yourname/mola
cd mola
pip install -r requirements.txt
```
## ⚡ Training Example
```bash
python mola_test.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir ./mola-qwen25-sft \
  --target_modules o_proj,down_proj \
  --rank 32 \
  --alpha 64 \
  --top_k 2 \
  --experts 2,4,6,8 \
  --lr 1e-4 \
  --epochs 2 \
  --batch 2 \
  --grad_accum 8 \
  --max_len 1024 \
  --fp16 \
  --sample_size 10000
```

## 📊 Dataset

Uses FINGU-AI/Translated_Finance_100k_plus_v4_clean
with ChatML instruction formatting for supervised fine-tuning.
