#!/usr/bin/env python
"""
MoLA + SFT (TRL) training script for Qwen-style models, using HF dataset:
FINGU-AI/Translated_Finance_100k_plus_v4_clean (sampled subset).

Usage example:

  python mola_test.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --output_dir ./mola-qwen25-finance10k \
    --target_modules o_proj,down_proj \
    --rank 8 \
    --alpha 16 \
    --top_k 2 \
    --experts 2,4,6,8 \
    --lr 2e-4 \
    --epochs 3 \
    --batch 1 \
    --grad_accum 8 \
    --max_len 1024 \
    --fp16 \
    --sample_size 10000
"""
from __future__ import annotations
import argparse
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# TRL imports
from trl import SFTTrainer, SFTConfig

# ---------------------------
# MoLA core (unchanged, compact)
# ---------------------------

class LoRAExpert(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int, alpha: float):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.A = nn.Linear(in_features, r, bias=False)
        self.B = nn.Linear(r, out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r == 0:
            return torch.zeros_like(x)
        return (self.B(self.A(x))) * (self.alpha / self.r)

class MoLARouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.proj = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (bsz, seq, hidden)
        logits = self.proj(x)  # (bsz, seq, E)
        topk_vals, topk_idx = torch.topk(logits, k=self.top_k, dim=-1)  # (..., k)
        gates = torch.softmax(topk_vals, dim=-1)
        return topk_idx, gates

class MoLAAdapter(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        hidden_size: int,
        num_experts: int,
        rank: int = 8,
        alpha: float = 16.0,
        top_k: int = 2,
    ):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        # Router sized to the input dimension to this linear
        self.router = MoLARouter(self.in_features, num_experts, top_k)
        self.experts = nn.ModuleList(
            [LoRAExpert(self.in_features, self.out_features, r=rank, alpha=alpha) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base path
        y = self.base(x)
        # router expects token-level hidden states (bsz, seq, hidden)
        topk_idx, gates = self.router(x)  # (bsz, seq, k), (bsz, seq, k)
        bsz, seq, k = gates.shape
        x2 = x.view(bsz * seq, -1)
        expert_outs = []
        for e in self.experts:
            eo = e(x2).view(bsz, seq, -1)
            expert_outs.append(eo)
        expert_outs = torch.stack(expert_outs, dim=0)  # (E, bsz, seq, out)
        gather_idx = topk_idx.permute(2, 0, 1).unsqueeze(-1).expand(-1, -1, -1, self.out_features)
        topk_outs = torch.gather(expert_outs, 0, gather_idx)  # (k, bsz, seq, out)
        gates_t = gates.permute(2, 0, 1).unsqueeze(-1)
        fused = (topk_outs * gates_t).sum(dim=0)  # (bsz, seq, out)
        return y + fused

# ---------------------------
# Instrumentation helpers
# ---------------------------

def find_target_linear_modules(model: nn.Module, target_names: List[str]) -> List[Tuple[str, nn.Module, nn.Module]]:
    hits = []
    named = dict(model.named_modules())
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for t in target_names:
                if name.endswith(t):
                    parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                    parent = named.get(parent_name, model)
                    hits.append((name, parent, module))
                    break
    return hits

def build_layerwise_allocation(num_layers: int, groups: List[int]) -> Dict[int, int]:
    assert len(groups) in (1, 2, 3, 4)
    if len(groups) == 1:
        return {i: groups[0] for i in range(num_layers)}
    boundaries = []
    if len(groups) == 4:
        boundaries = [0, num_layers//4, num_layers//2, 3*num_layers//4, num_layers]
    elif len(groups) == 3:
        boundaries = [0, num_layers//3, 2*num_layers//3, num_layers]
    else:  # 2
        boundaries = [0, num_layers//2, num_layers]

    alloc = {}
    for gi in range(len(groups)):
        start, end = boundaries[gi], boundaries[gi+1]
        for l in range(start, end):
            alloc[l] = groups[gi]
    return alloc

def attach_mola(
    model: nn.Module,
    hidden_size: int,
    num_layers: int,
    target_names: List[str],
    group_experts: List[int],
    rank: int = 8,
    alpha: float = 16.0,
    top_k: int = 2,
) -> Dict[str, int]:
    layer_alloc = build_layerwise_allocation(num_layers, group_experts)
    replaced: Dict[str, int] = {}
    hits = find_target_linear_modules(model, target_names)
    for full_name, parent, lin in hits:
        layer_id = None
        for tok in full_name.split('.'):
            if tok.isdigit():
                layer_id = int(tok)
        if layer_id is None:
            # fallback: use last group's count
            n_exp = group_experts[-1]
        else:
            n_exp = layer_alloc.get(layer_id, group_experts[-1])
        adapter = MoLAAdapter(lin, hidden_size=hidden_size, num_experts=n_exp,
                              rank=rank, alpha=alpha, top_k=top_k)
        attr = full_name.split('.')[-1]
        setattr(parent, attr, adapter)
        replaced[full_name] = n_exp
    return replaced

# ---------------------------
# Dataset & ChatML builder
# ---------------------------

def build_chatml(tokenizer, src: str, tgt: str) -> str:
    messages = [
        {"role": "system", "content": "You are translation asistant."},
        {"role": "user", "content": src},
        {"role": "assistant", "content": tgt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def load_fingu_dataset(tokenizer, split="train", sample_size=10000, num_proc=4):
    ds = load_dataset("FINGU-AI/Translated_Finance_100k_plus_v4_clean", split=split)
    if sample_size and sample_size < len(ds):
        ds = ds.shuffle(seed=42).select(range(sample_size))

    def to_chatml(example):
        src = example["input_text"]
        tgt = example["translated_text"]
        text = build_chatml(tokenizer, src, tgt)
        return {"text": text}

    return ds.map(to_chatml, remove_columns=ds.column_names, num_proc=num_proc)

# ---------------------------
# Main: TRL SFT training
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, required=True, help='HF model id or local path')
    ap.add_argument('--output_dir', type=str, required=True)
    ap.add_argument('--target_modules', type=str, default='o_proj,down_proj')
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--alpha', type=float, default=16.0)
    ap.add_argument('--top_k', type=int, default=2)
    ap.add_argument('--experts', type=str, default='2,4,6,8', help='group experts (comma list for groups)')
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--grad_accum', type=int, default=16)
    ap.add_argument('--max_len', type=int, default=1024)
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--sample_size', type=int, default=10000)
    args = ap.parse_args()

    # select dtype for loading model (user can pass --bf16 or --fp16)
    torch_dtype = None
    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # load model (dtype optional)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype, device_map="auto")

    # Freeze base weights
    for p in model.parameters():
        p.requires_grad = False

    # infer architecture
    hidden = getattr(model.config, "hidden_size", None)
    num_layers = getattr(model.config, "num_hidden_layers", None)
    if hidden is None or num_layers is None:
        raise ValueError("Could not infer model.config.hidden_size or num_hidden_layers")

    target_names = [t.strip() for t in args.target_modules.split(',') if t.strip()]
    # parse experts; allow single value like '4' or groups like '2,4,6,8'
    group_experts = [int(x) for x in args.experts.split(',')]

    replaced = attach_mola(
        model,
        hidden_size=hidden,
        num_layers=num_layers,
        target_names=target_names,
        group_experts=group_experts,
        rank=args.rank,
        alpha=args.alpha,
        top_k=args.top_k,
    )

    # enable grad only for MoLA parts
    for n, p in model.named_parameters():
        p.requires_grad = ('router' in n) or ('experts' in n)

    # Prepare dataset
    train_ds = load_fingu_dataset(tokenizer, split="train", sample_size=args.sample_size)
    def tokenize_fn(batch):
        toks = tokenizer(batch['text'], truncation=True, max_length=args.max_len, padding="max_length")
        toks['labels'] = toks['input_ids'].copy()
        return toks
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=['text'])

    # data collator for causal LM
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # TRL training args
    sft_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=1,
        save_strategy="steps",
        save_steps=500,
        fp16=args.fp16,
        # bf16=args.bf16,
        report_to="none",
        max_grad_norm=1.0,
        warmup_ratio=0.05,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        data_collator=collator,
    )

    print("\\n[MoLA] Replaced modules and expert counts:")
    for k, v in list(replaced.items())[:20]:
        print(f"  {k}: {v} experts")
    if len(replaced) > 20:
        print(f"  ... ({len(replaced)} total)")
    print("\\n[SFT training start] ...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == '__main__':
    main()
