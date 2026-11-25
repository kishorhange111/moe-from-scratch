"""Byte-level GPT whose feed-forward blocks are dense MLPs or MoE layers.

Parameter accounting matters when comparing dense against MoE, so ``param_counts`` reports
both the **total** parameters and the **active** parameters per token (router plus the k
experts a token uses, plus any shared experts). The experiments compare models at equal
active parameters, i.e. roughly equal training FLOPs per token.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .layers import MoE


@dataclass
class Config:
    vocab: int = 256
    ctx: int = 256
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    # MoE (n_experts = 0 means dense)
    n_experts: int = 0
    k: int = 2
    capacity_factor: float = 1.25
    routing: str = "token_choice"
    n_shared: int = 0
    moe_every: int = 1  # MoE in every n-th block, dense elsewhere


class Block(nn.Module):
    def __init__(self, c: Config, moe: bool) -> None:
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model)
        self.proj = nn.Linear(c.d_model, c.d_model)
        self.h = c.n_heads
        if moe:
            self.ff = MoE(c.d_model, c.d_ff, c.n_experts, c.k, c.capacity_factor, c.routing,
                          c.n_shared)  # fmt: skip
        else:
            self.ff = nn.Sequential(
                nn.Linear(c.d_model, c.d_ff), nn.GELU(), nn.Linear(c.d_ff, c.d_model)
            )

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).unbind(2)
        a = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        h = self.ln2(x)
        if isinstance(self.ff, MoE):
            y, stats = self.ff(h)
            return x + y, stats
        return x + self.ff(h), None


class GPT(nn.Module):
    def __init__(self, c: Config) -> None:
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d_model)
        self.pos = nn.Embedding(c.ctx, c.d_model)
        self.blocks = nn.ModuleList(
            Block(c, c.n_experts > 0 and i % c.moe_every == c.moe_every - 1)
            for i in range(c.n_layers)
        )
        self.ln = nn.LayerNorm(c.d_model)
        for m in self.modules():  # GPT-2 style init; expert weights are initialised in MoE
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, idx):
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        stats = []
        for b in self.blocks:
            x, s = b(x)
            if s is not None:
                stats.append(s)
        return self.ln(x) @ self.emb.weight.t(), stats


def param_counts(model: GPT) -> dict:
    c = model.c
    total = sum(p.numel() for p in model.parameters())
    emb = model.emb.weight.numel() + model.pos.weight.numel()
    active = total
    for b in model.blocks:
        if isinstance(b.ff, MoE):
            per_expert = b.ff.w1[0].numel() + b.ff.w2[0].numel()
            active -= per_expert * (b.ff.E - (c.k if c.routing == "token_choice" else 0))
            if c.routing == "expert_choice":  # on average each token sees E*C/T = k*cf experts
                active -= per_expert * b.ff.E
                active += per_expert * c.k * c.capacity_factor
    return {"total": total, "active": int(active), "non_embedding_active": int(active - emb)}
