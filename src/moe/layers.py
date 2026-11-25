"""Mixture-of-Experts feed-forward layer, written out in full.

Routing (``routing=``):

- ``"token_choice"``: each token picks its top-k experts (Shazeer et al. 2017; Switch for
  k=1, GShard/Mixtral for k=2). Each expert has a fixed capacity
  ``C = ceil(capacity_factor * T * k / E)`` slots. Assignments beyond capacity are
  **dropped** (the token skips that expert; the residual path still carries it), in priority
  order: all first choices before any second choices, as in GShard.
- ``"expert_choice"``: each expert picks its top-C tokens by router score (Zhou et al. 2022).
  Load is perfectly balanced by construction, but a token may get zero experts or several.

Experts are stored stacked, ``w1: (E, d, f)`` and ``w2: (E, f, d)``, and run as one batched
matmul over a dispatch buffer of shape ``(E, C, d)``. That's the layout expert parallelism
shards: each device owns a slice of E, and the dispatch/combine steps become all-to-alls.

Auxiliary losses:

- load balancing (Switch Transformer, eq. 4): ``E * sum_i f_i * P_i`` where ``f_i`` is the
  fraction of routed assignments that went to expert i and ``P_i`` its mean router
  probability. It equals 1 at perfect balance and grows as routing collapses.
- router z-loss (ST-MoE): ``mean(logsumexp(logits)^2)``, which keeps logits small for stability.

Optional ``n_shared`` dense experts see every token (DeepSeekMoE's shared experts).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class RouterStats:
    aux_loss: torch.Tensor
    z_loss: torch.Tensor
    load: torch.Tensor  # (E,) fraction of kept assignments per expert
    dropped_frac: float  # fraction of token->expert assignments dropped (token choice)
    unrouted_frac: float  # fraction of tokens that reached no expert
    extra: dict = field(default_factory=dict)


class MoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int,
        k: int = 2,
        capacity_factor: float = 1.25,
        routing: str = "token_choice",
        n_shared: int = 0,
        renormalize: bool = True,
    ) -> None:
        super().__init__()
        if routing not in ("token_choice", "expert_choice"):
            raise ValueError(routing)
        if not 1 <= k <= n_experts:
            raise ValueError("need 1 <= k <= n_experts")
        self.E, self.k, self.cf, self.routing, self.renorm = (
            n_experts, k, capacity_factor, routing, renormalize,
        )  # fmt: skip
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.w1 = nn.Parameter(torch.empty(n_experts, d_model, d_ff))
        self.w2 = nn.Parameter(torch.empty(n_experts, d_ff, d_model))
        nn.init.normal_(self.w1, std=d_model**-0.5)
        nn.init.normal_(self.w2, std=d_ff**-0.5)
        nn.init.normal_(self.router.weight, std=0.02)
        self.shared = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
            for _ in range(n_shared)
        )

    def capacity(self, n_tokens: int) -> int:
        return max(1, math.ceil(self.cf * n_tokens * self.k / self.E))

    def _experts(self, buf: torch.Tensor) -> torch.Tensor:
        """buf: (E, C, d) -> (E, C, d), one batched matmul per projection."""
        return torch.bmm(F.gelu(torch.bmm(buf, self.w1)), self.w2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, RouterStats]:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        T = x.shape[0]
        logits = self.router(x).float()  # router in fp32, as in Switch/ST-MoE
        probs = logits.softmax(-1)
        z_loss = torch.logsumexp(logits, -1).pow(2).mean()
        C = self.capacity(T)

        if self.routing == "token_choice":
            out, stats = self._token_choice(x, probs, C)
        else:
            out, stats = self._expert_choice(x, probs, C)
        stats.z_loss = z_loss
        for mlp in self.shared:
            out = out + mlp(x)
        return out.reshape(shape), stats

    def _token_choice(self, x, probs, C):
        T, E, k = x.shape[0], self.E, self.k
        gate, idx = probs.topk(k, dim=-1)  # (T, k)
        if self.renorm and k > 1:
            gate = gate / gate.sum(-1, keepdim=True)
        # Slot of each assignment inside its expert, filling first choices before second ones.
        onehot = F.one_hot(idx.t().reshape(-1), E)  # (k*T, E), choice-major order
        pos = (onehot.cumsum(0) - 1)[torch.arange(k * T, device=x.device), idx.t().reshape(-1)]
        pos = pos.view(k, T).t()  # (T, k)
        keep = pos < C

        # Dispatch: scatter kept tokens into an (E*C, d) buffer.
        flat_slot = (idx * C + pos)[keep]
        tok = torch.arange(T, device=x.device).unsqueeze(1).expand(T, k)[keep]
        buf = x.new_zeros(E * C, x.shape[1])
        buf[flat_slot] = x[tok]
        y = self._experts(buf.view(E, C, -1)).view(E * C, -1)

        # Combine: gather each kept assignment's output, weight by its gate, sum per token.
        out = torch.zeros_like(x)
        out.index_add_(0, tok, y[flat_slot] * gate[keep].unsqueeze(1).to(y.dtype))

        f = F.one_hot(idx, E).sum(1).float().mean(0) / k  # assignment fraction per expert
        aux = E * (f * probs.mean(0)).sum()
        kept_load = torch.bincount(idx[keep], minlength=E).float()
        stats = RouterStats(
            aux_loss=aux,
            z_loss=torch.zeros(()),
            load=kept_load / kept_load.sum().clamp(min=1),
            dropped_frac=float(1 - keep.float().mean()),
            unrouted_frac=float((~keep.any(1)).float().mean()),
        )
        return out, stats

    def _expert_choice(self, x, probs, C):
        T, E = x.shape[0], self.E
        C = min(C, T)
        gate, tok = probs.t().topk(C, dim=-1)  # (E, C): each expert's chosen tokens
        y = self._experts(x[tok])  # (E, C, d)
        out = torch.zeros_like(x)
        out.index_add_(0, tok.reshape(-1), (y * gate.unsqueeze(-1).to(y.dtype)).reshape(E * C, -1))
        counts = torch.bincount(tok.reshape(-1), minlength=T)
        stats = RouterStats(
            aux_loss=torch.zeros((), device=x.device),  # balanced by construction
            z_loss=torch.zeros(()),
            load=torch.full((E,), 1.0 / E, device=x.device),
            dropped_frac=0.0,
            unrouted_frac=float((counts == 0).float().mean()),
            extra={"max_experts_per_token": int(counts.max())},
        )
        return out, stats


def reference_token_choice(moe: MoE, x: torch.Tensor) -> torch.Tensor:
    """Slow loop implementation of the same routing, for testing dispatch/combine."""
    x = x.reshape(-1, x.shape[-1])
    T, E, k = x.shape[0], moe.E, moe.k
    probs = moe.router(x).float().softmax(-1)
    gate, idx = probs.topk(k, -1)
    if moe.renorm and k > 1:
        gate = gate / gate.sum(-1, keepdim=True)
    C = moe.capacity(T)
    used = [0] * E
    out = torch.zeros_like(x)
    for choice in range(k):  # first choices get priority, as in the fast path
        for t in range(T):
            e = int(idx[t, choice])
            if used[e] < C:
                used[e] += 1
                h = F.gelu(x[t] @ moe.w1[e]) @ moe.w2[e]
                out[t] += gate[t, choice] * h
    for mlp in moe.shared:
        out = out + mlp(x)
    return out
