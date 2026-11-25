"""Train and evaluate dense vs MoE byte-level GPTs.

    python -m moe.train --exp all --device cuda --out results

Corpus: the Python standard library's source, byte-level (no download, present on any
machine with Python), split by file. Every run appends one JSON line to
``<out>/runs.jsonl`` and is skipped if already there, so a reclaimed GPU session resumes.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sysconfig
import time
import zlib
from dataclasses import asdict, replace

import numpy as np
import torch
import torch.nn.functional as F

from .model import GPT, Config, param_counts


def load_corpus(root: str | None = None, val_frac: float = 0.05):
    root = root or sysconfig.get_paths()["stdlib"]
    files = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
    if not files:
        raise FileNotFoundError(root)
    tr, va = [], []
    for f in files:
        rel = os.path.relpath(f, root).replace("\\", "/")
        with open(f, "rb") as fh:
            (va if zlib.crc32(rel.encode()) % 1000 < val_frac * 1000 else tr).append(fh.read())
    return (np.frombuffer(b"\n".join(tr), np.uint8).copy(),
            np.frombuffer(b"\n".join(va), np.uint8).copy())  # fmt: skip


def batch(data, bs, ctx, gen, device):
    ix = torch.randint(len(data) - ctx - 1, (bs,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i : i + ctx + 1].astype(np.int64)) for i in ix])
    return x[:, :-1].to(device), x[:, 1:].to(device)


def run(cfg: Config, steps: int, bs: int, lr: float, aux_coef: float, z_coef: float,
        train, val, device: str, seed: int = 0, eval_every: int = 250) -> dict:  # fmt: skip
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    amp = device == "cuda"
    # fp16 autocast needs loss scaling: without it, small gradients flush to zero in fp16.
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    hist = []
    t0 = time.time()
    for step in range(1, steps + 1):
        frac = step / steps
        for g in opt.param_groups:  # warmup then cosine
            g["lr"] = lr * min(1.0, step / 100) * 0.5 * (1 + math.cos(math.pi * frac))
        x, y = batch(train, bs, cfg.ctx, gen, device)
        with torch.autocast("cuda", torch.float16, enabled=amp):
            logits, stats = model(x)
            loss = F.cross_entropy(logits.float().view(-1, cfg.vocab), y.reshape(-1))
        aux = sum((s.aux_loss for s in stats), torch.zeros((), device=device))
        z = sum((s.z_loss for s in stats), torch.zeros((), device=device))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss + aux_coef * aux + z_coef * z).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step % eval_every == 0 or step == steps:
            hist.append({"step": step, "train": loss.item(), **evaluate(model, val, cfg, device)})
    out = {
        "params": param_counts(model),
        "final": hist[-1],
        "history": hist,
        "seconds": time.time() - t0,
    }
    return out


@torch.no_grad()
def evaluate(model, val, cfg, device, n_batches: int = 20, bs: int = 16) -> dict:
    model.eval()
    gen = torch.Generator().manual_seed(1234)  # same eval batches for every model
    losses, drops, unrouted, cv = [], [], [], []
    for _ in range(n_batches):
        x, y = batch(val, bs, cfg.ctx, gen, device)
        logits, stats = model(x)
        losses.append(float(F.cross_entropy(logits.float().view(-1, cfg.vocab), y.reshape(-1))))
        for s in stats:
            drops.append(s.dropped_frac)
            unrouted.append(s.unrouted_frac)
            cv.append(float(s.load.std() / s.load.mean()))
    model.train()
    r = {"val": float(np.mean(losses)), "val_bpb": float(np.mean(losses)) / math.log(2)}
    if drops:
        r.update(dropped=float(np.mean(drops)), unrouted=float(np.mean(unrouted)),
                 load_cv=float(np.mean(cv)))  # fmt: skip
    return r


def experiments(base: Config) -> dict[str, list[tuple[str, Config, dict]]]:
    """Named experiment groups: (run name, config, training overrides)."""
    moe = replace(base, n_experts=8, k=2, d_ff=base.d_ff // 2)  # active FF = 2 * d_ff/2 = d_ff
    return {
        # Equal active parameters per token: dense d_ff vs 8 experts of d_ff/2 with top-2.
        "dense_vs_moe": [
            ("dense", base, {}),
            ("moe8 top2", moe, {}),
            ("moe8 top2 +1 shared", replace(moe, n_shared=1, d_ff=base.d_ff // 4), {}),
            ("dense 4x width (= moe total params)", replace(base, d_ff=base.d_ff * 4), {}),
        ],
        "aux_loss": [(f"aux={c}", moe, {"aux_coef": c}) for c in (0.0, 0.001, 0.01, 0.1)],
        "capacity": [
            (f"cf={cf}", replace(moe, capacity_factor=cf), {}) for cf in (0.5, 1.0, 1.25, 2.0)
        ],
        "routing": [
            ("token choice top2", moe, {}),
            ("expert choice cf=1", replace(moe, routing="expert_choice", capacity_factor=1.0), {}),
            ("switch top1 (2x expert width)", replace(moe, k=1, d_ff=base.d_ff), {}),
        ],
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "runs.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {(r["exp"], r["name"], r["seed"]) for r in map(json.loads, f) if r}
    train, val = load_corpus()
    base = Config(ctx=a.ctx, d_model=a.d_model, n_layers=a.layers, n_heads=a.d_model // 64,
                  d_ff=4 * a.d_model)  # fmt: skip
    groups = experiments(base)
    for exp, runs in groups.items():
        if a.exp not in ("all", exp):
            continue
        for name, cfg, over in runs:
            for seed in a.seeds:
                if (exp, name, seed) in done:
                    continue
                kw = {"aux_coef": 0.01, "z_coef": 1e-3, **over}
                r = run(cfg, a.steps, a.bs, a.lr, kw["aux_coef"], kw["z_coef"], train, val,
                        a.device, seed)  # fmt: skip
                rec = {"exp": exp, "name": name, "seed": seed, "config": asdict(cfg),
                       "train_kw": kw, "steps": a.steps, "bs": a.bs, **r}  # fmt: skip
                with open(path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                fin = r["final"]
                print(f"{exp:12s} {name:38s} seed={seed} val_bpb={fin['val_bpb']:.4f} "
                      f"drop={fin.get('dropped', 0):.3f} cv={fin.get('load_cv', 0):.2f} "
                      f"active={r['params']['active']:,} total={r['params']['total']:,} "
                      f"{r['seconds']:.0f}s", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
