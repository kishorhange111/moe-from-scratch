# moe-from-scratch

[![CI](https://github.com/kishorhange111/moe-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/moe-from-scratch/actions/workflows/ci.yml)

A Mixture-of-Experts layer written out in full, with no MoE library, plus controlled
experiments on what actually matters when training one.

## What's implemented ([`layers.py`](src/moe/layers.py))

- **Token-choice top-k routing** (Switch for k=1, GShard/Mixtral-style k=2), with gate
  renormalisation.
- **Capacity and token dropping:** each expert has `C = ⌈cf · T · k / E⌉` slots. Slots fill
  first choices before second choices (GShard priority), and overflow assignments are
  dropped, so the residual stream carries the token past that expert.
- **Expert-choice routing** (Zhou et al. 2022): each expert takes its top-C tokens. It is
  balanced by construction, but a token can get zero experts or several.
- **Batched experts:** weights stacked as `(E, d, f)` and run as one `bmm` over an `(E, C, d)`
  dispatch buffer. This is the layout expert parallelism shards across devices, where
  dispatch and combine become all-to-alls.
- **Auxiliary losses:** Switch load balancing `E · Σ fᵢ Pᵢ` (1.0 when balanced), and the
  ST-MoE router z-loss `mean(logsumexp(logits)²)`.
- **Shared experts** (DeepSeekMoE): dense experts every token passes through.
- **Router statistics:** dropped-assignment fraction, unrouted-token fraction, and
  expert-load coefficient of variation.

## Tests

`pytest -q` runs 11 tests:
- the vectorised dispatch/combine matches a slow per-token loop, across k and capacity factors;
- capacity is never exceeded, and collapsed routing raises the balance loss above 1;
- a 1-expert MoE equals a dense MLP;
- the balance loss is exactly 1 under uniform router probabilities;
- expert choice is balanced;
- gradients reach the router;
- active-parameter accounting is exact;
- a tiny model trains.

## Experiments ([`train.py`](src/moe/train.py))

Byte-level GPTs (d=256, 4 layers, context 256) trained on the Python standard library's
source. The corpus needs no download and is split by file. Every group compares models at
**equal active parameters per token** (8 experts of half width with top-2 = one dense MLP),
so training FLOPs are matched:

| group | question |
|---|---|
| `dense_vs_moe` | Does MoE beat dense at equal active params? How does it compare with a dense model of equal *total* params? Do shared experts help? |
| `aux_loss` | What does the balance loss coefficient (0 / 0.001 / 0.01 / 0.1) do to load balance, dropped tokens and loss? |
| `capacity` | How much quality does token dropping cost (capacity factor 0.5 / 1.0 / 1.25 / 2.0)? |
| `routing` | Token choice vs expert choice vs Switch top-1 |

```
python -m moe.train --exp all --device cuda --out results   # resumable: one JSON line per run
```

## Results (Tesla T4, fp16 with loss scaling, 3,000 steps, seed 0)

The raw records are in [`results/runs.jsonl`](results/runs.jsonl). The default MoE (8
experts, top-2, aux 0.01, capacity 1.25) appears in all four groups. Its four runs, with
the same seed, give **1.373–1.384 bits/byte**. That ±0.006 spread comes from GPU
nondeterminism in the scatter-add and is the noise floor for every comparison below.

**Dense vs MoE** (active = parameters used per token):

| model | active / total params | val bits/byte | wall time |
|---|---|---|---|
| dense | 3.29M / 3.29M | 1.381 | 67 s |
| MoE 8 × top-2 | 3.29M / 9.59M | 1.376 | 160 s |
| MoE 8 × top-2 + 1 shared expert (narrower experts) | 2.77M / 5.92M | 1.412 | 155 s |
| dense, 4× MLP width (= MoE total params) | 9.59M / 9.59M | **1.353** | 141 s |

**Balance loss**:

| aux coefficient | dropped tokens | load CV | val bits/byte |
|---|---|---|---|
| 0 | 45.4% | 1.18 | 1.524 |
| 0.001 | 6.0% | 0.29 | 1.452 |
| 0.01 | 0.1% | 0.14 | 1.373 |
| 0.1 | 0.0% | 0.10 | 1.375 |

**Capacity factor**:

| capacity factor | dropped tokens | val bits/byte |
|---|---|---|
| 0.5 | 50.1% | 1.628 |
| 1.0 | 5.2% | 1.408 |
| 1.25 | 0.1% | 1.383 |
| 2.0 | 0.0% | 1.398 |

**Routing**:

| routing | dropped | val bits/byte | wall time |
|---|---|---|---|
| token choice, top-2 | 0.1% | 1.384 | 157 s |
| expert choice, cf = 1 | 0% (by construction) | 1.382 | 98 s |
| Switch top-1, 2× expert width | 0.8% | 1.430 | 141 s |

### What it shows

- **At this scale, MoE matches dense at equal active parameters but does not beat it.**
  1.376 against 1.381 is inside the noise floor. Its 3× extra parameters buy nothing yet,
  while a dense model *of the same total size* is clearly better (1.353). The MoE scaling
  advantage reported in the literature appears with more experts, data and steps than
  3,000 steps on 10 MB of bytes. This is the honest small-scale result, not a failure of the
  layer.
- **The balance loss is what keeps MoE alive.** Without it the router collapses onto a few
  experts (load CV 1.18), and with capacity limits that drops 45% of tokens, costing 0.15
  bits/byte. 0.01 is the sweet spot; 0.1 balances harder for no gain.
- **Dropped tokens cost loss almost linearly.** Halving capacity drops half the tokens and
  costs 0.25 bits/byte. Above about 1.25 there is nothing left to gain.
- **Expert choice ties token choice on loss and is 1.6× faster here.** Each expert takes
  exactly C tokens, so the dispatch is a fixed-shape gather with no capacity bookkeeping.
- **Top-1 with double-width experts is worse than top-2** at the same active parameters
  (1.430 vs 1.384).
- **MoE costs 2.4× the wall time of dense** in this implementation: plain PyTorch dispatch
  with a scatter-add combine and no fused grouped-GEMM kernel. That is the systems side of
  MoE: matched FLOPs do not mean matched time without a dispatch kernel.

Caveats: one seed per configuration (bounded by the four repeated runs above), a small
model, and a short budget.
