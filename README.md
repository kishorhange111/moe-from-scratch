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

**Results are pending.** The runs are executing on Colab GPUs, and the tables will be
filled in from `results/runs.jsonl`.
