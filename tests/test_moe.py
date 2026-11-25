import math

import pytest
import torch
from torch import nn

from moe.layers import MoE, reference_token_choice
from moe.model import GPT, Config, param_counts
from moe.train import run


def _moe(**kw):
    torch.manual_seed(0)
    return MoE(16, 32, kw.pop("E", 4), **kw)


@pytest.mark.parametrize("k,cf", [(1, 1.0), (2, 1.25), (2, 0.5), (3, 0.75)])
def test_fast_dispatch_matches_loop_reference(k, cf):
    m = _moe(k=k, capacity_factor=cf)
    x = torch.randn(40, 16)
    y, stats = m(x)
    torch.testing.assert_close(y, reference_token_choice(m, x), atol=1e-5, rtol=1e-5)
    if cf < 1:
        assert stats.dropped_frac > 0


def test_capacity_is_never_exceeded():
    m = _moe(k=2, capacity_factor=0.5)
    with torch.no_grad():  # make expert 0 overwhelmingly popular
        m.router.weight.zero_()
        m.router.weight[0] = 10.0
    x = torch.rand(64, 16) + 0.5
    _, stats = m(x)
    C = m.capacity(64)
    assert stats.load[0] * (1 - stats.dropped_frac) * 64 * 2 <= C + 1e-6
    assert stats.dropped_frac > 0.4
    assert stats.aux_loss > 1.5  # collapsed routing is penalised (1.0 = balanced)


def test_single_expert_equals_dense_mlp():
    m = _moe(E=1, k=1, capacity_factor=1.0)
    dense = nn.Sequential(nn.Linear(16, 32, bias=False), nn.GELU(), nn.Linear(32, 16, bias=False))
    with torch.no_grad():
        dense[0].weight.copy_(m.w1[0].t())
        dense[2].weight.copy_(m.w2[0].t())
    x = torch.randn(10, 16)
    torch.testing.assert_close(m(x)[0], dense(x), atol=1e-6, rtol=1e-6)


def test_balance_loss_is_one_when_uniform():
    m = _moe(E=4, k=1, capacity_factor=4.0)
    with torch.no_grad():
        m.router.weight.zero_()  # uniform probabilities; top-1 ties break to expert 0 though
    x = torch.randn(32, 16)
    _, stats = m(x)
    # P_i is exactly uniform, so aux = E * sum_i f_i / E = sum_i f_i = 1 whatever f is.
    assert float(stats.aux_loss.detach()) == pytest.approx(1.0)


def test_expert_choice_is_balanced_and_can_leave_tokens_unrouted():
    m = _moe(E=4, k=1, capacity_factor=1.0, routing="expert_choice")
    x = torch.randn(32, 16)
    y, stats = m(x)
    assert y.shape == x.shape
    assert stats.dropped_frac == 0.0
    assert 0.0 <= stats.unrouted_frac < 1.0


def test_gradients_reach_router_and_experts():
    m = _moe(k=2)
    y, stats = m(torch.randn(24, 16))
    (y.pow(2).mean() + 0.01 * stats.aux_loss).backward()
    assert m.router.weight.grad.abs().sum() > 0
    assert m.w1.grad.abs().sum() > 0


def test_active_params_match_dense_baseline():
    base = Config(d_model=64, n_heads=1, d_ff=256, n_layers=2, ctx=32)
    dense = param_counts(GPT(base))
    moe = param_counts(GPT(Config(**{**base.__dict__, "n_experts": 8, "k": 2, "d_ff": 128})))
    router = 2 * 64 * 8
    biases = 2 * (256 + 64)  # dense MLP biases; the experts have none
    assert moe["active"] == dense["active"] + router - biases
    assert moe["total"] > 3 * dense["total"] - 4 * 64 * 64 * 2 * 2
    # expert choice with cf = 1 activates k·cf = 2 experts per token on average: same count
    ec = param_counts(
        GPT(
            Config(
                **{
                    **base.__dict__,
                    "n_experts": 8,
                    "k": 2,
                    "d_ff": 128,
                    "routing": "expert_choice",
                    "capacity_factor": 1.0,
                }
            )
        )
    )
    assert ec["active"] == moe["active"]


def test_tiny_moe_trains():
    rng = torch.Generator().manual_seed(0)
    data = torch.randint(0, 8, (5000,), generator=rng).numpy().astype("uint8").repeat(2)
    cfg = Config(d_model=32, n_heads=1, d_ff=64, n_layers=1, ctx=16, n_experts=4)
    r = run(cfg, 60, 8, 3e-3, 0.01, 1e-3, data, data, "cpu", eval_every=30)
    first, last = r["history"][0], r["history"][-1]
    assert last["val"] < math.log(256)
    assert last["val"] <= first["val"] + 1e-6
    assert 0 <= last["dropped"] <= 1
