"""The library's machinery on a GPT that uses the fused RMS norm."""

import pytest
import torch

from culprit.attribution import EKFACInfluence, ExactInfluence, make_attributor, method_names
from culprit.curvature import dense_empirical_fisher, dense_ggn, dense_hessian, ekfac
from culprit.curvature.products import (
    ggn_vp,
    hvp_double_backward,
    hvp_forward_over_reverse,
    hvp_reverse_over_forward,
)
from culprit.flat import Layout
from culprit.grads import per_example_grads
from culprit.grads.ghost import capture, gradient_dots
from culprit.grads.hooks import hook_per_example_grads
from culprit.models.rmsnorm import RMSNorm
from helpers import loop_grads, rms_gpt_problem


def test_the_model_uses_the_fused_norm_everywhere():
    task, _, _ = rms_gpt_problem()
    norms = [m for m in task.model.modules() if isinstance(m, RMSNorm)]
    assert len(norms) == 5 and all(norm.fused for norm in norms)
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in task.model.modules())


def test_per_example_gradients_agree_three_ways():
    task, x, y = rms_gpt_problem()
    params = task.params()
    by_loop = loop_grads(task, x, y)
    by_vmap = per_example_grads(task.example_loss, params, x, y)
    by_hooks = hook_per_example_grads(task, params, x, y)
    for name in by_loop:
        assert torch.allclose(by_vmap[name], by_loop[name], atol=1e-12), name
        assert torch.allclose(by_hooks[name], by_loop[name], atol=1e-12), name


@pytest.mark.parametrize(
    "hvp", [hvp_double_backward, hvp_forward_over_reverse, hvp_reverse_over_forward]
)
def test_all_three_hvps_match_the_dense_hessian(hvp):
    """Including reverse over forward, which is wrong through `nn.LayerNorm`."""
    task, x, y = rms_gpt_problem()
    params = task.params()
    layout = Layout.of(params)
    flat = torch.randn(layout.size, generator=torch.Generator().manual_seed(0))
    product = layout.flatten(hvp(task, params, x, y, layout.unflatten(flat)))
    expected = len(x) * dense_hessian(task, params, x, y) @ flat
    assert torch.allclose(product, expected, atol=1e-9)


def test_the_dense_hessian_is_symmetric_and_the_ggn_matches_its_products():
    task, x, y = rms_gpt_problem()
    params = task.params()
    layout = Layout.of(params)
    hessian = dense_hessian(task, params, x, y)
    assert torch.allclose(hessian, hessian.T, atol=1e-12)
    flat = torch.randn(layout.size, generator=torch.Generator().manual_seed(1))
    product = layout.flatten(ggn_vp(task, params, x, y, layout.unflatten(flat)))
    assert torch.allclose(product, len(x) * dense_ggn(task, params, x, y) @ flat, atol=1e-10)


def test_gradient_dot_products_match_materialised_gradients():
    task, x, y = rms_gpt_problem(n=6)
    params = task.params()
    layout = Layout.of(params)
    dots = gradient_dots(
        task, params, capture(task, params, x[:2], y[:2]), capture(task, params, x[2:], y[2:])
    )
    left = layout.flatten(loop_grads(task, x[:2], y[:2]))
    right = layout.flatten(loop_grads(task, x[2:], y[2:]))
    assert torch.allclose(dots, left @ right.T, atol=1e-12)


def test_the_norm_blocks_of_ekfac_are_exact_for_the_empirical_fisher():
    task, x, y = rms_gpt_problem()
    params = task.params()
    where = Layout.of(params).slices()["ln_f.weight"]
    true = dense_empirical_fisher(task, params, x, y)[where, where]
    mine = ekfac(task, params, x, y, fisher="empirical").dense()[where, where]
    assert torch.allclose(mine, true, atol=1e-12)


def test_ekfac_influence_tracks_exact_influence():
    task, x, y = rms_gpt_problem(n=12)
    train, queries = (x[4:], y[4:]), (x[:4], y[:4])
    exact = ExactInfluence(task, damping=0.05).fit(*train).score(*queries)
    approximate = EKFACInfluence(task, damping=0.05, samples=20).fit(*train).score(*queries)
    correlation = torch.corrcoef(torch.stack([exact.flatten(), approximate.flatten()]))[0, 1]
    assert correlation > 0.8


@pytest.mark.parametrize("name", method_names())
def test_every_method_runs_on_the_model(name):
    task, x, y = rms_gpt_problem(n=8)
    scores = make_attributor(name, task).fit(x[2:], y[2:]).score(x[:2], y[:2])
    assert scores.shape == (2, 6) and torch.isfinite(scores).all()
