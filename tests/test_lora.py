import pytest
import torch
from torch import nn

from culprit.attribution import make_attributor, method_names
from culprit.curvature import dense_empirical_fisher, dense_ggn, ekfac, kfac
from culprit.flat import Layout
from culprit.grads import per_example_grads
from culprit.grads.ghost import capture, gradient_dots
from culprit.grads.hooks import hook_per_example_grads
from culprit.lora import A_NAME, B_NAME, add_lora, is_lora, merge_lora
from culprit.task import Task
from culprit.train import TrainConfig, train
from helpers import GPT_CONFIG, gpt_problem, loop_grads, mlp_problem

TARGETS = ["attn.qkv", "mlp.proj"]


def lora_problem(n=6, rank=2, make=gpt_problem, targets=TARGETS):
    """A GPT with adapters on some of its linear layers, moved off their zero start."""
    task, x, y = make(n=n)
    names = add_lora(task.model, targets, rank=rank, alpha=4.0, seed=0)
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, param in task.model.named_parameters():
            if name.endswith(B_NAME):
                param.copy_(0.1 * torch.randn(param.shape, generator=generator))
    return Task(task.model), x, y, names


def test_only_the_adapters_are_trainable_and_the_task_sees_only_them():
    task, _, _, names = lora_problem()
    assert sorted(task.params()) == sorted(names)
    assert len(names) == 2 * 2 * len(TARGETS)  # two blocks, two halves each
    frozen = [n for n, p in task.model.named_parameters() if not p.requires_grad]
    assert "blocks.0.attn.qkv.parametrizations.weight.original" in frozen
    assert "wte.weight" in frozen
    assert is_lora(task.model.blocks[0].attn.qkv) and not is_lora(task.model.blocks[0].mlp.fc)


def test_a_fresh_adapter_leaves_the_model_unchanged():
    task, x, _ = gpt_problem()
    before = task.model(x)
    add_lora(task.model, TARGETS, rank=2)
    assert torch.equal(task.model(x), before)


def test_the_effective_weight_is_the_frozen_one_plus_the_low_rank_update():
    task, _, _, _ = lora_problem()
    layer = task.model.blocks[0].attn.qkv
    adapter = layer.parametrizations.weight[0]
    expected = layer.parametrizations.weight.original + adapter.scale * adapter.B @ adapter.A
    assert torch.allclose(layer.weight, expected)
    assert (
        adapter.scale == 2.0
        and torch.linalg.matrix_rank(layer.weight - expected + adapter.B @ adapter.A) <= 2
    )


def test_an_unknown_target_is_an_error():
    task, _, _ = gpt_problem()
    with pytest.raises(ValueError, match="no linear layer"):
        add_lora(task.model, ["does.not.exist"])


def test_training_moves_the_adapters_and_nothing_else():
    task, x, y, names = lora_problem(n=32)
    before = {name: value.clone() for name, value in task.model.named_parameters()}
    train(task, x, y, TrainConfig(epochs=2, batch_size=8, lr=0.01))
    for name, value in task.model.named_parameters():
        assert torch.equal(value, before[name]) == (name not in names), name


def test_merging_folds_the_update_into_the_weight():
    task, x, _, _ = lora_problem()
    before = task.model(x)
    merge_lora(task.model)
    assert not any(is_lora(module) for module in task.model.modules())
    assert isinstance(task.model.blocks[0].attn.qkv, nn.Linear)
    assert torch.allclose(task.model(x), before, atol=1e-12)


@pytest.mark.parametrize("measure", [False, True])
def test_per_example_gradients_of_an_adapter_agree_three_ways(measure):
    task, x, y, _ = lora_problem()
    params = task.params()
    fn = task.example_measure if measure else task.example_loss
    by_loop = loop_grads(task, x, y, measure=measure)
    by_vmap = per_example_grads(fn, params, x, y)
    by_hooks = hook_per_example_grads(task, params, x, y, measure=measure)
    assert list(by_hooks) == list(params)
    for name in by_loop:
        assert torch.allclose(by_vmap[name], by_loop[name], atol=1e-12), name
        assert torch.allclose(by_hooks[name], by_loop[name], atol=1e-12), name


def test_hook_gradients_use_the_adapter_values_that_were_passed():
    """The rule reads `A` and `B` during the forward pass, not from the module afterwards."""
    task, x, y, _ = lora_problem()
    shifted = {name: value + 0.05 for name, value in task.params().items()}
    by_hooks = hook_per_example_grads(task, shifted, x, y)
    by_vmap = per_example_grads(task.example_loss, shifted, x, y)
    for name in shifted:
        assert torch.allclose(by_hooks[name], by_vmap[name], atol=1e-12), name


def test_only_the_adapted_layers_are_recorded():
    from culprit.grads.hooks import Recorder

    task, x, y, _ = lora_problem()
    with Recorder(task.model, task.params()) as recorder:
        task.model(x)
    assert sorted(recorder.records) == [
        "blocks.0.attn.qkv",
        "blocks.0.mlp.proj",
        "blocks.1.attn.qkv",
        "blocks.1.mlp.proj",
    ]


def test_gradient_dot_products_match_materialised_gradients():
    task, x, y, _ = lora_problem(n=7)
    params = task.params()
    layout = Layout.of(params)
    left, right = (x[:3], y[:3]), (x[3:], y[3:])
    dots = gradient_dots(
        task, params, capture(task, params, *left, measure=True), capture(task, params, *right)
    )
    expected = (
        layout.flatten(loop_grads(task, *left, measure=True))
        @ layout.flatten(loop_grads(task, *right)).T
    )
    assert torch.allclose(dots, expected, atol=1e-12)


def test_kfac_of_an_adapter_is_exact_for_one_example_without_a_sequence():
    """An MLP, so each half of the adapter sees one input: the Kronecker product is the truth."""
    task, x, y, names = lora_problem(n=1, make=mlp_problem, targets=["layers.0", "layers.2"])
    task = Task(task.model, param_names=tuple(n for n in names if "layers.2" in n))
    params = task.params()
    operator = kfac(task, params, x, y, fisher="exact")
    dense = operator.dense()
    true = dense_ggn(task, params, x, y)
    layout = Layout.of(params)
    for name, where in layout.slices().items():
        # K-FAC keeps each half's own block and drops the block between the halves.
        assert torch.allclose(dense[where, where], true[where, where], atol=1e-12), name


def test_ekfac_fits_the_blocks_of_an_adapter():
    """With the data's labels the eigenvalue fit must reproduce each block's trace."""
    task, x, y, _ = lora_problem()
    params = task.params()
    layout = Layout.of(params)
    true = dense_empirical_fisher(task, params, x, y)
    corrected = ekfac(task, params, x, y, fisher="empirical")
    dense = corrected.dense()
    assert len(corrected.blocks) == len(params)
    for name, where in layout.slices().items():
        mine, theirs = dense[where, where], true[where, where]
        assert torch.trace(mine).item() == pytest.approx(torch.trace(theirs).item(), rel=1e-9), name
        mismatch = torch.trace(mine @ (mine - theirs))
        assert abs(mismatch.item()) < 1e-10 * torch.trace(theirs).item() ** 2


def test_adapter_factors_have_the_small_sides():
    task, x, y, _ = lora_problem(rank=2)
    operator = kfac(task, task.params(), x, y)
    width = GPT_CONFIG.d_model
    block_a = operator.blocks["blocks.0.attn.qkv.lora_A"]
    block_b = operator.blocks["blocks.0.attn.qkv.lora_B"]
    assert block_a.a_sum.shape == (width, width) and block_a.s_sum.shape == (2, 2)
    assert block_b.a_sum.shape == (2, 2) and block_b.s_sum.shape == (3 * width, 3 * width)


@pytest.mark.parametrize("name", [n for n in method_names() if n != "exact"])
def test_every_method_scores_a_model_with_adapters(name):
    task, x, y, _ = lora_problem(n=10)
    scores = make_attributor(name, task).fit(x[3:], y[3:]).score(x[:3], y[:3])
    assert scores.shape == (3, 7) and torch.isfinite(scores).all()


def test_exact_influence_over_adapters_only():
    task, x, y, _ = lora_problem(n=10)
    scores = make_attributor("exact", task, damping=0.1).fit(x[3:], y[3:]).score(x[:3], y[:3])
    ekfac_scores = (
        make_attributor("ekfac", task, damping=0.1, fisher="empirical")
        .fit(x[3:], y[3:])
        .score(x[:3], y[:3])
    )
    assert scores.shape == ekfac_scores.shape == (3, 7)
    assert torch.corrcoef(torch.stack([scores.flatten(), ekfac_scores.flatten()]))[0, 1] > 0.5


def test_names_follow_the_parametrization():
    task, _, _, names = lora_problem()
    assert f"blocks.0.attn.qkv.{A_NAME}" in names and f"blocks.0.attn.qkv.{B_NAME}" in names
    assert A_NAME == "parametrizations.weight.0.A"
