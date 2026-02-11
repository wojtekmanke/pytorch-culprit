import pytest
import torch
from torch import nn

from culprit.flat import Layout
from culprit.grads.ghost import capture, ghost_is_cheaper, gradient_dots
from helpers import gpt_problem, loop_grads, mlp_problem

PROBLEMS = {"mlp": mlp_problem, "gpt": gpt_problem}


def expected_dots(task, left, right):
    """The same table from fully materialised per-example gradients."""
    layout = Layout.of(task.params())
    left_grads = layout.flatten(loop_grads(task, *left, measure=True))
    right_grads = layout.flatten(loop_grads(task, *right))
    return left_grads @ right_grads.T


@pytest.mark.parametrize("problem", PROBLEMS)
def test_dots_match_materialised_gradients(problem):
    task, x, y = PROBLEMS[problem](n=7)
    left, right = (x[:3], y[:3]), (x[3:], y[3:])
    params = task.params()
    dots = gradient_dots(
        task, params, capture(task, params, *left, measure=True), capture(task, params, *right)
    )
    assert dots.shape == (3, 4)
    assert torch.allclose(dots, expected_dots(task, left, right), atol=1e-12)


@pytest.mark.parametrize(
    "names",
    [("layers.0.weight",), ("layers.2.bias",), ("layers.0.bias", "layers.2.weight")],
)
def test_dots_cover_only_the_tasks_parameters(names):
    task, x, y = mlp_problem(n=7, param_names=names)
    left, right = (x[:3], y[:3]), (x[3:], y[3:])
    params = task.params()
    dots = gradient_dots(
        task, params, capture(task, params, *left, measure=True), capture(task, params, *right)
    )
    assert torch.allclose(dots, expected_dots(task, left, right), atol=1e-12)


def test_capture_keeps_only_layers_with_wanted_parameters():
    task, x, y = gpt_problem(param_names=("head.weight", "blocks.0.ln1.bias"))
    captured = capture(task, task.params(), x, y)
    assert sorted(captured.layers) == ["blocks.0.ln1", "head"]
    assert not any(t.requires_grad for a, g, _ in captured.layers.values() for t in (a, g))


def test_a_batch_dotted_with_itself_gives_squared_gradient_norms():
    task, x, y = gpt_problem()
    params = task.params()
    captured = capture(task, params, x, y)
    dots = gradient_dots(task, params, captured, captured)
    norms = Layout.of(params).flatten(loop_grads(task, x, y)).square().sum(1)
    assert torch.allclose(dots.diagonal(), norms, atol=1e-12)
    assert torch.allclose(dots, dots.T, atol=1e-12)


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("strategy", ["auto", "ghost", "materialise"])
def test_every_strategy_gives_the_same_table(problem, strategy):
    task, x, y = PROBLEMS[problem](n=7)
    left, right = (x[:3], y[:3]), (x[3:], y[3:])
    params = task.params()
    dots = gradient_dots(
        task,
        params,
        capture(task, params, *left, measure=True),
        capture(task, params, *right),
        strategy=strategy,
    )
    assert torch.allclose(dots, expected_dots(task, left, right), atol=1e-12)


@pytest.mark.parametrize("strategy", ["ghost", "materialise"])
def test_the_two_sides_may_have_different_sequence_lengths(strategy):
    task, x, y = gpt_problem(n=7)
    left, right = (x[:3, :4], y[:3, :4]), (x[3:], y[3:])
    params = task.params()
    dots = gradient_dots(
        task,
        params,
        capture(task, params, *left, measure=True),
        capture(task, params, *right),
        strategy=strategy,
    )
    assert torch.allclose(dots, expected_dots(task, left, right), atol=1e-12)


@pytest.mark.parametrize("strategy", ["ghost", "materialise"])
@pytest.mark.parametrize(
    "names",
    [("head.weight",), ("blocks.0.mlp.fc.bias",), ("wte.weight", "blocks.1.attn.qkv.weight")],
)
def test_strategies_respect_which_parameters_are_wanted(strategy, names):
    task, x, y = gpt_problem(n=6, param_names=names)
    left, right = (x[:2], y[:2]), (x[2:], y[2:])
    params = task.params()
    dots = gradient_dots(
        task,
        params,
        capture(task, params, *left, measure=True),
        capture(task, params, *right),
        strategy=strategy,
    )
    assert torch.allclose(dots, expected_dots(task, left, right), atol=1e-12)


def test_the_cheaper_strategy_depends_on_sequence_length_and_layer_size():
    wide, narrow = nn.Linear(512, 512), nn.Linear(8, 8)
    assert ghost_is_cheaper(wide, 10, 10)  # 100 positions pairs against 262,144 weights
    assert not ghost_is_cheaper(wide, 100, 100)
    assert not ghost_is_cheaper(narrow, 4, 4)
    assert ghost_is_cheaper(nn.Embedding(50_000, 16), 100, 100)
    assert not ghost_is_cheaper(nn.Embedding(50, 16), 10, 10)
    assert not ghost_is_cheaper(nn.LayerNorm(8), 1, 1)
