import pytest
import torch

from culprit.flat import Layout
from culprit.grads.ghost import capture, gradient_dots
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
    assert not any(t.requires_grad for pair in captured.layers.values() for t in pair)


def test_a_batch_dotted_with_itself_gives_squared_gradient_norms():
    task, x, y = gpt_problem()
    params = task.params()
    captured = capture(task, params, x, y)
    dots = gradient_dots(task, params, captured, captured)
    norms = Layout.of(params).flatten(loop_grads(task, x, y)).square().sum(1)
    assert torch.allclose(dots.diagonal(), norms, atol=1e-12)
    assert torch.allclose(dots, dots.T, atol=1e-12)
