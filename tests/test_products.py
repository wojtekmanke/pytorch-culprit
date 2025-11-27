import pytest
import torch

from culprit.curvature import dense_hessian
from culprit.curvature.products import (
    hvp_double_backward,
    hvp_forward_over_reverse,
    hvp_reverse_over_forward,
)
from culprit.flat import Layout
from helpers import gpt_problem, mlp_problem

PROBLEMS = {"mlp": mlp_problem, "gpt": gpt_problem}
HVPS = {
    "double backward": hvp_double_backward,
    "forward over reverse": hvp_forward_over_reverse,
    "reverse over forward": hvp_reverse_over_forward,
}
# Reverse over forward is wrong through a layer norm on current torch, so it is
# only held to the oracle on the MLP. tests/test_upstream.py isolates the cause.
LAYER_NORM = pytest.mark.xfail(reason="reverse over forward through layer norm", strict=False)
ON_BOTH_MODELS = [("mlp", name) for name in HVPS] + [
    pytest.param("gpt", name, marks=LAYER_NORM if name == "reverse over forward" else ())
    for name in HVPS
]


def random_direction(layout, seed=0):
    vector = torch.randn(layout.size, generator=torch.Generator().manual_seed(seed))
    return vector, layout.unflatten(vector)


@pytest.mark.parametrize(("problem", "method"), ON_BOTH_MODELS)
def test_hvp_matches_the_dense_hessian(problem, method):
    task, x, y = PROBLEMS[problem]()
    params = task.params()
    layout = Layout.of(params)
    flat, direction = random_direction(layout)
    product = HVPS[method](task, params, x, y, direction)
    assert list(product) == list(params)
    expected = len(x) * dense_hessian(task, params, x, y) @ flat  # products are of the sum
    assert torch.allclose(layout.flatten(product), expected, atol=1e-10)


@pytest.mark.parametrize("method", HVPS)
def test_hvp_is_linear_in_the_vector(method):
    task, x, y = gpt_problem()
    params = task.params()
    layout = Layout.of(params)
    (u, u_tree), (v, v_tree) = random_direction(layout, 1), random_direction(layout, 2)
    hvp = HVPS[method]
    combined = hvp(task, params, x, y, layout.unflatten(2.0 * u - 3.0 * v))
    separate = 2.0 * layout.flatten(hvp(task, params, x, y, u_tree)) - 3.0 * layout.flatten(
        hvp(task, params, x, y, v_tree)
    )
    assert torch.allclose(layout.flatten(combined), separate, atol=1e-10)


@pytest.mark.parametrize("method", HVPS)
def test_hvp_adds_up_over_batches(method):
    task, x, y = gpt_problem(n=6)
    params = task.params()
    layout = Layout.of(params)
    _, direction = random_direction(layout)
    hvp = HVPS[method]
    whole = layout.flatten(hvp(task, params, x, y, direction))
    halves = layout.flatten(hvp(task, params, x[:2], y[:2], direction)) + layout.flatten(
        hvp(task, params, x[2:], y[2:], direction)
    )
    assert torch.allclose(whole, halves, atol=1e-10)


@pytest.mark.parametrize("method", HVPS)
def test_hvp_of_a_subset_of_parameters_is_a_block_of_the_hessian(method):
    names = ("layers.0.weight", "layers.2.bias")
    task, x, y = mlp_problem(param_names=names)
    params = task.params()
    layout = Layout.of(params)
    flat, direction = random_direction(layout)
    product = HVPS[method](task, params, x, y, direction)
    block = dense_hessian(task, params, x, y)
    assert torch.allclose(layout.flatten(product), len(x) * block @ flat, atol=1e-10)


@pytest.mark.parametrize("method", HVPS)
def test_hvp_leaves_no_gradients_behind(method):
    task, x, y = mlp_problem()
    params = task.params()
    _, direction = random_direction(Layout.of(params))
    HVPS[method](task, params, x, y, direction)
    assert all(param.grad is None for param in task.model.parameters())
    assert not any(value.requires_grad for value in params.values())


@pytest.mark.mps
@pytest.mark.parametrize(
    ("problem", "method"),
    [("mlp", "forward over reverse"), ("mlp", "reverse over forward")]
    + [("gpt", "forward over reverse")],
)
def test_hvps_agree_with_double_backward_in_single_precision_on_the_gpu(problem, method):
    task, x, y = PROBLEMS[problem]()
    task.model.to("mps", torch.float32)
    x = x.to("mps", torch.float32) if x.is_floating_point() else x.to("mps")
    y = y.to("mps")
    params = task.params()
    layout = Layout.of(params)
    flat = torch.randn(layout.size, generator=torch.Generator().manual_seed(0), dtype=torch.float32)
    direction = layout.unflatten(flat.to("mps"))
    reference = layout.flatten(hvp_double_backward(task, params, x, y, direction))
    product = layout.flatten(HVPS[method](task, params, x, y, direction))
    assert (product - reference).abs().max() < 1e-4 * reference.abs().max()
