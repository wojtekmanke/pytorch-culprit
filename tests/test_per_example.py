import pytest
import torch

from culprit.flat import Layout
from culprit.grads import per_example_grads
from helpers import gpt_problem, loop_grads, mlp_problem

PROBLEMS = {"mlp": mlp_problem, "gpt": gpt_problem}


@pytest.mark.parametrize("problem", PROBLEMS)
def test_vmap_gradients_match_one_backward_pass_per_example(problem):
    task, x, y = PROBLEMS[problem]()
    got = per_example_grads(task.example_loss, task.params(), x, y)
    expected = loop_grads(task, x, y)
    assert got.keys() == expected.keys()
    for name in expected:
        assert got[name].shape == (len(x), *task.params()[name].shape)
        assert torch.allclose(got[name], expected[name], atol=1e-12)


@pytest.mark.parametrize("problem", PROBLEMS)
def test_measurement_gradients_match_the_loop(problem):
    task, x, y = PROBLEMS[problem]()
    got = per_example_grads(task.example_measure, task.params(), x, y)
    expected = loop_grads(task, x, y, measure=True)
    for name in expected:
        assert torch.allclose(got[name], expected[name], atol=1e-12)


def test_chunking_does_not_change_the_result():
    task, x, y = gpt_problem(n=7)
    whole = per_example_grads(task.example_loss, task.params(), x, y)
    chunked = per_example_grads(task.example_loss, task.params(), x, y, chunk_size=3)
    for name in whole:
        assert torch.allclose(whole[name], chunked[name], atol=1e-12)


def test_gradients_cover_only_the_tasks_parameters():
    task, x, y = mlp_problem(param_names=("layers.2.weight", "layers.2.bias"))
    got = per_example_grads(task.example_loss, task.params(), x, y)
    assert list(got) == ["layers.2.weight", "layers.2.bias"]


def test_per_example_gradients_sum_to_the_batch_gradient():
    task, x, y = gpt_problem()
    params = {name: value.clone().requires_grad_() for name, value in task.params().items()}
    total = torch.autograd.grad(task.losses(params, x, y).sum(), list(params.values()))
    got = per_example_grads(task.example_loss, task.params(), x, y)
    for name, expected in zip(params, total, strict=True):
        assert torch.allclose(got[name].sum(0), expected, atol=1e-12)


def test_layout_round_trips_with_and_without_batch_dimensions():
    task, x, y = gpt_problem()
    params = task.params()
    layout = Layout.of(params)
    assert layout.size == sum(value.numel() for value in params.values())

    vector = layout.flatten(params)
    assert vector.shape == (layout.size,)
    for name, value in layout.unflatten(vector).items():
        assert torch.equal(value, params[name])

    grads = per_example_grads(task.example_loss, params, x, y)
    matrix = layout.flatten(grads)
    assert matrix.shape == (len(x), layout.size)
    for name, value in layout.unflatten(matrix).items():
        assert torch.equal(value, grads[name])


def test_layout_slices_address_each_parameter():
    task, _, _ = mlp_problem()
    params = task.params()
    layout = Layout.of(params)
    vector = layout.flatten(params)
    for name, where in layout.slices().items():
        assert torch.equal(vector[where], params[name].flatten())


def test_layout_rejects_a_vector_of_the_wrong_size():
    task, _, _ = mlp_problem()
    layout = Layout.of(task.params())
    with pytest.raises(ValueError, match="last dimension"):
        layout.unflatten(torch.zeros(layout.size + 1))
