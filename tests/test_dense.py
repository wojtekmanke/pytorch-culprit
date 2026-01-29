import pytest
import torch

from culprit.curvature import dense_empirical_fisher, dense_ggn, dense_hessian
from culprit.data.blobs import gaussian_blobs
from culprit.flat import Layout
from culprit.models import MLP
from culprit.task import Task
from helpers import gpt_problem, loop_grads, mlp_problem

PROBLEMS = {"mlp": mlp_problem, "gpt": gpt_problem}


def mean_gradient(task, layout, vector, x, y):
    params = {name: value.requires_grad_() for name, value in layout.unflatten(vector).items()}
    grads = torch.autograd.grad(task.losses(params, x, y).mean(), list(params.values()))
    return torch.cat([g.flatten() for g in grads])


@pytest.mark.parametrize("problem", PROBLEMS)
def test_hessian_matches_finite_differences_of_the_gradient(problem):
    """Checks the oracle without using second-order autograd at all."""
    task, x, y = PROBLEMS[problem]()
    params = task.params()
    layout = Layout.of(params)
    vector = layout.flatten(params)
    full = dense_hessian(task, params, x, y)
    assert full.shape == (layout.size, layout.size)
    assert torch.allclose(full, full.T, atol=1e-12)

    # Along a unit direction the error of a central difference is the step
    # squared times a third derivative: about 1e-8 here, far below the 0.5 by
    # which torch.func.hessian is wrong for the GPT (see test_upstream.py).
    step = 1e-5
    generator = torch.Generator().manual_seed(0)
    for _ in range(3):
        direction = torch.randn(layout.size, generator=generator)
        direction /= direction.norm()
        ahead = mean_gradient(task, layout, vector + step * direction, x, y)
        behind = mean_gradient(task, layout, vector - step * direction, x, y)
        assert torch.allclose(full @ direction, (ahead - behind) / (2 * step), atol=1e-6)


@pytest.mark.parametrize("problem", PROBLEMS)
def test_ggn_is_symmetric_and_positive_semi_definite(problem):
    task, x, y = PROBLEMS[problem]()
    ggn = dense_ggn(task, task.params(), x, y)
    assert torch.allclose(ggn, ggn.T, atol=1e-12)
    assert torch.linalg.eigvalsh(ggn).min() > -1e-10


def test_ggn_equals_the_hessian_when_the_model_is_linear_in_its_parameters():
    x, y = gaussian_blobs(12, dim=5, classes=3)
    task = Task(MLP([5, 3]))
    params = task.params()
    assert torch.allclose(dense_ggn(task, params, x, y), dense_hessian(task, params, x, y))


@pytest.mark.parametrize("problem", PROBLEMS)
def test_ggn_differs_from_the_hessian_when_the_model_is_not_linear(problem):
    task, x, y = PROBLEMS[problem]()
    params = task.params()
    difference = dense_ggn(task, params, x, y) - dense_hessian(task, params, x, y)
    assert difference.abs().max() > 1e-4


def test_ggn_is_the_fisher_information_for_cross_entropy():
    """Averaging gradient outer products over the model's own label distribution."""
    task, x, _ = mlp_problem()
    params = task.params()
    layout = Layout.of(params)
    probabilities = task.outputs(params, x).softmax(-1)
    fisher = torch.zeros(layout.size, layout.size)
    for label in range(probabilities.shape[1]):
        labels = torch.full((len(x),), label)
        grads = layout.flatten(loop_grads(task, x, labels))
        fisher += torch.einsum("b,bi,bj->ij", probabilities[:, label], grads, grads)
    assert torch.allclose(dense_ggn(task, params, x, x.new_zeros(len(x)).long()), fisher / len(x))


def test_ggn_does_not_depend_on_the_labels_for_cross_entropy():
    task, x, y = mlp_problem()
    params = task.params()
    assert torch.allclose(dense_ggn(task, params, x, y), dense_ggn(task, params, x, (y + 1) % 3))


def test_empirical_fisher_is_the_mean_gradient_outer_product():
    task, x, y = mlp_problem()
    params = task.params()
    grads = Layout.of(params).flatten(loop_grads(task, x, y))
    expected = sum(torch.outer(g, g) for g in grads) / len(x)
    assert torch.allclose(dense_empirical_fisher(task, params, x, y), expected)


def test_matrices_cover_only_the_tasks_parameters():
    task, x, y = mlp_problem(param_names=("layers.2.weight",))
    params = task.params()
    whole_task, _, _ = mlp_problem()
    whole = dense_hessian(whole_task, whole_task.params(), x, y)
    where = Layout.of(whole_task.params()).slices()["layers.2.weight"]
    assert torch.allclose(dense_hessian(task, params, x, y), whole[where, where])


@pytest.mark.parametrize("matrix", [dense_hessian, dense_ggn, dense_empirical_fisher])
def test_forming_a_matrix_in_batches_gives_the_same_matrix(matrix):
    task, x, y = gpt_problem(n=7)
    params = task.params()
    whole = matrix(task, params, x, y)
    in_batches = matrix(task, params, x, y, batch_size=3)
    assert torch.allclose(in_batches, whole, atol=1e-12)
