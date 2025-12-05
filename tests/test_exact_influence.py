import pytest
import torch

from culprit.attribution import ExactInfluence
from culprit.curvature import dense_ggn, dense_hessian
from culprit.data.blobs import gaussian_blobs
from culprit.flat import Layout
from culprit.losses import CrossEntropy, LogProb
from culprit.models import MLP
from culprit.task import Task
from helpers import gpt_problem, loop_grads, mlp_problem, newton_fit, one_hot_weights

L2 = 0.05


def convex_problem(n=200, queries=6):
    """Softmax regression: convex, so retraining has one answer to compare with."""
    x, y = gaussian_blobs(n + queries, dim=5, classes=3, spread=1.5)
    task = Task(MLP([5, 3]), loss=CrossEntropy(), measure=LogProb())
    return task, (x[:n], y[:n]), (x[n:], y[n:])


def test_scores_are_the_textbook_formula():
    task, x, y = mlp_problem(n=12)
    queries_x, queries_y = x[:3], (y[:3] + 1) % 3
    scores = ExactInfluence(task, damping=0.1).fit(x, y).score(queries_x, queries_y)

    layout = Layout.of(task.params())
    train_grads = layout.flatten(loop_grads(task, x, y))
    query_grads = layout.flatten(loop_grads(task, queries_x, queries_y, measure=True))
    curvature = dense_ggn(task, task.params(), x, y) + 0.1 * torch.eye(layout.size)
    expected = -query_grads @ torch.linalg.inv(curvature) @ train_grads.T / len(x)
    assert scores.shape == (3, 12)
    assert torch.allclose(scores, expected, atol=1e-12)


def test_an_example_supports_its_own_prediction():
    """With the measurement set to minus the loss, self-influence is a positive quadratic form."""
    task, x, y = mlp_problem(n=12)
    scores = ExactInfluence(task, damping=0.1).fit(x, y).score(x, y)
    assert torch.allclose(scores, scores.T, atol=1e-12)
    assert bool((scores.diagonal() > 0).all())
    assert torch.linalg.eigvalsh(scores).min() > -1e-12


def test_the_hessian_can_be_used_in_place_of_the_gauss_newton_matrix():
    task, x, y = mlp_problem(n=12)
    layout = Layout.of(task.params())
    # Enough damping to make the indefinite Hessian of this untrained model invertible.
    damping = 1.0 - float(torch.linalg.eigvalsh(dense_hessian(task, task.params(), x, y)).min())
    scores = ExactInfluence(task, damping=damping, curvature="hessian").fit(x, y).score(x, y)
    train_grads = layout.flatten(loop_grads(task, x, y))
    curvature = dense_hessian(task, task.params(), x, y) + damping * torch.eye(layout.size)
    expected = train_grads @ torch.linalg.inv(curvature) @ train_grads.T / len(x)
    assert torch.allclose(scores, expected, atol=1e-10)


def test_scoring_before_fitting_is_an_error():
    task, x, y = mlp_problem()
    with pytest.raises(RuntimeError, match="call fit"):
        ExactInfluence(task).score(x, y)


def test_batching_does_not_change_the_scores():
    task, x, y = gpt_problem(n=5)
    whole = ExactInfluence(task).fit(x, y).score(x[:2], y[:2])
    batched = ExactInfluence(task, batch_size=2).fit(x, y).score(x[:2], y[:2])
    assert torch.allclose(whole, batched, atol=1e-10)


def fitted_scores():
    task, (x, y), queries = convex_problem()
    fitted = newton_fit(task, x, y, L2)
    attributor = ExactInfluence(task, fitted, damping=L2, curvature="hessian").fit(x, y)
    return task, (x, y), queries, fitted, attributor.score(*queries)


def test_scores_are_the_derivative_of_retraining():
    """Retraining with an example weighted slightly less moves the measurement by the score."""
    task, (x, y), queries, fitted, scores = fitted_scores()
    before = task.measures(fitted, *queries)
    step = 1e-4
    for index in range(0, len(x), 25):
        lighter = newton_fit(task, x, y, L2, weights=one_hot_weights(len(x), index, 1 - step))
        slope = (before - task.measures(lighter, *queries)) / step
        assert torch.allclose(slope, scores[:, index], rtol=1e-3, atol=1e-9)


def test_scores_predict_leave_one_out_retraining():
    """Removing an example outright is a finite step, so the agreement is only approximate.

    On this problem the largest disagreement was a tenth of the largest
    effect, with a correlation of 0.999.
    """
    task, (x, y), queries, fitted, scores = fitted_scores()
    before = task.measures(fitted, *queries)
    removed = torch.arange(0, len(x), 10)
    actual = torch.stack(
        [
            before
            - task.measures(
                newton_fit(task, x, y, L2, weights=one_hot_weights(len(x), index, 0.0)), *queries
            )
            for index in removed
        ],
        dim=1,
    )
    predicted = scores[:, removed]
    assert (actual - predicted).abs().max() < 0.15 * actual.abs().max()
    correlation = torch.corrcoef(torch.stack([actual.flatten(), predicted.flatten()]))[0, 1]
    assert correlation > 0.995


def test_scores_add_up_to_the_effect_of_removing_a_group():
    task, (x, y), queries, fitted, scores = fitted_scores()
    weights = torch.ones(len(x))
    weights[:20] = 0.0
    without = newton_fit(task, x, y, L2, weights=weights)
    actual = task.measures(fitted, *queries) - task.measures(without, *queries)
    predicted = scores[:, :20].sum(1)
    # A tenth of the data is a large step for a first-order expansion.
    assert (actual - predicted).abs().max() < 0.15 * actual.abs().max()
    assert torch.corrcoef(torch.stack([actual, predicted]))[0, 1] > 0.99
