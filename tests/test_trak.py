import pytest
import torch

from culprit.attribution import TRAK
from culprit.curvature import dense_empirical_fisher
from culprit.flat import Layout
from culprit.losses import Margin
from culprit.projection import Projector
from helpers import gpt_problem, loop_grads, mlp_problem


class Identity(Projector):
    """No projection at all, so that TRAK can be compared with dense algebra."""

    def __init__(self, in_size, seed):
        self.in_size = self.out_size = in_size

    def project(self, matrix):
        return matrix


def correlation(a, b):
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()


def test_with_the_loss_kernel_and_no_projection_it_is_empirical_fisher_influence():
    task, x, y = gpt_problem()
    queries = (x[:2], y[:2])
    scores = TRAK(task, kernel="loss", damping=0.1, projector=Identity).fit(x, y).score(*queries)

    params = task.params()
    layout = Layout.of(params)
    fisher = dense_empirical_fisher(task, params, x, y)
    ridge = 0.1 * torch.trace(fisher)
    train_grads = layout.flatten(loop_grads(task, x, y))
    query_grads = layout.flatten(loop_grads(task, *queries, measure=True))
    damped = fisher + ridge * torch.eye(layout.size)
    expected = -query_grads @ torch.linalg.inv(damped) @ train_grads.T / len(x)
    assert scores.shape == (2, len(x))
    assert torch.allclose(scores, expected, atol=1e-10)


def test_with_the_margin_it_is_the_estimator_of_the_paper():
    """phi(q)^T K^-1 phi(i) (1 - p_i), with K built from gradients of the margin."""
    task, x, y = mlp_problem(n=20, measure=Margin())
    queries = (x[:4], (y[:4] + 1) % 3)
    scores = TRAK(task, damping=0.1, projector=Identity).fit(x, y).score(*queries)

    layout = Layout.of(task.params())
    phi = layout.flatten(loop_grads(task, x, y, measure=True))
    phi_queries = layout.flatten(loop_grads(task, *queries, measure=True))
    p = task.outputs(task.params(), x).softmax(-1).gather(1, y[:, None]).squeeze(1)
    kernel = phi.T @ phi / len(x)
    kernel = kernel + 0.1 * torch.trace(kernel) * torch.eye(layout.size)
    expected = phi_queries @ torch.linalg.inv(kernel) @ phi.T * (1 - p) / len(x)
    assert torch.allclose(scores, expected, atol=1e-10)


def test_a_wide_random_projection_approaches_the_unprojected_scores():
    task, x, y = mlp_problem(n=40)
    queries = (x[:6], (y[:6] + 1) % 3)
    reference = TRAK(task, damping=0.1, projector=Identity).fit(x, y).score(*queries)
    narrow = TRAK(task, damping=0.1, projection_size=8).fit(x, y).score(*queries)
    wide = TRAK(task, damping=0.1, projection_size=4096).fit(x, y).score(*queries)
    assert correlation(wide, reference) > 0.95
    assert correlation(wide, reference) > correlation(narrow, reference)


def test_scores_are_averaged_over_models():
    task, x, y = mlp_problem(n=20)
    first = task.params()
    second = {name: value + 0.3 for name, value in first.items()}
    queries = (x[:3], y[:3])
    both = TRAK(task, [first, second], seed=5, projection_size=64).fit(x, y).score(*queries)
    one = TRAK(task, [first], seed=5, projection_size=64).fit(x, y).score(*queries)
    # The second model of an ensemble gets the next seed.
    other = TRAK(task, [second], seed=6, projection_size=64).fit(x, y).score(*queries)
    assert torch.allclose(both, (one + other) / 2, atol=1e-12)
    assert not torch.allclose(one, other)


def test_scores_depend_on_the_seed_of_the_projection_only():
    task, x, y = gpt_problem()

    def scores(seed):
        return TRAK(task, seed=seed, projection_size=64).fit(x, y).score(x[:2], y[:2])

    torch.manual_seed(0)
    first = scores(1)
    torch.manual_seed(1)
    assert torch.equal(first, scores(1))
    assert not torch.equal(first, scores(2))


def test_batching_does_not_change_the_scores():
    task, x, y = gpt_problem(n=5)
    whole = TRAK(task, projection_size=64).fit(x, y).score(x[:2], y[:2])
    batched = TRAK(task, projection_size=64, batch_size=2).fit(x, y).score(x[:2], y[:2])
    assert torch.allclose(whole, batched, atol=1e-10)


def test_scoring_before_fitting_is_an_error():
    task, x, y = mlp_problem()
    with pytest.raises(RuntimeError, match="call fit"):
        TRAK(task).score(x, y)
