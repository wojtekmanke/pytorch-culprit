import torch

from culprit.attribution import ExactInfluence, LowRankInfluence
from culprit.curvature import dense_ggn
from culprit.flat import Layout
from helpers import gpt_problem, loop_grads, mlp_problem


def correlation(a, b):
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()


def problem(make=mlp_problem, n=12):
    task, x, y = make(n=n)
    return task, (x[4:], y[4:]), (x[:4], y[:4])


def test_it_is_exact_once_the_rank_covers_the_curvature():
    """Eight examples with three classes give a Gauss-Newton matrix of rank at most 16."""
    task, train, queries = problem()
    rank = int(torch.linalg.matrix_rank(dense_ggn(task, task.params(), *train)))
    assert rank < 39  # fewer directions than parameters, or the test shows nothing
    exact = ExactInfluence(task, damping=0.05).fit(*train).score(*queries)
    scores = (
        LowRankInfluence(task, rank=rank, damping=0.05, lanczos_steps=39)
        .fit(*train)
        .score(*queries)
    )
    assert torch.allclose(scores, exact, atol=1e-8 * exact.abs().max())


def test_the_eigenpairs_are_those_of_the_curvature():
    task, train, _ = problem(gpt_problem)
    attributor = LowRankInfluence(task, rank=4, lanczos_steps=60).fit(*train)
    expected = torch.linalg.eigvalsh(dense_ggn(task, task.params(), *train)).flip(0)[:4]
    assert torch.allclose(attributor.eigenvalues, expected, rtol=1e-6)
    basis = attributor.eigenvectors
    assert torch.allclose(basis.T @ basis, torch.eye(4), atol=1e-8)


def test_a_higher_rank_is_closer_to_exact():
    task, train, queries = problem(gpt_problem, n=12)
    exact = ExactInfluence(task, damping=0.05).fit(*train).score(*queries)

    def error(rank):
        scores = (
            LowRankInfluence(task, rank=rank, damping=0.05, lanczos_steps=200)
            .fit(*train)
            .score(*queries)
        )
        return (scores - exact).norm().item()

    assert error(64) < error(16) < error(2)


def test_without_the_complement_only_the_top_directions_count():
    task, train, queries = problem()
    attributor = LowRankInfluence(task, rank=3, damping=0.05, complement=False).fit(*train)
    scores = attributor.score(*queries)

    layout = Layout.of(task.params())
    train_grads = layout.flatten(loop_grads(task, *train))
    query_grads = layout.flatten(loop_grads(task, *queries, measure=True))
    u, values = attributor.eigenvectors, attributor.eigenvalues
    expected = -(query_grads @ u) / (values + 0.05) @ (train_grads @ u).T / len(train[0])
    assert torch.allclose(scores, expected, atol=1e-10)


def test_dropping_the_complement_changes_the_ranking():
    """With it, directions outside the basis are divided by the damping and dominate."""
    task, train, queries = problem(gpt_problem, n=12)
    kept = LowRankInfluence(task, rank=8, damping=0.05).fit(*train).score(*queries)
    dropped = (
        LowRankInfluence(task, rank=8, damping=0.05, complement=False).fit(*train).score(*queries)
    )
    assert correlation(kept, dropped) < 0.99
