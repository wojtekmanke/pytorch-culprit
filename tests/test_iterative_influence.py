import pytest
import torch

from culprit.attribution import CGInfluence, ExactInfluence, LiSSAInfluence
from culprit.curvature import dense_ggn, dense_hessian
from culprit.data.blobs import gaussian_blobs
from culprit.models import MLP
from culprit.task import Task
from helpers import gpt_problem, mlp_problem


def correlation(a, b):
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()


def problem_with_queries(make=mlp_problem, n=30):
    task, x, y = make(n=n)
    return task, (x[5:], y[5:]), (x[:5], y[:5])


@pytest.mark.parametrize("make", [mlp_problem, gpt_problem])
@pytest.mark.parametrize("curvature", ["ggn", "hessian"])
def test_cg_run_to_convergence_is_exact_influence(make, curvature):
    task, train, queries = problem_with_queries(make, n=12)
    damping = 0.05
    if curvature == "hessian":
        # The Hessian of an untrained model is indefinite; damp it until it is not.
        hessian = dense_hessian(task, task.params(), *train)
        damping = 1.0 - float(torch.linalg.eigvalsh(hessian).min())
    exact = ExactInfluence(task, damping=damping, curvature=curvature).fit(*train).score(*queries)
    attributor = CGInfluence(
        task, damping=damping, curvature=curvature, tol=1e-12, max_iterations=2000, batch_size=4
    ).fit(*train)
    scores = attributor.score(*queries)
    assert attributor.info.converged
    assert torch.allclose(scores, exact, atol=1e-8 * exact.abs().max())


def test_cg_reports_an_unfinished_solve():
    task, train, queries = problem_with_queries(gpt_problem, n=12)
    attributor = CGInfluence(task, damping=1e-4, tol=1e-12, max_iterations=3).fit(*train)
    attributor.score(*queries)
    assert attributor.info.iterations == 3 and not attributor.info.converged
    assert attributor.info.residual.shape == (5,)


def test_more_cg_iterations_get_closer_to_exact():
    task, train, queries = problem_with_queries(gpt_problem, n=12)
    exact = ExactInfluence(task, damping=1e-3).fit(*train).score(*queries)

    def error(iterations):
        scores = (
            CGInfluence(task, damping=1e-3, tol=1e-14, max_iterations=iterations)
            .fit(*train)
            .score(*queries)
        )
        return (scores - exact).norm().item()

    assert error(40) < error(10) < error(2)


def badly_scaled_problem():
    """A classifier whose inputs span two orders of magnitude in scale."""
    x, y = gaussian_blobs(206, dim=10, classes=3)
    x = x * torch.logspace(-1, 1, 10)
    task = Task(MLP([10, 16, 3]))
    return task, (x[6:], y[6:]), (x[:6], y[:6])


def test_ekfac_preconditioning_saves_iterations_where_the_curvature_is_nearly_full_rank():
    task, train, queries = badly_scaled_problem()
    options = {"damping": 1e-4, "tol": 1e-8, "max_iterations": 3000}
    plain = CGInfluence(task, **options).fit(*train)
    helped = CGInfluence(task, precondition=True, **options).fit(*train)
    plain_scores, helped_scores = plain.score(*queries), helped.score(*queries)
    assert plain.info.converged and helped.info.converged
    assert helped.info.iterations < 0.8 * plain.info.iterations
    assert torch.allclose(helped_scores, plain_scores, atol=1e-5 * plain_scores.abs().max())


def test_preconditioning_does_not_change_what_is_solved_even_where_it_hurts():
    """On the small GPT it costs iterations; see docs/notes/negative-results.md."""
    task, train, queries = problem_with_queries(gpt_problem, n=12)
    options = {"damping": 1e-2, "tol": 1e-9, "max_iterations": 5000}
    plain = CGInfluence(task, **options).fit(*train)
    hindered = CGInfluence(task, precondition=True, **options).fit(*train)
    plain_scores, hindered_scores = plain.score(*queries), hindered.score(*queries)
    assert plain.info.converged and hindered.info.converged
    assert hindered.info.iterations > plain.info.iterations
    assert torch.allclose(hindered_scores, plain_scores, atol=1e-5 * plain_scores.abs().max())


def test_lissa_with_the_full_dataset_converges_to_exact_influence():
    task, train, queries = problem_with_queries()
    exact = ExactInfluence(task, damping=0.1).fit(*train).score(*queries)

    def scores(steps):
        return (
            LiSSAInfluence(task, damping=0.1, steps=steps, sample_size=len(train[0]))
            .fit(*train)
            .score(*queries)
        )

    few, many = scores(5), scores(400)
    assert torch.allclose(many, exact, atol=1e-6 * exact.abs().max())
    assert (many - exact).norm() < (few - exact).norm()


def test_lissa_picks_a_scale_above_the_largest_eigenvalue():
    task, train, _ = problem_with_queries()
    attributor = LiSSAInfluence(task, damping=0.1).fit(*train)
    largest = torch.linalg.eigvalsh(dense_ggn(task, task.params(), *train))[-1].item() + 0.1
    assert largest < attributor.scale < 1.2 * largest


def test_stochastic_lissa_tracks_exact_influence_and_is_seeded():
    task, train, queries = problem_with_queries()
    exact = ExactInfluence(task, damping=0.1).fit(*train).score(*queries)

    def scores(seed):
        return (
            LiSSAInfluence(task, damping=0.1, steps=300, repeats=4, sample_size=8, seed=seed)
            .fit(*train)
            .score(*queries)
        )

    first = scores(0)
    assert correlation(first, exact) > 0.95
    assert torch.equal(first, scores(0))
    assert not torch.equal(first, scores(1))
