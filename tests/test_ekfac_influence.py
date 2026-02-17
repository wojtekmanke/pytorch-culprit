import pytest
import torch

from culprit.attribution import EKFACInfluence, ExactInfluence
from culprit.data.blobs import gaussian_blobs
from culprit.flat import Layout
from culprit.models import MLP
from culprit.task import Task
from helpers import gpt_problem, loop_grads, mlp_problem


def correlation(a, b):
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()


def test_scores_are_the_formula_with_the_factored_inverse():
    task, x, y = gpt_problem()
    queries = (x[:2], y[:2])
    attributor = EKFACInfluence(task, damping=0.05, samples=2).fit(x, y)
    scores = attributor.score(*queries)

    layout = Layout.of(task.params())
    train_grads = layout.flatten(loop_grads(task, x, y))
    query_grads = layout.flatten(loop_grads(task, *queries, measure=True))
    damped = attributor.curvature.dense() + 0.05 * torch.eye(layout.size)
    expected = -query_grads @ torch.linalg.inv(damped) @ train_grads.T / len(x)
    assert scores.shape == (2, len(x))
    assert torch.allclose(scores, expected, atol=1e-10)


def test_scores_equal_exact_influence_where_kfac_is_exact():
    x, y = gaussian_blobs(1, dim=5, classes=3)
    queries = gaussian_blobs(4, dim=5, classes=3, seed=1)
    task = Task(MLP([5, 3]))
    exact = ExactInfluence(task, damping=0.1).fit(x, y).score(*queries)
    approximate = EKFACInfluence(task, damping=0.1, fisher="exact").fit(x, y).score(*queries)
    assert torch.allclose(approximate, exact, atol=1e-12)


def test_scores_track_exact_influence_on_a_small_network():
    task, x, y = mlp_problem(n=60)
    queries = gaussian_blobs(10, dim=5, classes=3, seed=5)
    exact = ExactInfluence(task, damping=0.01).fit(x, y).score(*queries)
    corrected = EKFACInfluence(task, damping=0.01, fisher="exact").fit(x, y).score(*queries)
    plain = (
        EKFACInfluence(task, damping=0.01, fisher="exact", eigenvalue_correction=False)
        .fit(x, y)
        .score(*queries)
    )
    assert correlation(corrected, exact) > 0.9
    assert correlation(corrected, exact) >= correlation(plain, exact) - 0.02


def test_batching_does_not_change_the_scores():
    task, x, y = gpt_problem(n=5)
    whole = EKFACInfluence(task, fisher="empirical").fit(x, y).score(x[:2], y[:2])
    batched = EKFACInfluence(task, fisher="empirical", batch_size=2).fit(x, y).score(x[:2], y[:2])
    assert torch.allclose(whole, batched, atol=1e-10)


def test_scoring_before_fitting_is_an_error():
    task, x, y = mlp_problem()
    with pytest.raises(RuntimeError, match="call fit"):
        EKFACInfluence(task).score(x, y)


def test_scores_come_back_in_double_precision_on_the_cpu():
    task, x, y = mlp_problem()
    scores = EKFACInfluence(task).fit(x, y).score(x[:3], y[:3])
    assert scores.dtype == torch.float64 and scores.device.type == "cpu"


@pytest.mark.mps
def test_scores_on_the_gpu_match_the_cpu():
    task, x, y = gpt_problem()
    reference = EKFACInfluence(task, damping=0.05, fisher="empirical").fit(x, y).score(x, y)
    task.model.to("mps", torch.float32)
    x, y = x.to("mps"), y.to("mps")
    scores = EKFACInfluence(task, damping=0.05, fisher="empirical").fit(x, y).score(x, y)
    assert (scores - reference).abs().max() < 1e-3 * reference.abs().max()


def test_damping_can_be_changed_without_fitting_again():
    task, x, y = gpt_problem()
    fitted = EKFACInfluence(task, damping=0.5, fisher="empirical").fit(x, y)
    refit = EKFACInfluence(task, damping=0.05, fisher="empirical").fit(x, y).score(x[:2], y[:2])
    assert torch.allclose(fitted.set_damping(0.05).score(x[:2], y[:2]), refit, atol=1e-12)
    dense = fitted.curvature.dense()
    assert fitted.mean_eigenvalue() == pytest.approx(torch.trace(dense).item() / len(dense))
    with pytest.raises(RuntimeError, match="call fit"):
        EKFACInfluence(task).set_damping(1.0)
