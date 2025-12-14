import pytest
import torch

from culprit.attribution import TracIn
from culprit.data.blobs import gaussian_blobs
from culprit.flat import Layout
from culprit.models import MLP
from culprit.task import Task
from culprit.train import TrainConfig, train
from helpers import gpt_problem, loop_grads, mlp_problem


def test_without_checkpoints_the_score_is_minus_the_gradient_dot_product():
    task, x, y = gpt_problem()
    scores = TracIn(task).fit(x, y).score(x[:2], y[:2])
    layout = Layout.of(task.params())
    train_grads = layout.flatten(loop_grads(task, x, y))
    query_grads = layout.flatten(loop_grads(task, x[:2], y[:2], measure=True))
    assert scores.shape == (2, len(x))
    assert torch.allclose(scores, -query_grads @ train_grads.T, atol=1e-12)


def test_checkpoints_are_weighted_by_their_learning_rate():
    x, y = gaussian_blobs(40, dim=6, classes=3)
    task = Task(MLP([6, 8, 3]))
    run = train(task, x, y, TrainConfig(epochs=3, batch_size=10, lr=0.05, checkpoint_every=4))
    scores = TracIn(task, run.checkpoints, batch_size=16).fit(x, y).score(x[:5], y[:5])

    expected = torch.zeros(5, 40)
    layout = Layout.of(task.params())
    for checkpoint in run.checkpoints:
        with torch.no_grad():
            for name, param in task.model.named_parameters():
                param.copy_(checkpoint.params[name])
        train_grads = layout.flatten(loop_grads(task, x, y))
        query_grads = layout.flatten(loop_grads(task, x[:5], y[:5], measure=True))
        expected -= checkpoint.lr * query_grads @ train_grads.T
    assert len(run.checkpoints) > 1
    assert torch.allclose(scores, expected, atol=1e-12)


def test_an_example_supports_itself():
    task, x, y = mlp_problem()
    scores = TracIn(task).fit(x, y).score(x, y)
    assert bool((scores.diagonal() > 0).all())


def test_scoring_before_fitting_is_an_error():
    task, x, y = mlp_problem()
    with pytest.raises(RuntimeError, match="call fit"):
        TracIn(task).score(x, y)
