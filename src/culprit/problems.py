"""Named problems: a model, its training set, queries and how to train it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import Tensor

from culprit.data.blobs import gaussian_blobs
from culprit.models import MLP
from culprit.task import Task
from culprit.train import TrainConfig


@dataclass(frozen=True)
class Problem:
    name: str
    task: Task
    train: tuple[Tensor, Tensor]
    queries: tuple[Tensor, Tensor]
    config: TrainConfig


def blobs(seed: int = 0) -> Problem:
    """A small classifier on overlapping Gaussian clusters: runs anywhere in seconds."""
    x, y = gaussian_blobs(600, dim=10, classes=3, spread=1.5, seed=seed)
    return Problem(
        name="blobs",
        task=Task(MLP([10, 16, 3], seed=seed)),
        train=(x[:500], y[:500]),
        queries=(x[500:], y[500:]),
        config=TrainConfig(epochs=30, batch_size=50, lr=0.01, checkpoint_every=60, seed=seed),
    )


PROBLEMS: dict[str, Callable[[int], Problem]] = {"blobs": blobs}
