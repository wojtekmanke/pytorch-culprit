"""Named problems: a model, its training set, queries and how to train it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from culprit.data.blobs import gaussian_blobs
from culprit.data.mnist import mnist
from culprit.data.planted import PlantedFactsConfig, planted_facts
from culprit.losses import Margin
from culprit.models import GPT, MLP, GPTConfig
from culprit.task import Task
from culprit.train import TrainConfig


@dataclass(frozen=True)
class Problem:
    name: str
    task: Task
    train: tuple[Tensor, Tensor]
    queries: tuple[Tensor, Tensor]
    config: TrainConfig
    # Where it is known by construction: `[queries, examples]`, true for the
    # training examples that are responsible for each query.
    sources: Tensor | None = None


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


def _digits(name: str, seed: int, config: TrainConfig) -> Problem:
    images, labels = mnist()
    order = torch.randperm(len(images), generator=torch.Generator().manual_seed(seed))
    train, queries = order[:5000], order[5000:5200]
    return Problem(
        name=name,
        # The margin, not the log-probability: most digits are classified
        # with a probability near one, where the log-probability stops
        # moving and has nothing left to attribute.
        task=Task(MLP([196, 32, 32, 10], seed=seed), measure=Margin()),
        train=(images[train], labels[train]),
        queries=(images[queries], labels[queries]),
        config=config,
    )


def digits(seed: int = 0) -> Problem:
    """MNIST at 14 by 14: 5,000 training digits, 200 queries, 7,690 parameters.

    Small enough that the exact influence of every training example can be
    computed, and real enough that the examples are not interchangeable.
    The subset is drawn with the seed; the same seed gives the same digits.
    """
    config = TrainConfig(
        epochs=20, batch_size=64, lr=3e-3, weight_decay=1e-3, checkpoint_every=312, seed=seed
    )
    return _digits("digits", seed, config)


def digits_sgd(seed: int = 0) -> Problem:
    """The same digits and the same model, trained with SGD and momentum.

    It exists to separate what a result owes to the method from what it
    owes to the optimiser the model happened to be trained with.
    """
    config = TrainConfig(
        epochs=20,
        batch_size=64,
        lr=0.05,
        optimizer="sgd",
        momentum=0.9,
        weight_decay=1e-3,
        checkpoint_every=312,
        seed=seed,
    )
    return _digits("digits-sgd", seed, config)


def facts(seed: int = 0) -> Problem:
    """A two-layer GPT on the planted-facts corpus.

    The training examples are the 800 documents, the queries are the 160
    facts, each asked in the first template, and the measurement is the
    margin of the object token. Which documents state each fact is known,
    so a method can be asked for them directly as well as graded against
    retraining.
    """
    corpus = planted_facts(PlantedFactsConfig(seed=seed))
    config = GPTConfig(
        vocab_size=corpus.vocab_size, block_size=corpus.config.block_size, d_model=64
    )
    return Problem(
        name="facts",
        task=Task(GPT(config, seed=seed), measure=Margin()),
        train=corpus.train_batch(),
        queries=corpus.query_batch(),
        config=TrainConfig(
            epochs=60,
            batch_size=32,
            lr=3e-3,
            weight_decay=1e-2,
            warmup_steps=50,
            checkpoint_every=300,
            seed=seed,
        ),
        sources=corpus.source_mask(),
    )


PROBLEMS: dict[str, Callable[[int], Problem]] = {
    "blobs": blobs,
    "digits": digits,
    "digits-sgd": digits_sgd,
    "facts": facts,
}
