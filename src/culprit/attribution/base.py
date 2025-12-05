"""What every attribution method has in common."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Self

from torch import Tensor

from culprit.task import Params, Task


class Attributor(ABC):
    """Scores every training example for every query.

    The score of training example `i` for query `q` estimates how much
    *including* `i` in the training set changed the measurement of `q`. With
    the default measurement, the log-probability of the query's target, a
    positive score says the example pushed the model towards that answer.
    For methods derived from a first-order expansion the score has the scale
    of the measurement itself, so scores can be summed over a group of
    examples to predict the effect of removing the group.

    `fit` does the work that depends only on the training data, so that any
    number of queries can be scored afterwards.
    """

    def __init__(self, task: Task, params: Mapping[str, Tensor] | None = None) -> None:
        self.task = task
        self.params: Params = dict(params) if params is not None else task.params()

    @abstractmethod
    def fit(self, inputs: Tensor, targets: Tensor) -> Self:
        """Prepare for scoring against these training examples."""

    @abstractmethod
    def score(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Scores of shape `[queries, training examples]`."""
