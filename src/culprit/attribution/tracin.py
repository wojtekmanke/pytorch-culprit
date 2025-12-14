"""TracIn: add up gradient dot products along the training trajectory."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import torch
from torch import Tensor

from culprit.attribution.base import Attributor
from culprit.batching import batches
from culprit.device import cpu_double
from culprit.grads.ghost import capture, gradient_dots
from culprit.task import Task
from culprit.train import Checkpoint


class TracIn(Attributor):
    """The checkpoint form of TracIn (Pruthi et al., 2020).

    A gradient step on a training example moves the parameters against that
    example's gradient, which changes a measurement of a query by minus the
    learning rate times the dot product of the two gradients. Adding that
    up over the steps that used the example would trace its whole effect on
    the query. Checkpoints stand in for the steps:

        score(q, i)  =  -sum over checkpoints of  lr * <grad m(q), grad loss(i)>

    There is no curvature, so this needs no second pass and no inverse, and
    the dot products are taken layer by layer without forming a gradient.
    The price is that the scores are not in the units of the measurement:
    they rank training examples for a query, and should not be summed to
    predict the size of an effect.

    Without `checkpoints` it uses the task's current parameters once, with a
    learning rate of one, which is the plain gradient dot product.
    """

    def __init__(
        self,
        task: Task,
        checkpoints: Sequence[Checkpoint] | None = None,
        *,
        batch_size: int | None = None,
    ) -> None:
        super().__init__(task)
        if checkpoints is None:
            checkpoints = [Checkpoint(step=0, lr=1.0, params=self.params)]
        self.checkpoints = list(checkpoints)
        self.batch_size = batch_size
        self._train: tuple[Tensor, Tensor] | None = None

    def fit(self, inputs: Tensor, targets: Tensor) -> Self:
        self._train = (inputs, targets)
        return self

    def score(self, inputs: Tensor, targets: Tensor) -> Tensor:
        if self._train is None:
            raise RuntimeError("call fit with the training examples before score")
        total = torch.zeros(len(inputs), len(self._train[0]), dtype=torch.float64)
        for checkpoint in self.checkpoints:
            # Checkpoints read back from disk are on the CPU; the model may not be.
            params = {
                name: checkpoint.params[name].to(value) for name, value in self.params.items()
            }
            queries = capture(self.task, params, inputs, targets, measure=True)
            columns = [
                gradient_dots(self.task, params, queries, capture(self.task, params, x, y))
                for x, y in batches(*self._train, self.batch_size)
            ]
            total -= checkpoint.lr * cpu_double(torch.cat(columns, dim=1))
        return total
