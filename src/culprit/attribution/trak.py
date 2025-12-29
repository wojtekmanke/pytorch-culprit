"""TRAK: influence in a random low-dimensional projection of gradient space."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal, Self

import torch
from torch import Tensor

from culprit.attribution.base import Attributor
from culprit.batching import batches
from culprit.device import cpu_double
from culprit.flat import Layout
from culprit.grads import gradient_matrix
from culprit.projection import Projector, SparseProjector
from culprit.task import Params, Task

ProjectorFactory = Callable[[int, int], Projector]


class TRAK(Attributor):
    """Projected gradients, a kernel inverse, and an average over models.

    For each model, with `phi` the randomly projected gradient of the
    measurement and `psi` the projected gradient of the loss,

        score(q, i)  =  -1/n * phi(q)^T (K / n + damping I)^-1 psi(i),

    where `K` is the sum over training examples of `phi phi^T`. The scores
    of several models are averaged. The models can be checkpoints of one
    run or independently trained copies; independent copies average out
    more of the noise of training.

    This is the estimator of Park et al. (2023) in a form that also covers
    examples with many predictions. For a classifier with the margin as the
    measurement, the gradient of the loss is `-(1 - p)` times the gradient
    of the margin, and the expression above is their
    `phi(q)^T K^-1 phi(i) (1 - p_i)` exactly. Two things differ from the
    paper: the scores of the models are averaged whole, where the paper
    averages the kernel part and the `1 - p` part separately, and there is
    no soft thresholding.

    `kernel` chooses what `K` is built from. `"output"` uses the gradients
    of the measurement, as the paper does. That leaves out how sharply the
    loss curves in the output, so the scores rank training examples but are
    not in the units of the measurement. `"loss"` uses the gradients of the
    loss, which makes `K / n` the empirical Fisher in the projected space.

    `damping` is relative to the trace of `K / n`, which is the mean squared
    norm of the projected gradients. The scale of `K` depends on the kernel,
    so an absolute value would mean nothing; and unlike the mean eigenvalue,
    the trace does not change with the size of the projection, so neither
    does the meaning of a given `damping`.
    """

    def __init__(
        self,
        task: Task,
        checkpoints: Sequence[Mapping[str, Tensor]] | None = None,
        *,
        projection_size: int = 2048,
        damping: float = 1e-3,
        kernel: Literal["output", "loss"] = "output",
        seed: int = 0,
        batch_size: int | None = None,
        projector: ProjectorFactory | None = None,
    ) -> None:
        super().__init__(task)
        self.checkpoints: list[Params] = (
            [self.params]
            if checkpoints is None
            else [
                {name: params[name].to(value) for name, value in self.params.items()}
                for params in checkpoints
            ]
        )
        self.damping, self.kernel, self.batch_size = damping, kernel, batch_size
        size = Layout.of(self.params).size
        make = projector or (
            lambda in_size, seed: SparseProjector(in_size, projection_size, seed=seed)
        )
        # A different projection for each model, so that their errors average out.
        self.projectors = [make(size, seed + index) for index in range(len(self.checkpoints))]
        self._solved: list[Tensor] = []  # per model: (K / n + damping I)^-1 Psi^T, [k, n]
        self._count = 0

    def _features(self, model: int, inputs: Tensor, targets: Tensor, *, measure: bool) -> Tensor:
        """Projected per-example gradients under one model, `[examples, k]` on the CPU."""
        rows = [
            cpu_double(
                self.projectors[model].project(
                    gradient_matrix(self.task, self.checkpoints[model], x, y, measure=measure)
                )
            )
            for x, y in batches(inputs, targets, self.batch_size)
        ]
        return torch.cat(rows)

    def fit(self, inputs: Tensor, targets: Tensor) -> Self:
        self._count = len(inputs)
        self._solved = []
        for model in range(len(self.checkpoints)):
            psi = self._features(model, inputs, targets, measure=False)
            phi = (
                psi
                if self.kernel == "loss"
                else self._features(model, inputs, targets, measure=True)
            )
            kernel = phi.T @ phi / self._count
            ridge = self.damping * torch.trace(kernel)
            kernel = kernel + ridge * torch.eye(kernel.shape[0], dtype=kernel.dtype)
            self._solved.append(torch.linalg.solve(kernel, psi.T))
        return self

    def score(self, inputs: Tensor, targets: Tensor) -> Tensor:
        if not self._solved:
            raise RuntimeError("call fit with the training examples before score")
        total = torch.zeros(len(inputs), self._count, dtype=torch.float64)
        for model, solved in enumerate(self._solved):
            total -= self._features(model, inputs, targets, measure=True) @ solved
        return total / (self._count * len(self._solved))
