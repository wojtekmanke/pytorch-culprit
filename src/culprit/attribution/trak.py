"""TRAK: influence in a random low-dimensional projection of gradient space."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, Self

import torch
from torch import Tensor

from culprit.attribution.base import Attributor
from culprit.batching import batches
from culprit.device import cpu_double
from culprit.flat import Layout
from culprit.grads import gradient_matrix
from culprit.projection import Projector, SparseProjector
from culprit.store import GradientStore, digest
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

    The projected gradients of the training examples are the expensive part.
    With `store` they are written to that directory one batch at a time, so
    an interrupted `fit` resumes at the batch it stopped at, and a second
    `fit` with the same models, data and seed reads them back. Scoring
    streams them from there, so they are never all in memory.
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
        store: Path | None = None,
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
        self.seed, self.store = seed, store
        size = Layout.of(self.params).size
        make = projector or (
            lambda in_size, seed: SparseProjector(in_size, projection_size, seed=seed)
        )
        # A different projection for each model, so that their errors average out.
        self.projectors = [make(size, seed + index) for index in range(len(self.checkpoints))]
        self._factors: list[Tensor] = []  # per model: Cholesky factor of K / n + ridge
        self._psi: list[GradientStore] = []  # per model: projected loss gradients
        self._count = 0

    def _project(self, model: int, inputs: Tensor, targets: Tensor, *, measure: bool) -> Tensor:
        """Projected per-example gradients under one model, on the CPU."""
        grads = gradient_matrix(
            self.task, self.checkpoints[model], inputs, targets, measure=measure
        )
        return self.projectors[model].project(grads).cpu()

    def _train_features(
        self, model: int, inputs: Tensor, targets: Tensor, *, measure: bool
    ) -> GradientStore:
        """The projected gradients of every training example, computed or read back."""
        kind = "output" if measure else "loss"
        params = self.checkpoints[model]
        store = GradientStore(
            None if self.store is None else self.store / f"model-{model}" / kind,
            rows=len(inputs),
            width=self.projectors[model].out_size,
            shard_rows=self.batch_size or len(inputs),
            fingerprint={
                "seed": self.seed + model,
                "kind": kind,
                "parameters": digest(*params.values()),
                "examples": digest(inputs, targets),
            },
        )
        for index in store.missing():
            rows = store.shard_range(index)
            x, y = inputs[rows.start : rows.stop], targets[rows.start : rows.stop]
            store.write(index, self._project(model, x, y, measure=measure))
        return store

    def fit(self, inputs: Tensor, targets: Tensor) -> Self:
        self._count = len(inputs)
        self._factors, self._psi = [], []
        for model in range(len(self.checkpoints)):
            psi = self._train_features(model, inputs, targets, measure=False)
            phi = (
                psi
                if self.kernel == "loss"
                else self._train_features(model, inputs, targets, measure=True)
            )
            width = phi.width
            kernel = torch.zeros(width, width, dtype=torch.float64)
            for shard in phi:
                rows = shard.double()
                kernel += rows.T @ rows
            kernel /= self._count
            ridge = self.damping * torch.trace(kernel)
            kernel += ridge * torch.eye(width, dtype=torch.float64)
            self._factors.append(torch.linalg.cholesky(kernel))
            self._psi.append(psi)
        return self

    def score(self, inputs: Tensor, targets: Tensor) -> Tensor:
        if not self._factors:
            raise RuntimeError("call fit with the training examples before score")
        total = torch.zeros(len(inputs), self._count, dtype=torch.float64)
        for model, factor in enumerate(self._factors):
            queries = torch.cat(
                [
                    cpu_double(self._project(model, x, y, measure=True))
                    for x, y in batches(inputs, targets, self.batch_size)
                ]
            )
            # Solve against the few queries once, then walk the stored rows.
            solved = torch.cholesky_solve(queries.T, factor).T
            start = 0
            for shard in self._psi[model]:
                total[:, start : start + len(shard)] -= solved @ shard.double().T
                start += len(shard)
        return total / (self._count * len(self._factors))
