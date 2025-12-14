"""Influence functions whose inverse curvature is applied without a dense solve."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import Self

import torch
from torch import Tensor

from culprit.attribution.base import Attributor
from culprit.batching import batches
from culprit.curvature.kfac import FactoredCurvature, FisherKind, SequenceKind, kfac
from culprit.device import cpu_double
from culprit.grads import gradient_matrix
from culprit.task import Task


class InfluenceFunction(Attributor):
    """Everything of the form `-1/n * grad m(q)^T C^-1 grad loss(i)`.

    Subclasses differ only in how they apply the inverse curvature. It is
    applied to the gradients of the queries, of which there are few, and the
    result is then matched against the gradients of the training examples a
    batch at a time, so no matrix with a row per training example and a
    column per parameter is ever held.
    """

    def __init__(
        self,
        task: Task,
        params: Mapping[str, Tensor] | None = None,
        *,
        batch_size: int | None = None,
    ) -> None:
        super().__init__(task, params)
        self.batch_size = batch_size
        self._train: tuple[Tensor, Tensor] | None = None

    def fit(self, inputs: Tensor, targets: Tensor) -> Self:
        self._train = (inputs, targets)
        self._prepare(inputs, targets)
        return self

    @abstractmethod
    def _prepare(self, inputs: Tensor, targets: Tensor) -> None:
        """Build whatever `_solve` needs from the training examples."""

    @abstractmethod
    def _solve(self, rhs: Tensor) -> Tensor:
        """An approximation of `C^-1 @ rhs` for the columns of `rhs`, `[size, k]`."""

    def score(self, inputs: Tensor, targets: Tensor) -> Tensor:
        if self._train is None:
            raise RuntimeError("call fit with the training examples before score")
        queries = gradient_matrix(
            self.task, self.params, inputs, targets, measure=True, batch_size=self.batch_size
        )
        solved = self._solve(queries.T)
        rows = [
            gradient_matrix(self.task, self.params, x, y) @ solved
            for x, y in batches(*self._train, self.batch_size)
        ]
        return -cpu_double(torch.cat(rows).T) / len(self._train[0])


class EKFACInfluence(InfluenceFunction):
    """Influence functions with an EK-FAC approximation to the curvature.

    The inverse is a rotation into the Kronecker eigenbasis of each layer, a
    division, and a rotation back, so after two sweeps over the training
    data to fit the approximation, a query costs about as much as a
    gradient. This is the method of Grosse et al. (2023).

    Set `eigenvalue_correction` to false for plain K-FAC.
    """

    def __init__(
        self,
        task: Task,
        params: Mapping[str, Tensor] | None = None,
        *,
        damping: float = 1e-3,
        eigenvalue_correction: bool = True,
        fisher: FisherKind = "sampled",
        sequence: SequenceKind = "expand",
        samples: int = 1,
        seed: int = 0,
        batch_size: int | None = None,
    ) -> None:
        super().__init__(task, params, batch_size=batch_size)
        self.damping = damping
        self._options = {
            "eigenvalue_correction": eigenvalue_correction,
            "fisher": fisher,
            "sequence": sequence,
            "samples": samples,
            "seed": seed,
        }
        self.curvature: FactoredCurvature | None = None

    def _prepare(self, inputs: Tensor, targets: Tensor) -> None:
        self.curvature = kfac(
            self.task,
            self.params,
            inputs,
            targets,
            batch_size=self.batch_size,
            **self._options,  # type: ignore[arg-type]
        )
        self._inverse = self.curvature.inverse(self.damping)

    def _solve(self, rhs: Tensor) -> Tensor:
        return self._inverse @ rhs
