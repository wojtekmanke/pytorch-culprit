"""Influence functions with the curvature matrix inverted exactly."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self

import torch
from torch import Tensor

from culprit.attribution.base import Attributor
from culprit.curvature import dense_ggn, dense_hessian
from culprit.device import cpu_double
from culprit.flat import Layout
from culprit.grads import gradient_matrix
from culprit.task import Task


class ExactInfluence(Attributor):
    """The reference the approximations are measured against.

    Down-weighting one training example moves the minimiser of the training
    loss by the inverse curvature times that example's gradient. To first
    order, including example `i` among `n` therefore changes a measurement
    `m` of query `q` by

        -1/n * grad m(q)^T (C + damping I)^-1 grad loss(i)

    where `C` is the curvature of the mean training loss. Here `C` is formed
    as a dense matrix and solved against directly, which limits this class
    to models with a few thousand parameters.

    `damping` plays the part of an L2 penalty: if the model was trained with
    one, passing its coefficient makes `C + damping I` the true Hessian of
    the training objective. The linear algebra is done in double precision
    on the CPU wherever the model lives.
    """

    def __init__(
        self,
        task: Task,
        params: Mapping[str, Tensor] | None = None,
        *,
        damping: float = 1e-3,
        curvature: Literal["ggn", "hessian"] = "ggn",
        batch_size: int | None = None,
    ) -> None:
        super().__init__(task, params)
        self.damping, self.curvature, self.batch_size = damping, curvature, batch_size
        self.layout = Layout.of(self.params)
        self._solved: Tensor | None = None  # (C + damping I)^-1 G_train^T, [size, n]
        self._count = 0

    def fit(self, inputs: Tensor, targets: Tensor) -> Self:
        dense = dense_ggn if self.curvature == "ggn" else dense_hessian
        matrix = cpu_double(
            dense(self.task, self.params, inputs, targets, batch_size=self.batch_size)
        )
        matrix = matrix + self.damping * torch.eye(self.layout.size, dtype=torch.float64)
        grads = cpu_double(
            gradient_matrix(self.task, self.params, inputs, targets, batch_size=self.batch_size)
        )
        self._solved = torch.linalg.solve(matrix, grads.T)
        self._count = len(inputs)
        return self

    def score(self, inputs: Tensor, targets: Tensor) -> Tensor:
        if self._solved is None:
            raise RuntimeError("call fit with the training examples before score")
        queries = cpu_double(
            gradient_matrix(
                self.task, self.params, inputs, targets, measure=True, batch_size=self.batch_size
            )
        )
        return -(queries @ self._solved) / self._count
