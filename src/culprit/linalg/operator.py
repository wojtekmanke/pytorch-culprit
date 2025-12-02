"""Symmetric linear maps on the flat parameter vector, known through their products.

A curvature matrix of a model with a million parameters has a trillion
entries, but multiplying it by a vector costs about as much as a gradient.
Solvers that only ever multiply (`culprit.linalg.solvers`) can therefore work
with it. An operator is the interface between the two: something with a
size, a `matvec`, and optionally a faster `matmat` for several vectors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Literal

import torch
from torch import Tensor
from torch.func import vmap

from culprit.batching import batches
from culprit.curvature.products import ggn_vp, hvp_forward_over_reverse
from culprit.flat import Layout
from culprit.task import Task


class LinearOperator(ABC):
    """A symmetric matrix of shape `[size, size]` that may never be formed."""

    size: int
    dtype: torch.dtype
    device: torch.device

    @abstractmethod
    def matvec(self, vector: Tensor) -> Tensor:
        """The product with one vector of shape `[size]`."""

    def matmat(self, matrix: Tensor) -> Tensor:
        """The product with each column of a `[size, k]` matrix.

        The default applies `matvec` column by column. Operators that can do
        better override it.
        """
        return torch.stack([self.matvec(column) for column in matrix.unbind(1)], dim=1)

    def __matmul__(self, other: Tensor) -> Tensor:
        if other.shape[0] != self.size or other.dim() not in (1, 2):
            raise ValueError(
                f"cannot apply an operator of size {self.size} to shape {tuple(other.shape)}"
            )
        return self.matvec(other) if other.dim() == 1 else self.matmat(other)

    def __add__(self, other: LinearOperator) -> LinearOperator:
        return Sum(self, other)

    def damped(self, damping: float) -> LinearOperator:
        """This operator plus `damping` times the identity."""
        return Damped(self, damping)

    def dense(self) -> Tensor:
        """The matrix itself. Only sensible for small sizes, and for tests."""
        return self.matmat(torch.eye(self.size, dtype=self.dtype, device=self.device))

    def _like(self, other: LinearOperator) -> None:
        self.size, self.dtype, self.device = other.size, other.dtype, other.device


class DenseOperator(LinearOperator):
    def __init__(self, matrix: Tensor) -> None:
        if matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"expected a square matrix, got shape {tuple(matrix.shape)}")
        self.matrix = matrix
        self.size, self.dtype, self.device = matrix.shape[0], matrix.dtype, matrix.device

    def matvec(self, vector: Tensor) -> Tensor:
        return self.matrix @ vector

    def matmat(self, matrix: Tensor) -> Tensor:
        return self.matrix @ matrix


class Sum(LinearOperator):
    def __init__(self, first: LinearOperator, second: LinearOperator) -> None:
        if first.size != second.size:
            raise ValueError(f"cannot add operators of sizes {first.size} and {second.size}")
        self.first, self.second = first, second
        self._like(first)

    def matvec(self, vector: Tensor) -> Tensor:
        return self.first.matvec(vector) + self.second.matvec(vector)

    def matmat(self, matrix: Tensor) -> Tensor:
        return self.first.matmat(matrix) + self.second.matmat(matrix)


class Damped(LinearOperator):
    """`operator + damping * I`.

    Damping makes a positive semi-definite matrix positive definite, which is
    what lets it be inverted. It also bounds how far the inverse can amplify
    directions the data says almost nothing about.
    """

    def __init__(self, operator: LinearOperator, damping: float) -> None:
        self.operator, self.damping = operator, damping
        self._like(operator)

    def matvec(self, vector: Tensor) -> Tensor:
        return self.operator.matvec(vector) + self.damping * vector

    def matmat(self, matrix: Tensor) -> Tensor:
        return self.operator.matmat(matrix) + self.damping * matrix


class BlockDiagonal(LinearOperator):
    """One operator per parameter, acting on that parameter's part of the vector.

    This is the structure of every layer-wise curvature approximation: the
    interactions between different layers are dropped, and each block can
    then be stored and inverted on its own.
    """

    def __init__(self, layout: Layout, blocks: Mapping[str, LinearOperator]) -> None:
        if tuple(blocks) != layout.names:
            raise ValueError("blocks must be given for every parameter, in the layout's order")
        for name, size in zip(layout.names, layout.sizes, strict=True):
            if blocks[name].size != size:
                raise ValueError(f"block {name} has size {blocks[name].size}, expected {size}")
        self.layout, self.blocks = layout, dict(blocks)
        self._slices = layout.slices()
        self._like(next(iter(blocks.values())))
        self.size = layout.size

    def matvec(self, vector: Tensor) -> Tensor:
        return torch.cat(
            [block.matvec(vector[self._slices[name]]) for name, block in self.blocks.items()]
        )

    def matmat(self, matrix: Tensor) -> Tensor:
        return torch.cat(
            [block.matmat(matrix[self._slices[name]]) for name, block in self.blocks.items()]
        )


class CurvatureOperator(LinearOperator):
    """The Gauss-Newton matrix or the Hessian of the mean loss over a dataset.

    Each product walks the dataset once in batches of `batch_size`. `matmat`
    runs all of its columns through each batch together with `vmap`, so that
    solving for several right-hand sides costs one pass over the data per
    iteration, not one per right-hand side; `chunk_size` bounds how many
    columns share a pass.
    """

    def __init__(
        self,
        task: Task,
        params: Mapping[str, Tensor],
        inputs: Tensor,
        targets: Tensor,
        *,
        kind: Literal["ggn", "hessian"] = "ggn",
        batch_size: int | None = None,
        chunk_size: int | None = None,
    ) -> None:
        self.task, self.params = task, dict(params)
        self.inputs, self.targets = inputs, targets
        self.batch_size, self.chunk_size = batch_size, chunk_size
        self.layout = Layout.of(self.params)
        self.size = self.layout.size
        first = next(iter(self.params.values()))
        self.dtype, self.device = first.dtype, first.device
        # Forward over reverse, because it is the one way of taking a
        # Hessian-vector product that is right through a layer norm and stays
        # right under the vmap in matmat. See docs/notes/pitfalls.md.
        self._product = ggn_vp if kind == "ggn" else hvp_forward_over_reverse

    def matvec(self, vector: Tensor) -> Tensor:
        direction = self.layout.unflatten(vector)
        total = torch.zeros_like(vector)
        for inputs, targets in batches(self.inputs, self.targets, self.batch_size):
            product = self._product(self.task, self.params, inputs, targets, direction)
            # Not `+=`: under vmap `total` starts without a batch dimension
            # and cannot be written to in place by something that has one.
            total = total + self.layout.flatten(product)
        return total / len(self.inputs)

    def matmat(self, matrix: Tensor) -> Tensor:
        out: Tensor = vmap(self.matvec, in_dims=1, out_dims=1, chunk_size=self.chunk_size)(matrix)
        return out
