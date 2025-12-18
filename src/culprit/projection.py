"""Random projections of gradients to a few thousand dimensions.

A random projection approximately preserves inner products: if `P` has
independent entries of mean zero and variance `1 / out_size`, then
`<P g, P g'>` is an unbiased estimate of `<g, g'>` whose error shrinks like
`1 / sqrt(out_size)` (Johnson and Lindenstrauss). Methods that only need
inner products between gradients can therefore keep `out_size` numbers per
example where the gradient has millions.

The projection matrix itself has `in_size * out_size` entries, more than
the gradients it is meant to compress. Neither projector here ever holds
it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Projector(ABC):
    in_size: int
    out_size: int

    @abstractmethod
    def project(self, matrix: Tensor) -> Tensor:
        """Map rows of `[n, in_size]` to rows of `[n, out_size]`."""

    def dense(self, dtype: torch.dtype = torch.float64) -> Tensor:
        """The projection matrix `[in_size, out_size]`, for tests and tiny sizes."""
        return self.project(torch.eye(self.in_size, dtype=dtype))


class SparseProjector(Projector):
    """A sparse Johnson-Lindenstrauss transform (Kane and Nelson, 2014).

    The output is split into `nonzeros` equal blocks. Every input coordinate
    is sent to one random bucket in each block with a random sign, so each
    row of the projection matrix has exactly `nonzeros` entries of
    `+-1 / sqrt(nonzeros)`. Projecting is then a scatter-add: it costs
    `nonzeros` operations per input coordinate, whatever `out_size` is, and
    what is stored is one bucket and one sign per coordinate and block.

    With `nonzeros=1` this is a count sketch. A few more reduce the variance
    for gradients whose mass sits in a few coordinates.
    """

    def __init__(self, in_size: int, out_size: int, *, nonzeros: int = 4, seed: int = 0) -> None:
        if out_size % nonzeros:
            raise ValueError(f"out_size {out_size} is not a multiple of nonzeros {nonzeros}")
        self.in_size, self.out_size, self.nonzeros = in_size, out_size, nonzeros
        generator = torch.Generator().manual_seed(seed)
        block = out_size // nonzeros
        offsets = torch.arange(nonzeros).unsqueeze(1) * block
        self._buckets = torch.randint(block, (nonzeros, in_size), generator=generator) + offsets
        signs = torch.randint(2, (nonzeros, in_size), generator=generator) * 2 - 1
        self._signs = signs.to(torch.int8)

    def project(self, matrix: Tensor) -> Tensor:
        if matrix.shape[-1] != self.in_size:
            raise ValueError(f"expected {self.in_size} columns, got {matrix.shape[-1]}")
        out = matrix.new_zeros(matrix.shape[0], self.out_size)
        buckets = self._buckets.to(matrix.device)
        signs = self._signs.to(matrix.device, matrix.dtype)
        for block in range(self.nonzeros):
            out.index_add_(1, buckets[block], matrix * signs[block])
        return out / self.nonzeros**0.5


class RademacherProjector(Projector):
    """A dense matrix of `+-1 / sqrt(out_size)`, generated a slab at a time.

    This is the textbook projection and the reference the sparse one is
    compared with. The matrix is a function of the seed: every call
    regenerates it slab by slab, uses each slab once and drops it, so memory
    stays at `slab * out_size` however large `in_size` is. That costs
    `in_size * out_size` random numbers per call, which is why it is the
    reference and not the default.
    """

    def __init__(self, in_size: int, out_size: int, *, seed: int = 0, slab: int = 4096) -> None:
        self.in_size, self.out_size, self.seed, self.slab = in_size, out_size, seed, slab

    def _slab(self, index: int, rows: int, like: Tensor) -> Tensor:
        # Each slab has its own generator, so a slab can be produced without
        # generating the ones before it.
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + index)
        signs = torch.randint(2, (rows, self.out_size), generator=generator) * 2 - 1
        return signs.to(like.device, like.dtype)

    def project(self, matrix: Tensor) -> Tensor:
        if matrix.shape[-1] != self.in_size:
            raise ValueError(f"expected {self.in_size} columns, got {matrix.shape[-1]}")
        out = matrix.new_zeros(matrix.shape[0], self.out_size)
        for index, start in enumerate(range(0, self.in_size, self.slab)):
            columns = matrix[:, start : start + self.slab]
            out += columns @ self._slab(index, columns.shape[1], matrix)
        return out / self.out_size**0.5
