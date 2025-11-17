"""Parameters as one flat vector, and back.

The transforms in `torch.func` work on dictionaries of tensors; linear
algebra works on vectors. A `Layout` fixes the order of the parameters once
so that the two views can be converted without the caller tracking offsets.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from culprit.task import Params


@dataclass(frozen=True)
class Layout:
    names: tuple[str, ...]
    shapes: tuple[torch.Size, ...]

    @classmethod
    def of(cls, params: Mapping[str, Tensor]) -> Layout:
        return cls(tuple(params), tuple(value.shape for value in params.values()))

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(math.prod(shape) for shape in self.shapes)

    @property
    def size(self) -> int:
        return sum(self.sizes)

    def slices(self) -> dict[str, slice]:
        """The range of the flat vector that each parameter occupies."""
        out, start = {}, 0
        for name, size in zip(self.names, self.sizes, strict=True):
            out[name] = slice(start, start + size)
            start += size
        return out

    def flatten(self, tree: Mapping[str, Tensor]) -> Tensor:
        """Concatenate the parameters, keeping any leading batch dimensions.

        A dictionary of per-example gradients with values `[batch, *shape]`
        becomes a matrix `[batch, size]`.
        """
        parts = []
        for name, shape in zip(self.names, self.shapes, strict=True):
            value = tree[name]
            batch_dims = value.dim() - len(shape)
            parts.append(value.reshape(*value.shape[:batch_dims], -1))
        return torch.cat(parts, dim=-1)

    def unflatten(self, vector: Tensor) -> Params:
        """Inverse of `flatten`, for a vector with any leading dimensions."""
        if vector.shape[-1] != self.size:
            raise ValueError(f"expected a last dimension of {self.size}, got {vector.shape[-1]}")
        parts = vector.split(list(self.sizes), dim=-1)
        return {
            name: part.reshape(*vector.shape[:-1], *shape)
            for name, shape, part in zip(self.names, self.shapes, parts, strict=True)
        }
