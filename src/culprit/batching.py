"""Walking over a dataset that is held as a pair of tensors."""

from __future__ import annotations

from collections.abc import Iterator

from torch import Tensor


def batches(
    inputs: Tensor, targets: Tensor, batch_size: int | None
) -> Iterator[tuple[Tensor, Tensor]]:
    """Consecutive slices of at most `batch_size` examples; everything at once if `None`."""
    size = batch_size or len(inputs)
    for start in range(0, len(inputs), size):
        yield inputs[start : start + size], targets[start : start + size]
