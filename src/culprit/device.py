"""Moving numbers between the GPU and double precision without losing them."""

from __future__ import annotations

from torch import Tensor


def cpu_double(tensor: Tensor) -> Tensor:
    """A float64 copy on the CPU.

    The move and the widening are two calls on purpose. MPS has no float64,
    and on torch 2.14 the single call `tensor.to("cpu", torch.float64)` on an
    MPS tensor returns zeros or uninitialised memory instead of raising. See
    `docs/notes/pitfalls.md`.
    """
    return tensor.cpu().double()
