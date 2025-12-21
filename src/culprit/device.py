"""Moving numbers between the GPU and double precision without losing them."""

from __future__ import annotations

import torch
from torch import Tensor


def cpu_double(tensor: Tensor) -> Tensor:
    """A float64 copy on the CPU.

    The move and the widening are two calls on purpose. MPS has no float64,
    and on torch 2.14 the single call `tensor.to("cpu", torch.float64)` on an
    MPS tensor returns zeros or uninitialised memory instead of raising. See
    `docs/notes/pitfalls.md`.
    """
    return tensor.cpu().double()


def solver_tolerance(dtype: torch.dtype) -> float:
    """The relative residual an iterative solve can be asked for in this precision.

    In float32 the residual that conjugate gradient tracks keeps shrinking
    after the true residual has stopped: asked for 1e-6, it stopped with a
    true residual of up to 7e-6 on a well-damped problem. Asking for less
    than this only buys iterations. See `docs/notes/precision.md`.
    """
    return 1e-8 if dtype == torch.float64 else 1e-5
