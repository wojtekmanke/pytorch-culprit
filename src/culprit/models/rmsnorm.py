"""RMS normalisation as one autograd node with hand-written rules.

`y = x / rms(x) * weight`, with `rms(x) = sqrt(mean(x^2) + eps)` over the
last dimension.

Written with ordinary operations this is five nodes in the autograd graph,
and each keeps a tensor the size of `x` alive for the backward pass. As a
custom `torch.autograd.Function` it is one node that saves only its inputs,
which exist anyway.

In the recorded run on the two-layer test GPT that lowered the peak memory
of a training step by 12% and, on the CPU, made the step 19% slower: each
rule is a call into Python, and the backward rule recomputes the scale. On
the GPU the two times were within 3%. `docs/results/benchmarks.md` has both
rows.

The other price is that nothing comes for free any more. Every transform
this library applies has to be given a rule by hand:

- `backward` for reverse mode;
- `jvp` for forward mode, which Gauss-Newton-vector products and forward
  over reverse Hessian-vector products use;
- `vmap` for per-example gradients and for training many models at once;
- and `backward` and `jvp` have to be differentiable themselves, for the
  second derivative.

The last point decides what is saved. Both rules recompute the scale from
`x` with ordinary operations instead of reusing the one the forward pass
computed. A saved scale would be a constant to whatever differentiates the
rule, and its dependence on `x` would be lost without an error: exactly
what goes wrong in `F.layer_norm` (see `docs/notes/pitfalls.md`).

One composition is out of reach. Forward mode over forward mode treats the
tensors a `jvp` rule reads from its context as constants, for any custom
Function, so the second derivative comes out wrong. Nothing in this library
uses that composition; the pitfalls note has the reproducer.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


def _scale(x: Tensor, eps: float) -> Tensor:
    """`1 / rms(x)`, keeping the last dimension so that it broadcasts against `x`."""
    return torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)


def _project(normalised: Tensor, direction: Tensor, scale: Tensor) -> Tensor:
    """The Jacobian of `x -> x / rms(x)` applied to `direction`.

    The Jacobian is `scale * (I - n n^T / d)` with `n` the normalised
    input: scale, and remove the component along `n`. It is symmetric, so
    forward and reverse mode both use this one function.
    """
    along = (normalised * direction).mean(dim=-1, keepdim=True)
    return scale * (direction - normalised * along)


class RMSNormFunction(torch.autograd.Function):
    """`rms_norm(x, weight, eps)`; `weight` is any shape that broadcasts against `x`."""

    @staticmethod
    def forward(x: Tensor, weight: Tensor, eps: float) -> Tensor:
        return x * _scale(x, eps) * weight

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Tensor) -> None:
        x, weight, eps = inputs
        ctx.eps = eps
        ctx.save_for_backward(x, weight)
        ctx.save_for_forward(x, weight)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor | None, Tensor | None, None]:
        x, weight = ctx.saved_tensors
        scale = _scale(x, ctx.eps)
        normalised = x * scale
        grad_x = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_x = _project(normalised, grad_output * weight, scale)
        if ctx.needs_input_grad[1]:
            grad_weight = (grad_output * normalised).sum_to_size(weight.shape)
        return grad_x, grad_weight, None

    @staticmethod
    def jvp(ctx: Any, tangent_x: Tensor | None, tangent_weight: Tensor | None, _: None) -> Tensor:
        x, weight = ctx.saved_tensors
        scale = _scale(x, ctx.eps)
        normalised = x * scale
        out = torch.zeros_like(x * weight)
        if tangent_x is not None:
            out = out + _project(normalised, tangent_x, scale) * weight
        if tangent_weight is not None:
            out = out + normalised * tangent_weight
        return out

    @staticmethod
    def vmap(
        info: Any, in_dims: tuple[int | None, ...], x: Tensor, weight: Tensor, eps: float
    ) -> tuple[Tensor, int]:
        """Run the whole batch as one call, with the batch as the first dimension.

        The operation acts on the last dimension only, so a batched `x`
        needs nothing but its batch dimension moved to the front. A batched
        `weight`, which is what training several models at once produces,
        is reshaped to `[batch, 1, ..., 1, *weight.shape]` so that each
        model's weight broadcasts against that model's inputs.
        """
        x_dim, weight_dim, _ = in_dims
        x = x.movedim(x_dim, 0) if x_dim is not None else x.unsqueeze(0)
        if weight_dim is not None:
            weight = weight.movedim(weight_dim, 0)
            padding = (1,) * max(0, x.dim() - weight.dim())
            weight = weight.reshape(weight.shape[0], *padding, *weight.shape[1:])
        return RMSNormFunction.apply(x, weight, eps), 0


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    """The fused operation."""
    out: Tensor = RMSNormFunction.apply(x, weight, eps)
    return out


def rms_norm_reference(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    """The same function from ordinary operations, which autograd differentiates itself."""
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * weight


class RMSNorm(nn.Module):
    """Normalise the last dimension to unit root mean square, then scale it.

    `fused=False` computes the same thing from ordinary operations. It
    exists to be compared against.
    """

    def __init__(self, width: int, eps: float = 1e-5, fused: bool = True) -> None:
        super().__init__()
        self.width, self.eps, self.fused = width, eps, fused
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        fn = rms_norm if self.fused else rms_norm_reference
        return fn(x, self.weight, self.eps)

    def normalise(self, x: Tensor) -> Tensor:
        """The input before it is scaled, which is what the weight's gradient pairs with."""
        return x * _scale(x, self.eps)

    def extra_repr(self) -> str:
        return f"{self.width}, eps={self.eps}, fused={self.fused}"
