"""Optimisers as pure functions of parameters, gradients and state.

`torch.optim` updates tensors in place and keeps its state inside an object.
That is at odds with two things this library does: training many models at
once, where the parameters are stacked along a new first dimension and the
update has to be an ordinary tensor expression, and differentiating through
training, where every update has to be a function that `torch.func` can
transform.

The functions here take dictionaries and return new ones. Each performs the
same floating-point operations in the same order as its `torch.optim`
counterpart, so that a model trained with one can be compared, update for
update, with a model trained with the other.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from culprit.task import Params


@dataclass(frozen=True)
class AdamWState:
    step: int  # updates applied so far
    mean: Params  # running mean of the gradients
    square: Params  # running mean of the squared gradients


def adamw_init(params: Mapping[str, Tensor]) -> AdamWState:
    zeros = {name: torch.zeros_like(value) for name, value in params.items()}
    return AdamWState(step=0, mean=zeros, square={n: v.clone() for n, v in zeros.items()})


def adamw_step(
    params: Mapping[str, Tensor],
    grads: Mapping[str, Tensor],
    state: AdamWState,
    *,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> tuple[Params, AdamWState]:
    """One update of AdamW with decoupled weight decay, as `torch.optim.AdamW` does it.

    Every operation is elementwise, so parameters stacked along a leading
    model dimension are updated as that many independent models.
    """
    beta1, beta2 = betas
    step = state.step + 1
    correction1 = 1 - beta1**step
    correction2_root = (1 - beta2**step) ** 0.5
    new_params, mean, square = {}, {}, {}
    for name, value in params.items():
        grad = grads[name]
        decayed = value * (1 - lr * weight_decay)
        mean[name] = state.mean[name] + (grad - state.mean[name]) * (1 - beta1)
        square[name] = state.square[name] * beta2 + (1 - beta2) * grad * grad
        denominator = square[name].sqrt() / correction2_root + eps
        new_params[name] = decayed - (lr / correction1) * mean[name] / denominator
    return new_params, AdamWState(step=step, mean=mean, square=square)


@dataclass(frozen=True)
class SGDState:
    step: int
    momentum: Params  # empty when the momentum coefficient is zero


def sgd_init(params: Mapping[str, Tensor]) -> SGDState:
    return SGDState(step=0, momentum={n: torch.zeros_like(v) for n, v in params.items()})


def sgd_step(
    params: Mapping[str, Tensor],
    grads: Mapping[str, Tensor],
    state: SGDState,
    *,
    lr: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
) -> tuple[Params, SGDState]:
    """One update of SGD with L2 weight decay and momentum, as `torch.optim.SGD` does it."""
    new_params, buffers = {}, {}
    for name, value in params.items():
        grad = grads[name] + weight_decay * value if weight_decay else grads[name]
        if momentum:
            # torch.optim.SGD starts the buffer at the first gradient.
            grad = grad if state.step == 0 else momentum * state.momentum[name] + grad
        buffers[name] = grad
        new_params[name] = value - lr * grad
    return new_params, SGDState(step=state.step + 1, momentum=buffers)
