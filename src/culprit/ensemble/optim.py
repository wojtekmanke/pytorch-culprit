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

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor

from culprit.task import Params
from culprit.train import TrainConfig


def _root(value: Tensor) -> Tensor:
    """The square root, with a derivative of zero instead of infinity at zero.

    A parameter that has never received a gradient has a running square of
    exactly zero. The root is zero there and its derivative is infinite,
    which turns into `0 * inf` as soon as the update is differentiated. The
    value returned is the ordinary root everywhere.
    """
    positive = value > 0
    safe = torch.where(positive, value, torch.ones_like(value))
    return torch.where(positive, safe.sqrt(), torch.zeros_like(value))


@dataclass(frozen=True)
class AdamWState:
    step: int  # updates applied so far
    mean: Params  # running mean of the gradients
    square: Params  # running mean of the squared gradients

    def tensors(self) -> tuple[Params, ...]:
        """The part of the state that an update is a differentiable function of."""
        return (self.mean, self.square)

    def with_tensors(self, tensors: tuple[Params, ...]) -> AdamWState:
        return replace(self, mean=tensors[0], square=tensors[1])


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
        denominator = _root(square[name]) / correction2_root + eps
        new_params[name] = decayed - (lr / correction1) * mean[name] / denominator
    return new_params, AdamWState(step=step, mean=mean, square=square)


@dataclass(frozen=True)
class SGDState:
    step: int
    momentum: Params  # the last update direction; unused when the coefficient is zero

    def tensors(self) -> tuple[Params, ...]:
        return (self.momentum,)

    def with_tensors(self, tensors: tuple[Params, ...]) -> SGDState:
        return replace(self, momentum=tensors[0])


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


@dataclass(frozen=True)
class FunctionalOptimizer:
    """An optimiser with its hyperparameters bound: only the learning rate varies by step."""

    init: Callable[[Mapping[str, Tensor]], Any]
    step: Callable[[Mapping[str, Tensor], Mapping[str, Tensor], Any, float], tuple[Params, Any]]


def optimizer_for(config: TrainConfig) -> FunctionalOptimizer:
    """The functional counterpart of the optimiser `culprit.train.train` would build."""
    if config.optimizer == "adamw":
        return FunctionalOptimizer(
            init=adamw_init,
            step=lambda params, grads, state, lr: adamw_step(
                params, grads, state, lr=lr, weight_decay=config.weight_decay
            ),
        )
    if config.optimizer == "sgd":
        return FunctionalOptimizer(
            init=sgd_init,
            step=lambda params, grads, state, lr: sgd_step(
                params,
                grads,
                state,
                lr=lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            ),
        )
    raise ValueError(f"unknown optimizer {config.optimizer!r}")
