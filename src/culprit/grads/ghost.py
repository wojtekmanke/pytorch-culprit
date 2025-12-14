"""Dot products between per-example gradients that never form the gradients.

Many attribution scores are, in the end, an inner product between the
gradient of a query and the gradient of a training example. Forming the two
gradients costs a vector the size of the model per example. For a linear
layer it is not needed: the gradient is the outer product `g a^T` of a
backprop and an input, and

    <g a^T, g' a'^T>  =  (g . g') (a . a'),

so a whole table of dot products between two batches is the elementwise
product of two small Gram matrices. This is the "ghost" trick, first used to
get per-example gradient norms for differentially private training.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from culprit.grads.hooks import RULES, Recorder, _check_parameters_are_covered
from culprit.task import Task


@dataclass(frozen=True)
class Captured:
    """The input and backprop of every layer for one batch, detached."""

    layers: dict[str, tuple[Tensor, Tensor]]


def capture(
    task: Task,
    params: Mapping[str, Tensor],
    inputs: Tensor,
    targets: Tensor,
    *,
    measure: bool = False,
) -> Captured:
    """One forward and one backward pass; keep what the dot products need."""
    with Recorder(task.model) as recorder:
        _check_parameters_are_covered(task, recorder.modules)
        fn = task.measures if measure else task.losses
        backprops = recorder.backprops(fn(params, inputs, targets).sum())
    wanted = set(params)
    layers = {
        name: (record.input, backprops[name])
        for name, record in recorder.records.items()
        if any(
            f"{name}.{local}" in wanted for local, _ in recorder.modules[name].named_parameters()
        )
    }
    return Captured(layers)


def _materialised(module: nn.Module, a: Tensor, g: Tensor, locals_: frozenset[str]) -> Tensor:
    """Per-example gradients of the covered parameters of one layer, `[batch, size]`."""
    grads = RULES[type(module)](module, a, g)
    return torch.cat([grad.flatten(1) for local, grad in grads.items() if local in locals_], dim=1)


def _layer_dots(
    module: nn.Module,
    left: tuple[Tensor, Tensor],
    right: tuple[Tensor, Tensor],
    locals_: frozenset[str],
) -> Tensor:
    """`[left batch, right batch]` dot products of one layer's per-example gradients."""
    (a_left, g_left), (a_right, g_right) = left, right
    if isinstance(module, nn.Linear) and a_left.dim() == 2:
        # One input per example: the product of two Gram matrices. The bias
        # is the weight of a constant input of one, which adds one to the
        # dot product of the inputs.
        inputs = a_left.new_zeros(a_left.shape[0], a_right.shape[0])
        if "weight" in locals_:
            inputs = inputs + a_left @ a_right.T
        if "bias" in locals_:
            inputs = inputs + 1.0
        return (g_left @ g_right.T) * inputs
    return (
        _materialised(module, a_left, g_left, locals_)
        @ _materialised(module, a_right, g_right, locals_).T
    )


def gradient_dots(
    task: Task, params: Mapping[str, Tensor], left: Captured, right: Captured
) -> Tensor:
    """The table `[left batch, right batch]` of dot products between per-example gradients.

    Equal to multiplying the two matrices of flattened per-example
    gradients, summed over every parameter in `params`.
    """
    wanted = set(params)
    modules = dict(task.model.named_modules())
    total: Tensor | None = None
    for name, pair in left.layers.items():
        module = modules[name]
        locals_ = frozenset(
            local for local, _ in module.named_parameters() if f"{name}.{local}" in wanted
        )
        dots = _layer_dots(module, pair, right.layers[name], locals_)
        total = dots if total is None else total + dots
    if total is None:
        raise ValueError("no layer has parameters to take dot products over")
    return total
