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

When the layer is applied at every position of a sequence, an example's
gradient is a sum of such outer products, one per position, and the dot
product of two examples becomes a double sum over their positions:

    sum over p, r of  (g_p . g'_r) (a_p . a'_r).

That costs the product of the two sequence lengths per pair of examples,
where materialising costs the size of the weight. Neither always wins: long
sequences through a narrow layer favour materialising, short sequences
through a wide one favour the trick. `gradient_dots` picks per layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from culprit.grads.hooks import Recorder, _check_parameters_are_covered, rule_for
from culprit.lora import is_lora
from culprit.task import Task

Strategy = Literal["auto", "ghost", "materialise"]


@dataclass(frozen=True)
class Captured:
    """The input, backprop and rule state of every layer for one batch, detached."""

    layers: dict[str, tuple[Tensor, Tensor, Mapping[str, Tensor]]]


def capture(
    task: Task,
    params: Mapping[str, Tensor],
    inputs: Tensor,
    targets: Tensor,
    *,
    measure: bool = False,
) -> Captured:
    """One forward and one backward pass; keep what the dot products need."""
    with Recorder(task.model, params) as recorder:
        _check_parameters_are_covered(task, recorder.modules)
        fn = task.measures if measure else task.losses
        backprops = recorder.backprops(fn(params, inputs, targets).sum())
    return Captured(
        {
            name: (record.input, backprops[name], record.state)
            for name, record in recorder.records.items()
        }
    )


Layer = tuple[Tensor, Tensor, Mapping[str, Tensor]]


def _materialised(module: nn.Module, layer: Layer, locals_: frozenset[str]) -> Tensor:
    """Per-example gradients of the covered parameters of one layer, `[batch, size]`."""
    rule = rule_for(module)
    assert rule is not None
    grads = rule(module, *layer)
    return torch.cat([grad.flatten(1) for local, grad in grads.items() if local in locals_], dim=1)


def _positions(a: Tensor, g: Tensor) -> tuple[Tensor, Tensor]:
    """Collapse whatever lies between the batch and the features into one axis."""
    batch = g.shape[0]
    index_like = a.dim() == g.dim() - 1  # an embedding's input has no feature axis
    a = a.reshape(batch, -1) if index_like else a.reshape(batch, -1, a.shape[-1])
    return a, g.reshape(batch, -1, g.shape[-1])


def ghost_is_cheaper(module: nn.Module, left: int, right: int) -> bool:
    """Whether the double sum over positions costs less than forming the gradients.

    `left` and `right` are the numbers of positions per example on each
    side. The comparison is of multiplications per pair of examples: the
    double sum does `left * right` dot products of the inputs and of the
    backprops, while a pair of materialised gradients costs one pass over
    the weight.
    """
    if isinstance(module, nn.Linear):
        return left * right * (module.in_features + module.out_features) < (
            module.in_features * module.out_features
        )
    if isinstance(module, nn.Embedding):
        return left * right < module.num_embeddings
    return False


def _ghost_dots(module: nn.Module, left: Layer, right: Layer, locals_: frozenset[str]) -> Tensor:
    """The double sum over positions, as `[left batch, right batch]`."""
    a_left, g_left = _positions(left[0], left[1])
    a_right, g_right = _positions(right[0], right[1])
    backprops = torch.einsum("qpo,bro->qbpr", g_left, g_right)
    if isinstance(module, nn.Embedding):
        # Two positions share a row of the gradient exactly when they hold
        # the same token, so the inputs' dot product is an equality test.
        inputs = (a_left[:, None, :, None] == a_right[None, :, None, :]).to(backprops.dtype)
    else:
        inputs = torch.zeros_like(backprops)
        if "weight" in locals_:
            inputs = inputs + torch.einsum("qpi,bri->qbpr", a_left, a_right)
        if "bias" in locals_:
            inputs = inputs + 1.0
    return (backprops * inputs).sum((2, 3))


def _layer_dots(
    module: nn.Module, left: Layer, right: Layer, locals_: frozenset[str], strategy: Strategy
) -> Tensor:
    """`[left batch, right batch]` dot products of one layer's per-example gradients."""
    # An adapter's gradients are small enough to form, and its two halves
    # would each need the trick applied to a transformed input or backprop.
    if isinstance(module, nn.Linear | nn.Embedding) and not is_lora(module):
        left_positions = left[1][0, ..., 0].numel()
        right_positions = right[1][0, ..., 0].numel()
        if strategy == "ghost" or (
            strategy == "auto" and ghost_is_cheaper(module, left_positions, right_positions)
        ):
            return _ghost_dots(module, left, right, locals_)
    return _materialised(module, left, locals_) @ _materialised(module, right, locals_).T


def gradient_dots(
    task: Task,
    params: Mapping[str, Tensor],
    left: Captured,
    right: Captured,
    *,
    strategy: Strategy = "auto",
) -> Tensor:
    """The table `[left batch, right batch]` of dot products between per-example gradients.

    Equal to multiplying the two matrices of flattened per-example
    gradients, summed over every parameter in `params`. `strategy` forces
    one way of computing it for linear and embedding layers; the default
    chooses the cheaper one for each layer.
    """
    wanted = set(params)
    modules = dict(task.model.named_modules())
    total: Tensor | None = None
    for name, pair in left.layers.items():
        module = modules[name]
        locals_ = frozenset(
            local for local, _ in module.named_parameters() if f"{name}.{local}" in wanted
        )
        dots = _layer_dots(module, pair, right.layers[name], locals_, strategy)
        total = dots if total is None else total + dots
    if total is None:
        raise ValueError("no layer has parameters to take dot products over")
    return total
