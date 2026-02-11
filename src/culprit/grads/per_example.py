"""Per-example gradients by batching the gradient function itself."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import Tensor
from torch.func import grad, vmap

from culprit.batching import batches
from culprit.flat import Layout
from culprit.grads.hooks import hook_per_example_grads
from culprit.task import Params, Task

ExampleFn = Callable[[Mapping[str, Tensor], Tensor, Tensor], Tensor]


def per_example_grads(
    fn: ExampleFn,
    params: Mapping[str, Tensor],
    inputs: Tensor,
    targets: Tensor,
    *,
    chunk_size: int | None = None,
) -> Params:
    """The gradient of `fn` at every example, each value `[batch, *shape]`.

    `fn` maps the parameters and one example, given without its batch
    dimension, to a scalar: `task.example_loss` or `task.example_measure`.

    `grad` turns it into a function that returns one gradient, and `vmap`
    runs that function for the whole batch in a single pass. Memory grows
    with the batch, because every intermediate of the backward pass gains a
    batch dimension; `chunk_size` bounds it by running that many examples at
    a time.
    """
    batched = vmap(grad(fn), in_dims=(None, 0, 0), chunk_size=chunk_size)
    out: Params = batched(dict(params), inputs, targets)
    return out


def gradient_matrix(
    task: Task,
    params: Mapping[str, Tensor],
    inputs: Tensor,
    targets: Tensor,
    *,
    measure: bool = False,
    batch_size: int | None = None,
) -> Tensor:
    """Per-example gradients of a whole dataset as rows, `[examples, size]`.

    The rows are gradients of the loss, or of the measurement when `measure`
    is set, flattened in the order of `Layout.of(params)`. They are computed
    the way the task asks for, by `vmap` or from hooks.
    """
    layout = Layout.of(params)

    def grads(x: Tensor, y: Tensor) -> Params:
        if task.per_example == "hooks":
            return hook_per_example_grads(task, params, x, y, measure=measure)
        fn = task.example_measure if measure else task.example_loss
        return per_example_grads(fn, params, x, y)

    return torch.cat([layout.flatten(grads(x, y)) for x, y in batches(inputs, targets, batch_size)])
