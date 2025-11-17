"""Per-example gradients by batching the gradient function itself."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from torch import Tensor
from torch.func import grad, vmap

from culprit.task import Params

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
