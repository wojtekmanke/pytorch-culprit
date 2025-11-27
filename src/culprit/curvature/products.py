"""Curvature-vector products that never form the curvature matrix.

Every function here returns the product for the *summed* loss of the batch
it is given, as a dictionary shaped like the parameters. Sums add up across
batches; `culprit.linalg` turns them into operators for the mean loss of a
whole dataset.

A Hessian-vector product is the derivative of the gradient along a
direction. Reverse mode gives the gradient and either mode can then take the
directional derivative, which is why there are three ways to compute it.
They return the same numbers and differ in what they ask of the model's
operations and in how much memory they hold.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import Tensor
from torch.func import grad, jvp, vjp

from culprit.task import Params, Task


def _summed_loss(task: Task, inputs: Tensor, targets: Tensor) -> Callable[[Params], Tensor]:
    def loss(params: Params) -> Tensor:
        return task.losses(params, inputs, targets).sum()

    return loss


def hvp_double_backward(
    task: Task,
    params: Mapping[str, Tensor],
    inputs: Tensor,
    targets: Tensor,
    vector: Mapping[str, Tensor],
) -> Params:
    """Reverse over reverse, with `torch.autograd` and no function transforms.

    The first backward pass is itself recorded, then the inner product of the
    gradient with `vector` is differentiated. Every operation in the model
    needs a backward whose own backward is implemented.
    """
    leaves = {name: value.detach().requires_grad_() for name, value in params.items()}
    loss = task.losses(leaves, inputs, targets).sum()
    grads = torch.autograd.grad(loss, list(leaves.values()), create_graph=True)
    along = sum((g * vector[name]).sum() for name, g in zip(leaves, grads, strict=True))
    assert isinstance(along, Tensor)
    products = torch.autograd.grad(along, list(leaves.values()))
    return dict(zip(leaves, products, strict=True))


def hvp_forward_over_reverse(
    task: Task,
    params: Mapping[str, Tensor],
    inputs: Tensor,
    targets: Tensor,
    vector: Mapping[str, Tensor],
) -> Params:
    """Push `vector` forward through the function that computes the gradient.

    Usually the cheapest of the three: the tangents travel alongside the
    forward and backward passes and nothing has to be kept for a second
    backward pass. Every operation, and the backward of every operation,
    needs a forward-mode rule.
    """
    gradient = grad(_summed_loss(task, inputs, targets))
    out: Params = jvp(gradient, (dict(params),), (dict(vector),))[1]
    return out


def hvp_reverse_over_forward(
    task: Task,
    params: Mapping[str, Tensor],
    inputs: Tensor,
    targets: Tensor,
    vector: Mapping[str, Tensor],
) -> Params:
    """Differentiate the directional derivative of the loss along `vector`.

    The directional derivative is a scalar computed in one forward-mode pass;
    its gradient is the product. Operations need a forward-mode rule that can
    itself be differentiated in reverse.

    Do not use this on a model with a layer norm. On torch 2.14 the
    forward-mode rule of `F.layer_norm` builds its tangent from the saved
    mean and scale, reverse mode treats those as constants, and the product
    comes out wrong without an error. It is kept because it is correct for
    models without one and because comparing the three is how that was found;
    see `docs/notes/pitfalls.md`.
    """
    loss = _summed_loss(task, inputs, targets)
    direction = dict(vector)

    def directional_derivative(at: Params) -> Tensor:
        slope: Tensor = jvp(loss, (at,), (direction,))[1]
        return slope

    out: Params = grad(directional_derivative)(dict(params))
    return out


def ggn_vp(
    task: Task,
    params: Mapping[str, Tensor],
    inputs: Tensor,
    targets: Tensor,
    vector: Mapping[str, Tensor],
) -> Params:
    """The Gauss-Newton matrix `J^T H J` of the summed loss, times `vector`.

    Read right to left, the product is three passes and each one is a
    different transform:

    - `J v`: a forward-mode pass through the model, parameters to outputs;
    - `H (J v)`: a Hessian-vector product of the loss alone, in the space of
      the outputs, where it is cheap;
    - `J^T (...)`: an ordinary backward pass through the model.

    No pass differentiates another pass through the model, so unlike a
    Hessian-vector product this needs no second derivative of any layer.
    """

    def outputs_of(at: Params) -> Tensor:
        return task.outputs(at, inputs)

    def loss_of(outputs: Tensor) -> Tensor:
        return task.loss(outputs, targets).sum()

    point = dict(params)
    outputs, pushed = jvp(outputs_of, (point,), (dict(vector),))[:2]
    curved = jvp(grad(loss_of), (outputs,), (pushed,))[1]
    pullback = vjp(outputs_of, point)[1]
    out: Params = pullback(curved)[0]
    return out
