"""Curvature matrices written out in full, for models small enough to allow it.

These are the oracle. Everything else in `culprit.curvature` is a way of
working with one of these matrices without forming it, and is tested by
comparing against the matrix formed here.

All three are matrices of the mean loss over the examples given, indexed by
the flat parameter vector of `Layout.of(params)`.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor
from torch.func import grad, hessian, jacfwd, jacrev, vmap

from culprit.flat import Layout
from culprit.grads import per_example_grads
from culprit.task import Task


def dense_hessian(
    task: Task, params: Mapping[str, Tensor], inputs: Tensor, targets: Tensor
) -> Tensor:
    """The second derivative of the mean loss, `[size, size]`.

    This is `jacfwd(grad(f))` and deliberately not `torch.func.hessian(f)`,
    which is `jacfwd(jacrev(f))`. `jacrev` runs the backward pass under
    `vmap`, and the batching rule for the backward of an affine layer norm
    rebuilds the gradient of its weight from the saved mean and scale as if
    they were constants. Differentiating that once more loses the dependence
    of the mean and scale on the input, and the block of the Hessian between
    a layer norm's weight and anything upstream of it comes out wrong, with
    no error (torch 2.14). For a scalar function `grad` gives the same
    gradient as `jacrev` without the `vmap`. See `docs/notes/pitfalls.md`.
    """
    layout = Layout.of(params)

    def mean_loss(vector: Tensor) -> Tensor:
        return task.losses(layout.unflatten(vector), inputs, targets).mean()

    out: Tensor = jacfwd(grad(mean_loss))(layout.flatten(params))
    return out


def dense_ggn(task: Task, params: Mapping[str, Tensor], inputs: Tensor, targets: Tensor) -> Tensor:
    """The generalised Gauss-Newton matrix of the mean loss, `[size, size]`.

    With `J` the Jacobian of an example's outputs with respect to the
    parameters and `H` the Hessian of its loss with respect to those outputs,
    this is the mean of `J^T H J`. It is the Hessian with the second
    derivative of the model itself left out, which makes it positive
    semi-definite whenever the loss is convex in the outputs.
    """
    layout = Layout.of(params)

    def example_outputs(vector: Tensor, x: Tensor) -> Tensor:
        return task.outputs(layout.unflatten(vector), x.unsqueeze(0)).squeeze(0)

    def example_output_loss(outputs: Tensor, y: Tensor) -> Tensor:
        return task.loss(outputs.unsqueeze(0), y.unsqueeze(0)).squeeze(0)

    batch = len(inputs)
    jacobian = vmap(jacrev(example_outputs), in_dims=(None, 0))(layout.flatten(params), inputs)
    jacobian = jacobian.reshape(batch, -1, layout.size)
    width = jacobian.shape[1]
    output_hessian = vmap(hessian(example_output_loss))(task.outputs(params, inputs), targets)
    output_hessian = output_hessian.reshape(batch, width, width)
    return torch.einsum("boi,bop,bpj->ij", jacobian, output_hessian, jacobian) / batch


def dense_empirical_fisher(
    task: Task, params: Mapping[str, Tensor], inputs: Tensor, targets: Tensor
) -> Tensor:
    """The mean outer product of the per-example loss gradients, `[size, size]`.

    Despite the name this is not the Fisher information: it uses the labels
    in the data where the Fisher takes an expectation over the model's own
    predictions. It is what methods that only have gradients can build.
    """
    grads = Layout.of(params).flatten(per_example_grads(task.example_loss, params, inputs, targets))
    return grads.T @ grads / len(inputs)
