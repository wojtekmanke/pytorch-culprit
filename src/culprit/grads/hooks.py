"""Per-example gradients from what each layer saw and what flowed back to it.

For a layer whose parameters act on each example separately, the gradient of
an example's loss with respect to those parameters is a function of two
things: the layer's input for that example, and the gradient of the loss
with respect to the layer's output for that example. One ordinary forward
pass and one backward pass yield both for the whole batch.

The backward pass here stops at the outputs of the recorded layers. It never
reaches the parameters, so nothing is accumulated into `.grad` and no
batch-summed parameter gradient is computed only to be thrown away.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import TracebackType

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.hooks import RemovableHandle

from culprit.task import Params, Task


@dataclass
class Record:
    input: Tensor  # detached
    output: Tensor  # still attached to the graph, so it can be differentiated to


def _linear(module: nn.Module, a: Tensor, g: Tensor) -> Params:
    """`a` is `[batch, ..., in]` and `g` is `[batch, ..., out]`.

    Each position contributes the outer product of its backprop and its
    input; an example's gradient is the sum over its positions.
    """
    assert isinstance(module, nn.Linear)
    grads = {"weight": torch.einsum("b...o,b...i->boi", g, a)}
    if module.bias is not None:
        grads["bias"] = g.reshape(g.shape[0], -1, g.shape[-1]).sum(1)
    return grads


def _embedding(module: nn.Module, a: Tensor, g: Tensor) -> Params:
    """`a` holds indices `[batch, ...]` and `g` is `[batch, ..., width]`.

    Row `v` of an example's gradient is the sum of the backprops at the
    positions where that example used token `v`.
    """
    assert isinstance(module, nn.Embedding)
    batch, width = g.shape[0], g.shape[-1]
    index = a.reshape(batch, -1, 1).expand(-1, -1, width)
    grad = g.new_zeros(batch, module.num_embeddings, width)
    grad.scatter_add_(1, index, g.reshape(batch, -1, width))
    return {"weight": grad}


def _layer_norm(module: nn.Module, a: Tensor, g: Tensor) -> Params:
    """The affine part of the layer multiplies and shifts the normalised input."""
    assert isinstance(module, nn.LayerNorm)
    if not module.elementwise_affine:
        return {}
    normalised = F.layer_norm(a, module.normalized_shape, None, None, module.eps)
    shape = (g.shape[0], -1, *module.normalized_shape)
    grads = {"weight": (g * normalised).reshape(shape).sum(1)}
    if module.bias is not None:
        grads["bias"] = g.reshape(shape).sum(1)
    return grads


LayerRule = Callable[[nn.Module, Tensor, Tensor], Params]

# How to turn a layer's input and backprop into per-example gradients of its
# parameters, keyed by the parameter's name within the layer.
RULES: dict[type[nn.Module], LayerRule] = {
    nn.Linear: _linear,
    nn.Embedding: _embedding,
    nn.LayerNorm: _layer_norm,
}


class Recorder:
    """Records the input and output of every supported layer for one forward pass.

    Use it as a context manager around the forward pass, then call
    `backprops` with a scalar computed from the model's output.
    """

    def __init__(self, model: nn.Module) -> None:
        self.modules = {
            name: module for name, module in model.named_modules() if type(module) in RULES
        }
        self.records: dict[str, Record] = {}
        self._handles: list[RemovableHandle] = []

    def __enter__(self) -> Recorder:
        for name, module in self.modules.items():
            self._handles.append(module.register_forward_hook(self._hook(name)))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _hook(self, name: str) -> Callable[[nn.Module, tuple[Tensor, ...], Tensor], None]:
        def hook(module: nn.Module, args: tuple[Tensor, ...], output: Tensor) -> None:
            if name in self.records:
                raise RuntimeError(f"{name} ran twice in one forward pass")
            # When the parameters were passed in as plain tensors nothing
            # upstream requires grad, and the output is a leaf. Marking it
            # starts the graph here, which is all the backward pass needs.
            if not output.requires_grad:
                output.requires_grad_()
            self.records[name] = Record(args[0].detach(), output)

        return hook

    def backprops(self, value: Tensor) -> dict[str, Tensor]:
        """The gradient of the scalar `value` at each recorded layer's output."""
        names = list(self.records)
        grads = torch.autograd.grad(value, [self.records[name].output for name in names])
        return dict(zip(names, grads, strict=True))


def _check_parameters_are_covered(task: Task, modules: Mapping[str, nn.Module]) -> None:
    owners: dict[int, str] = {}
    covered = set()
    for module_name, module in modules.items():
        for local, param in module.named_parameters(recurse=False):
            full = f"{module_name}.{local}"
            if id(param) in owners:
                raise ValueError(
                    f"{full} and {owners[id(param)]} are the same tensor; "
                    "per-layer gradients cannot be attributed to a shared parameter"
                )
            owners[id(param)] = full
            covered.add(full)
    missing = [name for name in task.params() if name not in covered]
    if missing:
        raise ValueError(f"no rule gives per-example gradients for: {', '.join(missing)}")


def hook_per_example_grads(
    task: Task,
    params: Mapping[str, Tensor],
    inputs: Tensor,
    targets: Tensor,
    *,
    measure: bool = False,
) -> Params:
    """The gradient of each example's loss (or measurement), `[batch, *shape]`.

    Gives the same result as `per_example_grads`, from a single forward and
    backward pass of the ordinary batched model. It supports the layer types
    in `RULES` and relies on every recorded layer keeping the batch as its
    first dimension.
    """
    with Recorder(task.model) as recorder:
        _check_parameters_are_covered(task, recorder.modules)
        fn = task.measures if measure else task.losses
        # Summing is safe because examples do not interact: the gradient of
        # the sum at one example's activations is that example's own gradient.
        backprops = recorder.backprops(fn(params, inputs, targets).sum())

    grads: Params = {}
    for name, record in recorder.records.items():
        module = recorder.modules[name]
        for local, grad in RULES[type(module)](module, record.input, backprops[name]).items():
            grads[f"{name}.{local}"] = grad
    return {name: grads[name] for name in params}
