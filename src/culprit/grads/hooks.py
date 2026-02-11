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

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import TracebackType

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.hooks import RemovableHandle

from culprit.lora import A_NAME, B_NAME, is_lora, lora_of
from culprit.models.rmsnorm import RMSNorm
from culprit.task import Params, Task

State = Mapping[str, Tensor]


@dataclass
class Record:
    input: Tensor  # detached
    output: Tensor  # still attached to the graph, so it can be differentiated to
    # Parameter values a rule needs, as they were during the forward pass.
    # They are read then because a functional call puts the module's own
    # values back afterwards.
    state: dict[str, Tensor] = field(default_factory=dict)


def _linear(module: nn.Module, a: Tensor, g: Tensor, state: State) -> Params:
    """`a` is `[batch, ..., in]` and `g` is `[batch, ..., out]`.

    Each position contributes the outer product of its backprop and its
    input; an example's gradient is the sum over its positions.
    """
    assert isinstance(module, nn.Linear)
    grads = {"weight": torch.einsum("b...o,b...i->boi", g, a)}
    if module.bias is not None:
        grads["bias"] = g.reshape(g.shape[0], -1, g.shape[-1]).sum(1)
    return grads


def _embedding(module: nn.Module, a: Tensor, g: Tensor, state: State) -> Params:
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


def _layer_norm(module: nn.Module, a: Tensor, g: Tensor, state: State) -> Params:
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


def _rms_norm(module: nn.Module, a: Tensor, g: Tensor, state: State) -> Params:
    """The weight multiplies the normalised input, as in a layer norm without a bias."""
    assert isinstance(module, RMSNorm)
    return {"weight": (g * module.normalise(a)).reshape(g.shape[0], -1, module.width).sum(1)}


def _lora_linear(module: nn.Module, a: Tensor, g: Tensor, state: State) -> Params:
    """A linear layer with a LoRA adapter: two linear maps read off one layer.

    `B` sees the input `A a` and the backprop `scale * g`; `A` sees the input
    `a` and the backprop `scale * B^T g`. The frozen weight gets no gradient.
    """
    assert isinstance(module, nn.Linear)
    scale = lora_of(module).scale
    down = a @ state["A"].T
    back = g @ state["B"]
    grads = {
        A_NAME: scale * torch.einsum("b...r,b...i->bri", back, a),
        B_NAME: scale * torch.einsum("b...o,b...r->bor", g, down),
    }
    if module.bias is not None:
        grads["bias"] = g.reshape(g.shape[0], -1, g.shape[-1]).sum(1)
    return grads


LayerRule = Callable[[nn.Module, Tensor, Tensor, State], Params]

# How to turn a layer's input and backprop into per-example gradients of its
# parameters, keyed by the parameter's name within the layer.
RULES: dict[type[nn.Module], LayerRule] = {
    nn.Linear: _linear,
    nn.Embedding: _embedding,
    nn.LayerNorm: _layer_norm,
    RMSNorm: _rms_norm,
}


def rule_for(module: nn.Module) -> LayerRule | None:
    """The rule for `module`, or `None` if its parameters are not supported."""
    if is_lora(module):
        return _lora_linear
    return RULES.get(type(module))


def _state_of(module: nn.Module) -> dict[str, Tensor]:
    """The parameter values `module`'s rule will need, read during the forward pass."""
    if is_lora(module):
        adapter = lora_of(module)
        return {"A": adapter.A.detach(), "B": adapter.B.detach()}
    return {}


class Recorder:
    """Records the input and output of every supported layer for one forward pass.

    Use it as a context manager around the forward pass, then call
    `backprops` with a scalar computed from the model's output.
    """

    def __init__(self, model: nn.Module, params: Iterable[str] | None = None) -> None:
        """Record the supported layers of `model`.

        With `params`, only the layers that own one of those parameters are
        recorded. On a large model with a few trainable adapters that is the
        difference between keeping every activation and keeping a handful.
        """
        wanted = None if params is None else set(params)
        self.modules = {
            name: module
            for name, module in model.named_modules()
            if rule_for(module) is not None
            and (
                wanted is None
                or any(f"{name}.{local}" in wanted for local, _ in module.named_parameters())
            )
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
            self.records[name] = Record(args[0].detach(), output, _state_of(module))

        return hook

    def backprops(self, value: Tensor, *, retain_graph: bool = False) -> dict[str, Tensor]:
        """The gradient of the scalar `value` at each recorded layer's output.

        Pass `retain_graph` to take the backprops of several scalars from
        the same forward pass.
        """
        names = list(self.records)
        outputs = [self.records[name].output for name in names]
        grads = torch.autograd.grad(value, outputs, retain_graph=retain_graph)
        return dict(zip(names, grads, strict=True))


def _check_parameters_are_covered(task: Task, modules: Mapping[str, nn.Module]) -> None:
    """Refuse parameters no recorded layer accounts for, or that two layers share.

    `modules` are the recorded layers. Sharing is looked for across the whole
    model: a tied weight is listed under one name only, so the layer that
    reaches it under its other name is not among the recorded ones, and its
    half of the gradient would silently go missing.
    """
    names = set(task.params())
    wanted = {id(param) for name, param in task.model.named_parameters() if name in names}
    owners: dict[int, str] = {}
    for module_name, module in task.model.named_modules():
        if rule_for(module) is None:
            continue
        for local, param in module.named_parameters():
            full = f"{module_name}.{local}"
            if id(param) in owners and id(param) in wanted:
                raise ValueError(
                    f"{full} and {owners[id(param)]} are the same tensor; "
                    "per-layer gradients cannot be attributed to a shared parameter"
                )
            owners[id(param)] = full
    covered = {
        f"{module_name}.{local}"
        for module_name, module in modules.items()
        for local, _ in module.named_parameters()
    }
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
    with Recorder(task.model, params) as recorder:
        _check_parameters_are_covered(task, recorder.modules)
        fn = task.measures if measure else task.losses
        # Summing is safe because examples do not interact: the gradient of
        # the sum at one example's activations is that example's own gradient.
        backprops = recorder.backprops(fn(params, inputs, targets).sum())

    grads: Params = {}
    for name, record in recorder.records.items():
        module = recorder.modules[name]
        rule = rule_for(module)
        assert rule is not None
        for local, grad in rule(module, record.input, backprops[name], record.state).items():
            grads[f"{name}.{local}"] = grad
    return {name: grads[name] for name in params}
