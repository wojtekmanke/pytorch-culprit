"""A model, its training loss and the quantity being attributed, as pure functions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from torch import Tensor, nn
from torch.func import functional_call

from culprit.losses import CrossEntropy, LogProb, OutputLoss

Params = dict[str, Tensor]


@dataclass(frozen=True)
class Task:
    """What every attribution method is given.

    `loss` is what training minimised and `measure` is the property of a query
    whose causes are sought. Both are per-example functions of the model's
    outputs.

    The methods below take the parameters as an argument instead of reading
    them from the module, so that they can be differentiated, pushed forward
    and batched with `torch.func`. Parameters that are not passed keep the
    value they have in the module, which is how a method restricts itself to
    a subset of the parameters: set `param_names`.
    """

    model: nn.Module
    loss: OutputLoss = field(default_factory=CrossEntropy)
    measure: OutputLoss = field(default_factory=LogProb)
    param_names: tuple[str, ...] | None = None

    def params(self) -> Params:
        """The attributed parameters, detached from the module's own graph."""
        named = dict(self.model.named_parameters())
        if self.param_names is None:
            names = tuple(name for name, param in named.items() if param.requires_grad)
        else:
            names = self.param_names
        return {name: named[name].detach() for name in names}

    def outputs(self, params: Mapping[str, Tensor], inputs: Tensor) -> Tensor:
        out: Tensor = functional_call(self.model, dict(params), (inputs,))
        return out

    def losses(self, params: Mapping[str, Tensor], inputs: Tensor, targets: Tensor) -> Tensor:
        """Training loss of each example, shape `[batch]`."""
        return self.loss(self.outputs(params, inputs), targets)

    def measures(self, params: Mapping[str, Tensor], inputs: Tensor, targets: Tensor) -> Tensor:
        """Measurement of each example, shape `[batch]`."""
        return self.measure(self.outputs(params, inputs), targets)

    def example_loss(self, params: Mapping[str, Tensor], inputs: Tensor, targets: Tensor) -> Tensor:
        """Loss of one example given without its batch dimension, as a scalar.

        This is the function to hand to `vmap(grad(...))`.
        """
        return self.losses(params, inputs.unsqueeze(0), targets.unsqueeze(0)).squeeze(0)

    def example_measure(
        self, params: Mapping[str, Tensor], inputs: Tensor, targets: Tensor
    ) -> Tensor:
        return self.measures(params, inputs.unsqueeze(0), targets.unsqueeze(0)).squeeze(0)
