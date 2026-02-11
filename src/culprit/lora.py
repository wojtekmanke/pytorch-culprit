"""Low-rank adapters as a parametrization of a frozen weight.

A LoRA adapter (Hu et al., 2021) leaves a layer's weight `W` alone and
learns a low-rank correction to it: the layer uses `W + scale * B @ A`,
with `A` of shape `[rank, in]` and `B` of shape `[out, rank]`.

`torch.nn.utils.parametrize` expresses exactly that. Registering a
parametrization on `layer.weight` turns the attribute into a computed
property, keeps the original tensor as `layer.parametrizations.weight.original`,
and makes the adapter's own parameters ordinary parameters of the model. The
layer's code does not change and nothing has to be wrapped or replaced.

For attribution the useful fact is that the adapter is two linear maps in a
row. If the layer saw input `a` and received backprop `g`, the gradient of
its effective weight is `g a^T`, and by the chain rule

    grad B = scale * g (A a)^T        grad A = scale * (B^T g) a^T.

So `B` is a linear layer whose input is `A a` and whose backprop is
`scale * g`, and `A` is one whose input is `a` and whose backprop is
`scale * B^T g`. Everything this library does with a linear layer's input
and backprop (per-example gradients, Kronecker factors, gradient dot
products) therefore applies to an adapter, read off the layer it sits on.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize

A_NAME = "parametrizations.weight.0.A"
B_NAME = "parametrizations.weight.0.B"


class LoRA(nn.Module):
    """`weight -> weight + scale * B @ A`, starting as the identity because `B` is zero."""

    def __init__(
        self, out_features: int, in_features: int, rank: int, alpha: float, seed: int
    ) -> None:
        super().__init__()
        self.rank, self.scale = rank, alpha / rank
        bound = in_features**-0.5
        generator = torch.Generator().manual_seed(seed)
        self.A = nn.Parameter(
            torch.empty(rank, in_features).uniform_(-bound, bound, generator=generator)
        )
        self.B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, weight: Tensor) -> Tensor:
        return weight + self.scale * (self.B @ self.A)


def is_lora(module: nn.Module) -> bool:
    """Whether `module` is a linear layer whose weight carries a LoRA adapter."""
    return (
        isinstance(module, nn.Linear)
        and parametrize.is_parametrized(module, "weight")
        and isinstance(module.parametrizations.weight[0], LoRA)  # type: ignore[index,union-attr]
    )


def lora_of(module: nn.Module) -> LoRA:
    adapter = module.parametrizations.weight[0]  # type: ignore[index,union-attr]
    assert isinstance(adapter, LoRA)
    return adapter


def add_lora(
    model: nn.Module,
    targets: Sequence[str],
    *,
    rank: int = 8,
    alpha: float = 16.0,
    seed: int = 0,
) -> list[str]:
    """Freeze `model` and put an adapter on every linear layer named in `targets`.

    A layer is chosen when its name ends with one of `targets`, so
    `["q_proj", "v_proj"]` picks those projections in every block. Returns
    the names of the adapters' parameters, which are then the only ones that
    require a gradient.
    """
    for param in model.parameters():
        param.requires_grad_(False)
    chosen = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and any(name.endswith(target) for target in targets)
    ]
    if not chosen:
        raise ValueError(f"no linear layer is named like any of {list(targets)}")
    names = []
    for index, (name, module) in enumerate(chosen):
        adapter = LoRA(module.out_features, module.in_features, rank, alpha, seed + index)
        adapter.to(module.weight.device, module.weight.dtype)
        parametrize.register_parametrization(module, "weight", adapter)
        names += [f"{name}.{A_NAME}", f"{name}.{B_NAME}"]
    return names


def merge_lora(model: nn.Module) -> None:
    """Fold every adapter into its weight and remove the parametrization."""
    for module in model.modules():
        if is_lora(module):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
