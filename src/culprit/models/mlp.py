"""A small fully connected classifier."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}


class MLP(nn.Module):
    """Linear layers with an activation between them.

    `sizes` lists every width from input to output, so `MLP([196, 32, 10])` has
    one hidden layer. Inputs of any shape are flattened to `[batch, sizes[0]]`.

    The default activation is `tanh` because it is smooth: the Hessian of a
    ReLU network is only defined almost everywhere and equals its Gauss-Newton
    matrix there, which would make tests that tell the two apart vacuous.
    """

    def __init__(self, sizes: Sequence[int], activation: str = "tanh", seed: int = 0) -> None:
        super().__init__()
        if len(sizes) < 2:
            raise ValueError("sizes needs at least an input and an output width")
        act = _ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
            layers.append(nn.Linear(fan_in, fan_out))
            if i < len(sizes) - 2:
                layers.append(act())
        self.layers = nn.Sequential(*layers)
        self.reset_parameters(seed)

    def reset_parameters(self, seed: int) -> None:
        """Initialise from `seed` alone, without touching the global generator."""
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for module in self.layers:
                if isinstance(module, nn.Linear):
                    bound = module.in_features**-0.5
                    # Drawn on the CPU and copied over: the same seed then
                    # gives the same model on every device, and a CPU
                    # generator cannot fill a tensor that lives on the GPU.
                    drawn = torch.empty(module.weight.shape, dtype=module.weight.dtype)
                    module.weight.copy_(drawn.uniform_(-bound, bound, generator=generator))
                    module.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.layers(x.flatten(1))
        return out
