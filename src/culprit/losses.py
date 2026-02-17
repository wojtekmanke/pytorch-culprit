"""Per-example losses and measurements, as functions of the model's outputs.

Everything here maps `outputs` of shape `[batch, ..., classes]` and integer
`targets` of shape `[batch, ...]` to one number per example. Positions whose
target is `IGNORE` do not count, which is how a language-model example masks
its padding, and how a query picks out the one token it asks about.

Keeping the loss separate from the model is what lets the curvature code
split the Hessian at the model's output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor
from torch.nn import functional as F

IGNORE = -100


class OutputLoss(Protocol):
    def __call__(self, outputs: Tensor, targets: Tensor) -> Tensor:
        """One value per example, shape `[batch]`."""
        ...


def _target_log_probs(outputs: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    """The log-probability of each target, and which positions count."""
    keep = targets != IGNORE
    index = targets.clamp_min(0).unsqueeze(-1)
    picked = F.log_softmax(outputs, dim=-1).gather(-1, index).squeeze(-1)
    return picked, keep


def _sum_per_example(values: Tensor, keep: Tensor) -> Tensor:
    return values.masked_fill(~keep, 0.0).reshape(values.shape[0], -1).sum(1)


@dataclass(frozen=True)
class CrossEntropy:
    """Negative log-likelihood, summed over the positions of each example."""

    def __call__(self, outputs: Tensor, targets: Tensor) -> Tensor:
        picked, keep = _target_log_probs(outputs, targets)
        return _sum_per_example(-picked, keep)

    def sample(self, outputs: Tensor, targets: Tensor, generator: torch.Generator) -> Tensor:
        """Labels drawn from the model's own predictions, ignored where `targets` are.

        Only the counted positions are drawn for. With a vocabulary of a
        hundred thousand and one counted position per sequence, drawing for
        every position would move hundreds of megabytes to do nothing with.
        The draw happens on the CPU so that one seed gives the same labels
        on every device.
        """
        keep = targets != IGNORE
        probabilities = F.softmax(outputs.detach()[keep], dim=-1).cpu()
        drawn = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
        sampled = targets.clone()
        sampled[keep] = drawn.to(targets.device)
        return sampled


@dataclass(frozen=True)
class LogProb:
    """Log-probability of the targets: higher means the model agrees more."""

    def __call__(self, outputs: Tensor, targets: Tensor) -> Tensor:
        picked, keep = _target_log_probs(outputs, targets)
        return _sum_per_example(picked, keep)


@dataclass(frozen=True)
class Margin:
    """`log p - log(1 - p)` of the targets, the output function TRAK uses.

    Unlike the log-probability it does not saturate when the model is
    confident, so its gradient stays informative for well-fitted examples.
    """

    def __call__(self, outputs: Tensor, targets: Tensor) -> Tensor:
        keep = targets != IGNORE
        index = targets.clamp_min(0).unsqueeze(-1)
        target_logit = outputs.gather(-1, index).squeeze(-1)
        others = outputs.scatter(-1, index, float("-inf")).logsumexp(-1)
        return _sum_per_example(target_logit - others, keep)
