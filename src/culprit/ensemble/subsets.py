"""Which examples each retrained model sees."""

from __future__ import annotations

import torch
from torch import Tensor


def random_subsets(examples: int, models: int, fraction: float, seed: int) -> Tensor:
    """`[models, examples]` masks, each selecting the same number of examples at random.

    Every row keeps exactly `round(fraction * examples)` examples, so that
    all models train for the same number of steps and can be stacked. Over
    many rows each example is kept about `fraction` of the time, which is
    what lets the effect of any one example be separated from the rest.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    kept = round(fraction * examples)
    generator = torch.Generator().manual_seed(seed)
    masks = torch.zeros(models, examples, dtype=torch.bool)
    for row in masks:
        row[torch.randperm(examples, generator=generator)[:kept]] = True
    return masks


def pools(masks: Tensor) -> Tensor:
    """The indices each mask selects, in increasing order: `[models, kept]`."""
    counts = masks.sum(1)
    if bool((counts != counts[0]).any()):
        raise ValueError("every mask must select the same number of examples")
    return masks.nonzero()[:, 1].view(len(masks), int(counts[0]))
