"""Gaussian clusters: a classification problem small enough for dense algebra."""

from __future__ import annotations

import torch
from torch import Tensor


def gaussian_blobs(
    n: int, dim: int = 10, classes: int = 3, spread: float = 1.0, seed: int = 0
) -> tuple[Tensor, Tensor]:
    """`n` points around `classes` random centres, with their class labels.

    `spread` is the standard deviation of the points around a centre whose
    coordinates are themselves standard normal, so values near 1 make the
    classes overlap and no model can fit them perfectly.
    """
    generator = torch.Generator().manual_seed(seed)
    centres = torch.randn(classes, dim, generator=generator)
    labels = torch.randint(classes, (n,), generator=generator)
    points = centres[labels] + spread * torch.randn(n, dim, generator=generator)
    return points, labels
