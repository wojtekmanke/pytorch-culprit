"""How well attribution scores agree with what retraining showed."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from culprit.losses import IGNORE


def ranks(values: Tensor) -> Tensor:
    """The rank of each entry within its column, ties sharing their average rank."""
    out = torch.empty(values.shape, dtype=torch.float64)
    for column in range(values.shape[1]):
        order = values[:, column].argsort()
        ordinal = torch.empty(len(order), dtype=torch.float64)
        ordinal[order] = torch.arange(len(order), dtype=torch.float64)
        _, inverse, counts = values[:, column].unique(return_inverse=True, return_counts=True)
        sums = torch.zeros(len(counts), dtype=torch.float64).index_add_(0, inverse, ordinal)
        out[:, column] = (sums / counts)[inverse]
    return out


def spearman(a: Tensor, b: Tensor) -> Tensor:
    """The rank correlation of each column of `a` with the same column of `b`.

    A column that is constant has no ranking to correlate; its correlation
    is reported as zero, which is what a method that says nothing deserves.
    """
    ranked_a, ranked_b = ranks(a), ranks(b)
    centred_a = ranked_a - ranked_a.mean(0)
    centred_b = ranked_b - ranked_b.mean(0)
    scale = centred_a.norm(dim=0) * centred_b.norm(dim=0)
    correlation = (centred_a * centred_b).sum(0) / scale
    return torch.where(scale > 0, correlation, torch.zeros_like(correlation))


def predictions(scores: Tensor, masks: Tensor) -> Tensor:
    """What the scores say each retrained model should measure, `[models, queries]`.

    A score is the effect of including one example, so the prediction for a
    model is the sum of the scores of the examples it was trained on. Only
    differences between models matter, which is why no constant is needed.
    """
    return masks.to(torch.float64) @ scores.to(torch.float64).T


def lds(scores: Tensor, masks: Tensor, measurements: Tensor) -> Tensor:
    """The linear datamodeling score of each query (Park et al., 2023).

    `scores` is `[queries, examples]`. `masks` is `[models, examples]` and
    says which examples each retrained model saw; `measurements` is
    `[models, queries]` and says what it then measured. The score of a query
    is the rank correlation, across the retrained models, between what the
    scores predict and what was measured.

    It is 1 when the scores order the models exactly as retraining does and
    0 when they say nothing. It asks only for the right ordering of subsets,
    so it does not care about the scale of the scores.
    """
    return spearman(predictions(scores, masks), measurements.to(torch.float64))


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.low:.3f}, {self.high:.3f}]"


def bootstrap(
    values: Tensor, *, samples: int = 2000, level: float = 0.95, seed: int = 0
) -> Interval:
    """The mean of `values` with a percentile bootstrap interval.

    The values are resampled with replacement and the mean recomputed; the
    interval holds the middle `level` of those means. For per-query scores
    this says how much the average would move with a different set of
    queries. It says nothing about a different set of retrained models.
    """
    values = values.to(torch.float64)
    generator = torch.Generator().manual_seed(seed)
    picks = torch.randint(len(values), (samples, len(values)), generator=generator)
    means = values[picks].mean(1)
    tail = (1 - level) / 2
    low, high = torch.quantile(means, torch.tensor([tail, 1 - tail], dtype=torch.float64))
    return Interval(mean=float(values.mean()), low=float(low), high=float(high))


def source_recall(scores: Tensor, sources: Tensor) -> Tensor:
    """For each query, the share of its sources found at the top of its ranking.

    `sources` is `[queries, examples]` and marks the training examples known
    to be responsible for each query. A query with `k` sources is given `k`
    guesses, the `k` highest-scored examples, and the result is the fraction
    of them that are sources: 1 when the sources are ranked above everything
    else, and about `k / examples` for a random ranking. A query without
    sources has nothing to recall and gets `nan`.
    """
    counts = sources.sum(1)
    order = scores.argsort(dim=1, descending=True)
    hits = sources.gather(1, order).to(torch.float64)
    position = torch.arange(scores.shape[1]).expand_as(hits)
    within = (hits * (position < counts.unsqueeze(1))).sum(1)
    return torch.where(counts > 0, within / counts, torch.full_like(within, float("nan")))


def average_precision(scores: Tensor, sources: Tensor) -> Tensor:
    """For each query, the mean precision at the rank of each of its sources.

    Unlike `source_recall` this looks at the whole ranking: a source just
    below the cut still counts for something, and one at the bottom counts
    for almost nothing.
    """
    counts = sources.sum(1)
    order = scores.argsort(dim=1, descending=True)
    hits = sources.gather(1, order).to(torch.float64)
    precision = hits.cumsum(1) / torch.arange(1, scores.shape[1] + 1, dtype=torch.float64)
    total = (precision * hits).sum(1)
    return torch.where(counts > 0, total / counts, torch.full_like(total, float("nan")))


def accuracy(outputs: Tensor, targets: Tensor) -> float:
    """The share of counted positions at which the most probable class is the target."""
    counted = targets != IGNORE
    correct = (outputs.argmax(-1) == targets) & counted
    return float(correct.sum() / counted.sum())


def measured_effects(masks: Tensor, measurements: Tensor) -> Tensor:
    """What including each example did to each query, read off retraining directly.

    For every example, the mean measurement of the models that trained on it
    minus the mean of those that did not: `[queries, examples]`. The subsets
    are random, so everything else averages out as the number of models
    grows. This needs no attribution method, which is what makes it useful
    for checking one, and costs as many retrainings as it takes for the
    difference of two means to be less noisy than the effect.
    """
    inside = masks.to(torch.float64)
    outside = 1.0 - inside
    measured = measurements.to(torch.float64)
    return (measured.T @ inside) / inside.sum(0) - (measured.T @ outside) / outside.sum(0)
