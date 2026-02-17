"""Every attribution method under one name each."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from culprit.attribution.base import Attributor
from culprit.attribution.exact import ExactInfluence
from culprit.attribution.influence import (
    CGInfluence,
    EKFACInfluence,
    LiSSAInfluence,
    LowRankInfluence,
)
from culprit.attribution.tracin import TracIn
from culprit.attribution.trak import TRAK
from culprit.task import Task
from culprit.train import Checkpoint

Factory = Callable[..., Attributor]


def _kfac(task: Task, **options: Any) -> Attributor:
    return EKFACInfluence(task, eigenvalue_correction=False, **options)


def _graddot(task: Task, **options: Any) -> Attributor:
    return TracIn(task, None, **options)


# Methods that are built from the final parameters alone.
METHODS: dict[str, Factory] = {
    "exact": ExactInfluence,
    "cg": CGInfluence,
    "lissa": LiSSAInfluence,
    "low-rank": LowRankInfluence,
    "ekfac": EKFACInfluence,
    "kfac": _kfac,
    "graddot": _graddot,
}

# Methods that also use the checkpoints saved along the way.
CHECKPOINT_METHODS: dict[str, Callable[..., Attributor]] = {
    "tracin": lambda task, checkpoints, **options: TracIn(task, checkpoints, **options),
    "trak": lambda task, checkpoints, **options: TRAK(
        task, [checkpoint.params for checkpoint in checkpoints], **options
    ),
}


def method_names() -> list[str]:
    """The names `make_attributor` accepts.

    Unrolled differentiation is not among them: it explains a training run
    and is built from the run's configuration and starting point, not from
    a set of trained parameters.
    """
    return [*METHODS, *CHECKPOINT_METHODS]


def make_attributor(
    name: str,
    task: Task,
    checkpoints: Sequence[Checkpoint] | None = None,
    **options: Any,
) -> Attributor:
    """Build the method called `name` for a trained task.

    `checkpoints` are those of the training run. Methods that use the
    trajectory take all of them; a method that does not is unaffected. A
    trajectory method given none falls back to the final parameters.
    """
    if name in METHODS:
        return METHODS[name](task, **options)
    if name in CHECKPOINT_METHODS:
        if checkpoints is None:
            checkpoints = [Checkpoint(step=0, lr=1.0, params=task.params())]
        return CHECKPOINT_METHODS[name](task, list(checkpoints), **options)
    raise KeyError(f"unknown method {name!r}; choose from {', '.join(method_names())}")
