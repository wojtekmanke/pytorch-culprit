"""Training data attribution in PyTorch, graded against retraining.

The pieces most code needs are importable from here:

    from culprit import Task, EKFACInfluence

    scores = EKFACInfluence(Task(model)).fit(train_x, train_y).score(query_x, query_y)

`scores[q, i]` estimates how much including training example `i` changed the
measurement of query `q`. Everything else lives in the submodules:
`culprit.curvature`, `culprit.grads`, `culprit.linalg`, `culprit.ensemble`,
`culprit.evaluate`, `culprit.lora` and `culprit.llm`.
"""

from culprit.attribution import (
    TRAK,
    Attributor,
    CGInfluence,
    EKFACInfluence,
    ExactInfluence,
    LiSSAInfluence,
    LowRankInfluence,
    TracIn,
    Unrolled,
    make_attributor,
    method_names,
)
from culprit.losses import IGNORE, CrossEntropy, LogProb, Margin
from culprit.task import Task

__version__ = "0.0.1"

__all__ = [
    "IGNORE",
    "TRAK",
    "Attributor",
    "CGInfluence",
    "CrossEntropy",
    "EKFACInfluence",
    "ExactInfluence",
    "LiSSAInfluence",
    "LogProb",
    "LowRankInfluence",
    "Margin",
    "Task",
    "TracIn",
    "Unrolled",
    "make_attributor",
    "method_names",
]
