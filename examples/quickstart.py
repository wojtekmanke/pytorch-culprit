"""Which training examples does a prediction rest on, and is the answer right?

    python examples/quickstart.py

Trains a small classifier, scores its training examples for a few queries
with EK-FAC influence, and then checks the scores the only way they can be
checked: by retraining without some of the examples and seeing whether the
predictions move the way the scores said. Runs on a CPU in a few seconds.
"""

from __future__ import annotations

import torch

from culprit import EKFACInfluence
from culprit.evaluate import bootstrap, lds
from culprit.problems import PROBLEMS
from culprit.retrain import RetrainPlan, retrain
from culprit.train import train


def main() -> float:
    # A problem bundles a model, its training set, some queries and how to train.
    problem = PROBLEMS["blobs"](0)
    task, (train_x, train_y), (query_x, query_y) = problem.task, problem.train, problem.queries

    run = train(task, train_x, train_y, problem.config)
    print(f"trained for {len(run.losses)} steps, final loss {run.losses[-10:].mean():.3f}")

    # scores[q, i]: how much including training example i raised the model's
    # log-probability of the right answer for query q.
    scores = EKFACInfluence(task, damping=1e-2).fit(train_x, train_y).score(query_x, query_y)
    print(f"scores: {tuple(scores.shape)} = (queries, training examples)")

    query = 0
    order = scores[query].argsort(descending=True)
    print(f"\nquery {query} has label {int(query_y[query])}")
    for name, picks in (("most helpful", order[:3]), ("most harmful", order[-3:])):
        for index in picks.tolist():
            print(
                f"  {name}: training example {index:3d}, label {int(train_y[index])}, "
                f"score {scores[query, index]:+.2e}"
            )

    # The check. Train 64 models, each on a random half of the training set,
    # and record what each one says about every query.
    retrained = retrain(
        task, problem.train, problem.queries, problem.config, RetrainPlan(models=64, together=64)
    )
    # If the scores are right, a model's answer is predicted by the sum of
    # the scores of the examples it was trained on. The linear datamodeling
    # score is the rank correlation between the two, per query.
    per_query = lds(scores, retrained.masks, retrained.measurements)
    interval = bootstrap(per_query)
    print(f"\nlinear datamodeling score against 64 retrained models: {interval}")

    shuffled = scores[
        :, torch.randperm(scores.shape[1], generator=torch.Generator().manual_seed(0))
    ]
    control = bootstrap(lds(shuffled, retrained.masks, retrained.measurements))
    print(f"the same scores assigned to the wrong examples:        {control}")
    return interval.mean


if __name__ == "__main__":
    main()
