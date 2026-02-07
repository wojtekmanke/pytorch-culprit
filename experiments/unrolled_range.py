"""How far from the run it was taken on does the derivative of training hold?

    python -m experiments.unrolled_range digits-sgd --device mps

Unrolled differentiation gives the exact derivative of a measurement with
respect to each example's weight, through one particular training run. This
asks what that derivative predicts. Examples are removed by setting their
weight to zero while everything else about the run, its initialisation and
the order of its batches, is kept. That is the counterfactual the derivative
is a first-order statement about, with none of the noise of a new seed. The
share of examples removed is then grown until the prediction fails.

Writes `results/unrolled_range/<problem>.jsonl`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from benchmarks.common import machine
from torch.func import vmap

from culprit.attribution import ExactInfluence, TracIn, Unrolled
from culprit.attribution.unrolled import weighted_runs
from culprit.evaluate import bootstrap, spearman
from culprit.problems import PROBLEMS
from experiments.compare import on_device

RESULTS = Path(__file__).resolve().parent.parent / "results" / "unrolled_range"
REMOVED = (5, 50, 500, 2500)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("problem", choices=sorted(PROBLEMS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--runs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    problem = on_device(PROBLEMS[args.problem](args.seed), args.device)
    task, (x, y), queries = problem.task, problem.train, problem.queries
    initial = {name: value.clone() for name, value in task.params().items()}

    unrolled = Unrolled(task, problem.config, initial).fit(x, y)
    full = unrolled.params
    methods = {"unrolled": unrolled.score(*queries)}
    # Two methods that know only the final parameters, for comparison.
    methods["exact influence"] = (
        ExactInfluence(task, full, damping=0.01, batch_size=250).fit(x, y).score(*queries)
    )
    with torch.no_grad():
        for name, value in dict(task.model.named_parameters()).items():
            value.copy_(full[name])
    methods["gradient dot product"] = TracIn(task, batch_size=1000).fit(x, y).score(*queries)
    before = task.measures(full, *queries).cpu().double()

    generator = torch.Generator().manual_seed(args.seed)
    records: list[dict[str, Any]] = []
    for removed in REMOVED:
        weights = torch.ones(args.runs, len(x))
        for row in weights:
            row[torch.randperm(len(x), generator=generator)[:removed]] = 0.0
        finals = weighted_runs(task, x, y, problem.config, initial, weights)
        after = vmap(task.measures, in_dims=(0, None, None))(finals, *queries).cpu().double()
        actual = after - before  # [runs, queries]
        for label, scores in methods.items():
            # Removing an example takes away the effect of including it.
            predicted = -(1 - weights).double() @ scores.T
            rank = bootstrap(spearman(predicted, actual))
            # Is the size right as well as the order? The slope of the
            # least-squares line of what happened on what was predicted.
            slope = float((predicted * actual).sum() / (predicted * predicted).sum())
            record = {
                "problem": args.problem,
                "optimizer": problem.config.optimizer,
                "method": label,
                "removed": removed,
                "share": removed / len(x),
                "runs": args.runs,
                "queries": len(queries[0]),
                "rank_correlation": rank.mean,
                "low": rank.low,
                "high": rank.high,
                "slope": slope,
                "machine": machine(),
                "torch": torch.__version__,
            }
            records.append(record)
            print(
                f"{label:22s} removed {removed:5d}  rank correlation {rank}  slope {slope:+.3f}",
                flush=True,
            )
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{args.problem}.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    print(path)


if __name__ == "__main__":
    main()
