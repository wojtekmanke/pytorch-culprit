"""Grade every attribution method against retraining on one problem.

    python -m experiments.compare digits --device mps

trains the model, retrains it on random subsets, scores the queries with
each method over a small grid of its options, and writes one record per run
to `results/compare/<problem>.jsonl`. Options are chosen on the first half
of the queries and every number that is reported is from the second half.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from benchmarks.common import machine
from torch import Tensor

from culprit.attribution import make_attributor
from culprit.ensemble import train_ensemble
from culprit.evaluate import bootstrap, lds, spearman
from culprit.problems import PROBLEMS, Problem
from culprit.retrain import Retraining, RetrainPlan, retrain
from culprit.train import Checkpoint, TrainRun, train

RESULTS = Path(__file__).resolve().parent.parent / "results"
DAMPINGS = (1e-4, 1e-3, 1e-2, 1e-1)

Spec = tuple[str, str, str, list[dict[str, Any]]]  # label, method, checkpoints, option grid


def specs(batch: int) -> list[Spec]:
    """What is run: a label, the method, which parameters it gets, and its grid."""
    big, small = {"batch_size": batch}, {"batch_size": batch // 4}
    damped = [{"damping": damping} for damping in DAMPINGS]
    return [
        ("exact influence", "exact", "final", [{**d, **small} for d in damped]),
        ("EK-FAC", "ekfac", "final", [{**d, **big} for d in damped]),
        ("K-FAC", "kfac", "final", [{**d, **big} for d in damped]),
        *(
            (
                f"conjugate gradient, {iterations} iterations",
                "cg",
                "final",
                [{**d, **big, "max_iterations": iterations, "chunk_size": 50} for d in damped],
            )
            for iterations in (10, 100)
        ),
        (
            "LiSSA, 500 steps",
            "lissa",
            "final",
            [{**d, **big, "steps": 500, "sample_size": 64} for d in damped],
        ),
        *(
            (
                f"low rank, {rank} eigenpairs",
                "low-rank",
                "final",
                [{**d, **big, "rank": rank} for d in damped],
            )
            for rank in (16, 128)
        ),
        ("gradient dot product", "graddot", "final", [big]),
        ("TracIn over checkpoints", "tracin", "checkpoints", [big]),
        *(
            (
                f"TRAK, {label}",
                "trak",
                which,
                [{**d, **big, "projection_size": 2048} for d in damped],
            )
            for label, which in (
                ("final model", "final"),
                ("checkpoints", "checkpoints"),
                ("independent models", "independent"),
            )
        ),
    ]


def on_device(problem: Problem, device: str) -> Problem:
    problem.task.model.to(device)
    return replace(
        problem,
        train=(problem.train[0].to(device), problem.train[1].to(device)),
        queries=(problem.queries[0].to(device), problem.queries[1].to(device)),
    )


def independent_models(problem: Problem, count: int) -> list[Checkpoint]:
    """`count` models trained on everything, differing in their seed alone."""
    n = len(problem.train[0])
    seeds = [10_000 + index for index in range(count)]
    run = train_ensemble(
        problem.task,
        *problem.train,
        problem.config,
        pools=torch.arange(n).repeat(count, 1),
        init_seeds=seeds,
        order_seeds=seeds,
    )
    return [
        Checkpoint(
            step=0, lr=1.0, params={name: value[index] for name, value in run.params.items()}
        )
        for index in range(count)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("problem", choices=sorted(PROBLEMS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--models", type=int, default=500)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=8, help="models per subset; even")
    parser.add_argument("--together", type=int, default=250)
    parser.add_argument("--independent", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    problem = on_device(PROBLEMS[args.problem](args.seed), args.device)
    task = problem.task

    # The model whose outputs are being attributed.
    run: TrainRun = train(task, *problem.train, problem.config)
    accuracy = float(
        (task.model(problem.queries[0]).argmax(-1) == problem.queries[1]).float().mean()
    )
    print(f"trained: loss {run.losses[-20:].mean():.4f}, query accuracy {accuracy:.3f}", flush=True)
    final = [Checkpoint(step=run.checkpoints[-1].step, lr=1.0, params=run.final)]
    parameter_sets = {"final": final, "checkpoints": run.checkpoints}

    # The ground truth: every subset several times over, from different seeds.
    # What a single retrained model measures is mostly its seed; the mean
    # over seeds is what its subset is responsible for.
    retrainings: list[Retraining] = []
    started = time.perf_counter()
    for repeat in range(args.repeats):
        plan = RetrainPlan(args.models, args.fraction, args.seed, repeat, args.together)
        result = retrain(task, problem.train, problem.queries, problem.config, plan)
        result.save(RESULTS / "retrain" / args.problem / f"repeat-{repeat}")
        retrainings.append(result)
        print(f"  repeat {repeat + 1} of {args.repeats} retrained", flush=True)
    retrain_seconds = time.perf_counter() - started
    masks = retrainings[0].masks
    measured = torch.stack([result.measurements for result in retrainings])
    target = measured.mean(0)
    # How far the target can be trusted: split the repeats in two and see
    # whether the two halves order the subsets the same way.
    single = spearman(measured[0], measured[1])
    halves = spearman(measured[0::2].mean(0), measured[1::2].mean(0))
    print(
        f"retrained {args.repeats} x {args.models} models in {retrain_seconds:.0f}s; agreement "
        f"between two models {single.mean():.3f}, between two halves {halves.mean():.3f}",
        flush=True,
    )

    started = time.perf_counter()
    parameter_sets["independent"] = independent_models(problem, args.independent)
    independent_seconds = time.perf_counter() - started

    half = len(problem.queries[0]) // 2
    validation, test = slice(0, half), slice(half, None)
    records: list[dict[str, Any]] = [
        {
            "kind": "context",
            "problem": args.problem,
            "device": args.device,
            "machine": machine(),
            "torch": torch.__version__,
            "examples": len(problem.train[0]),
            "queries_validation": half,
            "queries_test": len(problem.queries[0]) - half,
            "parameters": sum(value.numel() for value in task.params().values()),
            "models": args.models,
            "fraction": args.fraction,
            "query_accuracy": accuracy,
            "final_loss": float(run.losses[-20:].mean()),
            "retrain_seconds": retrain_seconds,
            "independent_models": args.independent,
            "independent_seconds": independent_seconds,
            "repeats": args.repeats,
            "agreement_single": str(bootstrap(single[test])),
            "agreement_halves": str(bootstrap(halves[test])),
        }
    ]

    for label, method, which, grid in specs(args.batch):
        runs = []
        for options in grid:
            attributor = make_attributor(method, task, parameter_sets[which], **options)
            started = time.perf_counter()
            attributor.fit(*problem.train)
            fitted = time.perf_counter()
            scores: Tensor = attributor.score(*problem.queries)
            finished = time.perf_counter()
            per_query = lds(scores, masks, target)
            interval = bootstrap(per_query[test])
            shown = {k: v for k, v in options.items() if k not in ("batch_size", "chunk_size")}
            runs.append(
                {
                    "kind": "run",
                    "problem": args.problem,
                    "label": label,
                    "method": method,
                    "options": shown,
                    "lds_validation": float(per_query[validation].mean()),
                    "lds_test": interval.mean,
                    "lds_low": interval.low,
                    "lds_high": interval.high,
                    "fit_seconds": fitted - started,
                    "score_seconds": finished - fitted,
                    "selected": False,
                }
            )
            print(
                f"{label:36s} {json.dumps(shown):48s} "
                f"validation {runs[-1]['lds_validation']:+.3f}  test {interval}  "
                f"fit {runs[-1]['fit_seconds']:6.1f}s  score {runs[-1]['score_seconds']:6.1f}s",
                flush=True,
            )
        max(runs, key=lambda record: record["lds_validation"])["selected"] = True
        records.extend(runs)

    path = RESULTS / "compare" / f"{args.problem}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    print(path)


if __name__ == "__main__":
    main()
