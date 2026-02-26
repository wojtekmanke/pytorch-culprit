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
from safetensors.torch import load_file, save_file
from torch import Tensor

from benchmarks.common import machine
from culprit.attribution import Unrolled, make_attributor
from culprit.ensemble import train_ensemble
from culprit.evaluate import (
    accuracy,
    average_precision,
    bootstrap,
    lds,
    measured_effects,
    source_recall,
    spearman,
)
from culprit.problems import PROBLEMS, Problem
from culprit.retrain import Retraining, RetrainPlan, retrain
from culprit.train import Checkpoint, TrainRun, train

RESULTS = Path(__file__).resolve().parent.parent / "results"
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts" / "compare"
# Wide enough that no method's best damping sits at an end of the grid on any
# of the problems; a narrower grid had EK-FAC choosing its smallest value.
DAMPINGS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)

Spec = tuple[str, str, str, list[dict[str, Any]]]  # label, method, checkpoints, option grid


def specs(batch: int, parameters: int) -> list[Spec]:
    """What is run: a label, the method, which parameters it gets, and its grid."""
    big, small = {"batch_size": batch}, {"batch_size": batch // 4}
    damped = [{"damping": damping} for damping in DAMPINGS]
    listed: list[Spec] = []
    if parameters <= 20_000:
        # The curvature matrix has to fit in memory to be inverted exactly.
        listed.append(("exact influence", "exact", "final", [{**d, **small} for d in damped]))
    listed += [
        ("EK-FAC", "ekfac", "final", [{**d, **big} for d in damped]),
        ("K-FAC", "kfac", "final", [{**d, **big} for d in damped]),
        *(
            (
                f"conjugate gradient, {iterations} iterations",
                "cg",
                "final",
                [{**d, **big, "max_iterations": iterations, "chunk_size": 40} for d in damped],
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
                # Two projection sizes: TRAK's kernel is only well conditioned
                # when there are more training examples than dimensions, and
                # one of the problems has 800 examples.
                [{**d, **big, "projection_size": size} for size in (256, 2048) for d in damped],
            )
            for label, which in (
                ("final model", "final"),
                ("checkpoints", "checkpoints"),
                ("independent models", "independent"),
            )
        ),
    ]
    return listed


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


class Timed(Retraining):
    seconds: float


def retrained(problem: Problem, plan: RetrainPlan, root: Path) -> Timed:
    """Retrain according to `plan`, or read back an earlier run of the same plan.

    Retraining is deterministic, so a stored result of the same plan and
    training configuration is the result. The time it took is stored with it.
    """
    directory = root / f"repeat-{plan.repeat}"
    timing = directory / "seconds.json"
    if timing.exists():
        stored = Retraining.load(directory)
        if stored.plan == plan and stored.config == problem.config:
            result = Timed(**vars(stored))
            result.seconds = json.loads(timing.read_text())
            return result
    started = time.perf_counter()
    fresh = retrain(problem.task, problem.train, problem.queries, problem.config, plan)
    result = Timed(**vars(fresh))
    result.seconds = time.perf_counter() - started
    fresh.save(directory)
    timing.write_text(json.dumps(result.seconds))
    return result


def unrolled_scores(
    problem: Problem, seeds: list[int], segment: int
) -> tuple[Tensor, float, float]:
    """Unrolled scores averaged over training runs that differ in their seed.

    The first seed should be the one of the model being explained. Each run
    is repeated from its initialisation and differentiated through; returns
    the mean scores and the seconds spent training and differentiating. A
    run's scores are deterministic and slow to get, so they are kept under
    `artifacts/` with the time they took and read back when asked for again.
    """
    task = problem.task
    total: Tensor | None = None
    fit_seconds = score_seconds = 0.0
    kept = {name: value.clone() for name, value in task.params().items()}
    for seed in seeds:
        cache = ARTIFACTS / problem.name / f"unrolled-seed-{seed}.safetensors"
        if cache.exists():
            stored = load_file(cache)
            scores, timing = stored["scores"], stored["seconds"]
            fit_seconds += float(timing[0])
            score_seconds += float(timing[1])
        else:
            task.model.reset_parameters(seed)  # type: ignore[operator]
            initial = {name: value.clone() for name, value in task.params().items()}
            config = replace(problem.config, seed=seed)
            attributor = Unrolled(task, config, initial, segment=segment)
            started = time.perf_counter()
            attributor.fit(*problem.train)
            fitted = time.perf_counter()
            scores = attributor.score(*problem.queries)
            finished = time.perf_counter()
            fit_seconds += fitted - started
            score_seconds += finished - fitted
            cache.parent.mkdir(parents=True, exist_ok=True)
            timing = torch.tensor([fitted - started, finished - fitted], dtype=torch.float64)
            save_file({"scores": scores.contiguous(), "seconds": timing}, cache)
        total = scores if total is None else total + scores
    with torch.no_grad():
        for name, value in dict(task.model.named_parameters()).items():
            if name in kept:
                value.copy_(kept[name])
    assert total is not None
    return total / len(seeds), fit_seconds, score_seconds


def source_statistics(problem: Problem, masks: Tensor, target: Tensor) -> dict[str, Any]:
    """What retraining itself says about the examples known to be responsible.

    For each kind of source, the mean measured effect of a source on its own
    query, next to that of every other training example; and how many of the
    sources the measured effects themselves would recall, which is the most
    a method can be asked for.
    """
    if problem.sources is None:
        return {}
    effects = measured_effects(masks, target)
    sources = problem.sources.cpu()
    groups = {"sources": sources, **{k: v.cpu() for k, v in (problem.source_groups or {}).items()}}
    statistics: dict[str, Any] = {
        "effect_of_other_examples": float(effects[~sources].mean()),
        "source_groups": {},
    }
    for name, mask in groups.items():
        statistics["source_groups"][name] = {
            "pairs": int(mask.sum()),
            "mean_effect": float(effects[mask].mean()),
            "recall_by_measured_effects": float(source_recall(effects, mask).nanmean()),
        }
    return statistics


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
    parser.add_argument("--unrolled-runs", type=int, default=4)
    parser.add_argument("--segment", type=int, default=50)
    args = parser.parse_args()

    problem = on_device(PROBLEMS[args.problem](args.seed), args.device)
    task = problem.task

    # The model whose outputs are being attributed.
    run: TrainRun = train(task, *problem.train, problem.config)
    correct = accuracy(task.model(problem.queries[0]), problem.queries[1])
    print(f"trained: loss {run.losses[-20:].mean():.4f}, query accuracy {correct:.3f}", flush=True)
    final = [Checkpoint(step=run.checkpoints[-1].step, lr=1.0, params=run.final)]
    parameter_sets = {"final": final, "checkpoints": run.checkpoints}

    # The ground truth: every subset several times over, from different seeds.
    # What a single retrained model measures is mostly its seed; the mean
    # over seeds is what its subset is responsible for.
    retrainings: list[Retraining] = []
    started = time.perf_counter()
    for repeat in range(args.repeats):
        plan = RetrainPlan(args.models, args.fraction, args.seed, repeat, args.together)
        retrainings.append(retrained(problem, plan, RESULTS / "retrain" / args.problem))
        print(f"  repeat {repeat + 1} of {args.repeats} retrained", flush=True)
    retrain_seconds = sum(result.seconds for result in retrainings)
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
            "query_accuracy": correct,
            "optimizer": problem.config.optimizer,
            "steps": len(run.losses),
            "final_loss": float(run.losses[-20:].mean()),
            "retrain_seconds": retrain_seconds,
            "independent_models": args.independent,
            "independent_seconds": independent_seconds,
            "repeats": args.repeats,
            **source_statistics(problem, masks, target),
            "agreement_single": str(bootstrap(single[test])),
            "agreement_halves": str(bootstrap(halves[test])),
        }
    ]

    def graded(
        label: str, method: str, options: dict[str, Any], scores: Tensor, fit: float, score: float
    ) -> dict[str, Any]:
        per_query = lds(scores, masks, target)
        interval = bootstrap(per_query[test])
        shown = {k: v for k, v in options.items() if k not in ("batch_size", "chunk_size")}
        record = {
            "kind": "run",
            "problem": args.problem,
            "label": label,
            "method": method,
            "options": shown,
            "lds_validation": float(per_query[validation].mean()),
            "lds_test": interval.mean,
            "lds_low": interval.low,
            "lds_high": interval.high,
            "fit_seconds": fit,
            "score_seconds": score,
            "selected": False,
        }
        if problem.sources is not None:
            # Where the responsible examples are known, also ask for them directly.
            sources = problem.sources.cpu()[test]
            record["recall"] = float(source_recall(scores[test], sources).nanmean())
            record["average_precision"] = float(average_precision(scores[test], sources).nanmean())
            record["recall_by_group"] = {
                name: float(source_recall(scores[test], mask.cpu()[test]).nanmean())
                for name, mask in (problem.source_groups or {}).items()
            }
        print(
            f"{label:36s} {json.dumps(shown):48s} "
            f"validation {record['lds_validation']:+.3f}  test {interval}  "
            f"fit {fit:6.1f}s  score {score:6.1f}s",
            flush=True,
        )
        return record

    parameters = sum(value.numel() for value in task.params().values())
    kept: dict[str, Tensor] = {}
    for label, method, which, grid in specs(args.batch, parameters):
        runs, run_scores = [], []
        for options in grid:
            attributor = make_attributor(method, task, parameter_sets[which], **options)
            started = time.perf_counter()
            attributor.fit(*problem.train)
            fitted = time.perf_counter()
            scores: Tensor = attributor.score(*problem.queries)
            finished = time.perf_counter()
            runs.append(graded(label, method, options, scores, fitted - started, finished - fitted))
            run_scores.append(scores)
        best = max(range(len(runs)), key=lambda index: runs[index]["lds_validation"])
        runs[best]["selected"] = True
        kept[label] = run_scores[best].float()
        records.extend(runs)

    # Unrolling explains a training run, not a set of parameters, so it is
    # given the runs: the one being explained, then others from other seeds.
    seeds = [args.seed, *(20_000 + index for index in range(args.unrolled_runs - 1))]
    for count in sorted({1, args.unrolled_runs}):
        scores, fit, score = unrolled_scores(problem, seeds[:count], args.segment)
        label = "unrolled, the run itself" if count == 1 else f"unrolled, {count} runs averaged"
        record = graded(label, "unrolled", {}, scores, fit, score)
        record["selected"] = True
        kept[label] = scores.float()
        records.append(record)

    # The chosen scores of every method, for looking at where they go wrong.
    directory = ARTIFACTS / args.problem
    directory.mkdir(parents=True, exist_ok=True)
    save_file(
        {label: value.contiguous() for label, value in kept.items()},
        directory / "chosen.safetensors",
    )

    path = RESULTS / "compare" / f"{args.problem}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    print(path)


if __name__ == "__main__":
    main()
