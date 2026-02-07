"""The command line: train a problem, score it with a method, rebuild the reports.

culprit methods
culprit train blobs --out runs/blobs
culprit score runs/blobs --method ekfac --out runs/blobs/ekfac
culprit report
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from culprit.attribution.registry import make_attributor, method_names
from culprit.evaluate import accuracy
from culprit.problems import PROBLEMS, Problem
from culprit.report import write_reports
from culprit.retrain import RetrainPlan, retrain
from culprit.train import TrainRun, train


def _options(pairs: Sequence[str]) -> dict[str, Any]:
    """Turn `name=value` arguments into keyword arguments, reading values as JSON."""
    options: dict[str, Any] = {}
    for pair in pairs:
        name, separator, value = pair.partition("=")
        if not separator:
            raise SystemExit(f"expected name=value, got {pair!r}")
        try:
            options[name] = json.loads(value)
        except json.JSONDecodeError:
            options[name] = value
    return options


def _problem(name: str, seed: int, device: str) -> Problem:
    problem = PROBLEMS[name](seed)
    problem.task.model.to(device)
    return replace(
        problem,
        train=(problem.train[0].to(device), problem.train[1].to(device)),
        queries=(problem.queries[0].to(device), problem.queries[1].to(device)),
    )


def _train(args: argparse.Namespace) -> None:
    problem = _problem(args.problem, args.seed, args.device)
    run = train(problem.task, *problem.train, problem.config)
    run.save(args.out)
    meta = {"problem": args.problem, "seed": args.seed}
    (args.out / "problem.json").write_text(json.dumps(meta))
    correct = accuracy(problem.task.model(problem.queries[0]), problem.queries[1])
    print(f"trained {args.problem} for {len(run.losses)} steps")
    print(f"final loss {run.losses[-10:].mean():.4f}, query accuracy {correct:.3f}")
    print(f"saved {len(run.checkpoints)} checkpoints to {args.out}")


def _score(args: argparse.Namespace) -> None:
    meta = json.loads((args.run / "problem.json").read_text())
    problem = _problem(meta["problem"], meta["seed"], args.device)
    run = TrainRun.load(args.run)
    # Put the trained parameters into the model the task wraps.
    named = dict(problem.task.model.named_parameters())
    with torch.no_grad():
        for name, value in run.final.items():
            named[name].copy_(value)
    problem.task.model.eval()

    options = _options(args.option)
    attributor = make_attributor(args.method, problem.task, run.checkpoints, **options)
    started = time.perf_counter()
    attributor.fit(*problem.train)
    fitted = time.perf_counter()
    scores = attributor.score(*problem.queries)
    finished = time.perf_counter()

    args.out.mkdir(parents=True, exist_ok=True)
    save_file({"scores": scores.contiguous()}, args.out / "scores.safetensors")
    record = {
        **meta,
        "method": args.method,
        "options": options,
        "queries": scores.shape[0],
        "examples": scores.shape[1],
        "fit_seconds": fitted - started,
        "score_seconds": finished - fitted,
    }
    (args.out / "scores.json").write_text(json.dumps(record, indent=1))
    top = scores[0].topk(min(5, scores.shape[1]))
    print(f"{args.method}: {scores.shape[0]} queries against {scores.shape[1]} examples")
    print(f"fit {record['fit_seconds']:.2f}s, score {record['score_seconds']:.2f}s")
    print("most supportive examples for the first query:")
    for value, index in zip(top.values.tolist(), top.indices.tolist(), strict=True):
        print(f"  example {index:5d}  score {value:+.3e}")


def _retrain(args: argparse.Namespace) -> None:
    problem = _problem(args.problem, args.seed, args.device)
    plan = RetrainPlan(
        models=args.models, fraction=args.fraction, seed=args.seed, together=args.together
    )
    started = time.perf_counter()
    result = retrain(
        problem.task,
        problem.train,
        problem.queries,
        problem.config,
        plan,
        progress=lambda done, total: print(f"  {done} of {total} models", flush=True),
    )
    result.save(args.out)
    (args.out / "problem.json").write_text(json.dumps({"problem": args.problem, "seed": args.seed}))
    kept = int(result.masks[0].sum())
    print(f"retrained {args.problem}: {plan.models} models on {kept} examples each")
    print(f"{time.perf_counter() - started:.1f}s, saved to {args.out}")


def _methods(args: argparse.Namespace) -> None:
    for name in method_names():
        print(name)


def _report(args: argparse.Namespace) -> None:
    for path in write_reports(args.results, args.docs):
        print(path)


def load_scores(directory: Path) -> torch.Tensor:
    """The score matrix written by `culprit score`."""
    return load_file(directory / "scores.safetensors")["scores"]


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="culprit", description=__doc__.splitlines()[0])
    commands = root.add_subparsers(required=True, metavar="command")

    methods = commands.add_parser("methods", help="list the attribution methods")
    methods.set_defaults(handler=_methods)

    training = commands.add_parser("train", help="train a named problem and save the run")
    training.add_argument("problem", choices=sorted(PROBLEMS))
    training.add_argument("--out", type=Path, required=True)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--device", default="cpu")
    training.set_defaults(handler=_train)

    scoring = commands.add_parser("score", help="score a trained run with one method")
    scoring.add_argument("run", type=Path, help="a directory written by `culprit train`")
    scoring.add_argument("--method", choices=method_names(), required=True)
    scoring.add_argument("--out", type=Path, required=True)
    scoring.add_argument("--device", default="cpu")
    scoring.add_argument(
        "-o",
        "--option",
        action="append",
        default=[],
        metavar="name=value",
        help="an option of the method, for example -o damping=0.01",
    )
    scoring.set_defaults(handler=_score)

    retraining = commands.add_parser(
        "retrain", help="train many models on random subsets and record the queries"
    )
    retraining.add_argument("problem", choices=sorted(PROBLEMS))
    retraining.add_argument("--out", type=Path, required=True)
    retraining.add_argument("--models", type=int, default=100)
    retraining.add_argument("--fraction", type=float, default=0.5)
    retraining.add_argument("--together", type=int, default=64)
    retraining.add_argument("--seed", type=int, default=0)
    retraining.add_argument("--device", default="cpu")
    retraining.set_defaults(handler=_retrain)

    report = commands.add_parser("report", help="rebuild docs/results from results")
    report.add_argument("--results", type=Path, default=Path("results"))
    report.add_argument("--docs", type=Path, default=Path("docs"))
    report.set_defaults(handler=_report)
    return root


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
