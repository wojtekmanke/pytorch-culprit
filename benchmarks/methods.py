"""What each attribution method costs: time on both devices, peak memory on the CPU.

    python -m benchmarks.methods

writes `results/benchmarks/methods.jsonl`. Every method fits 5,000 digits
and scores 200 queries on the 7,690-parameter classifier, with fixed
options, so that the numbers compare the methods and not their tuning.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from functools import partial
from typing import Any

from torch import Tensor

from benchmarks.common import devices, synchronize, write_records
from benchmarks.profile import peak_and_operators
from culprit.attribution import make_attributor
from culprit.problems import PROBLEMS
from culprit.task import Task
from culprit.train import Checkpoint, train

METHODS: dict[str, tuple[str, dict[str, Any]]] = {
    "gradient dot product": ("graddot", {"batch_size": 1000}),
    "TracIn, 5 checkpoints": ("tracin", {"batch_size": 1000}),
    "K-FAC": ("kfac", {"damping": 1e-3, "batch_size": 1000}),
    "EK-FAC": ("ekfac", {"damping": 1e-3, "batch_size": 1000}),
    "TRAK, final model": ("trak", {"damping": 1e-3, "projection_size": 2048, "batch_size": 1000}),
    "low rank, 32 eigenpairs": ("low-rank", {"damping": 1e-2, "rank": 32, "batch_size": 1000}),
    "LiSSA, 100 steps": ("lissa", {"damping": 1e-2, "steps": 100, "sample_size": 64}),
    "conjugate gradient, 10 iterations": (
        "cg",
        {"damping": 1e-2, "max_iterations": 10, "batch_size": 1000, "chunk_size": 50},
    ),
    "exact influence": ("exact", {"damping": 1e-2, "batch_size": 250}),
}


def fit_and_score(
    method: str,
    task: Task,
    checkpoints: Sequence[Checkpoint],
    options: dict[str, Any],
    train_set: tuple[Tensor, Tensor],
    queries: tuple[Tensor, Tensor],
    device: str,
) -> tuple[float, float]:
    """Seconds to fit one method and seconds to score with it."""
    attributor = make_attributor(method, task, checkpoints, **options)
    synchronize(device)
    started = time.perf_counter()
    attributor.fit(*train_set)
    synchronize(device)
    fitted = time.perf_counter()
    attributor.score(*queries)
    synchronize(device)
    return fitted - started, time.perf_counter() - fitted


def main() -> None:
    records: list[dict[str, Any]] = []
    for device in devices():
        problem = PROBLEMS["digits"](0)
        task = problem.task
        task.model.to(device)
        train_set = (problem.train[0].to(device), problem.train[1].to(device))
        queries = (problem.queries[0].to(device), problem.queries[1].to(device))
        run = train(task, *train_set, problem.config)
        for label, (method, options) in METHODS.items():
            checkpoints = run.checkpoints if method == "tracin" else run.checkpoints[-1:]
            timed = partial(
                fit_and_score, method, task, checkpoints, options, train_set, queries, device
            )
            timed()  # once unmeasured, so that nothing is paying for a first call
            fit, score = timed()
            record: dict[str, Any] = {
                "method": label,
                "device": device,
                "examples": len(train_set[0]),
                "queries": len(queries[0]),
                "fit_seconds": fit,
                "score_seconds": score,
            }
            if device == "cpu":
                record["peak_bytes"], _ = peak_and_operators(timed)
            records.append(record)
            print(record, flush=True)
    print(write_records("methods", records))


if __name__ == "__main__":
    main()
