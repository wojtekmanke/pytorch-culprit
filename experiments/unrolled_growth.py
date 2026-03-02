"""How sensitive is the end of training to its earlier steps?

    python -m experiments.unrolled_growth --device mps

Unrolled differentiation pulls an adjoint back through training one step
at a time. This records the size of that adjoint on the way: its norm so
many steps before the end, relative to its norm at the end. A ratio that
stays near one says the past is remembered about as strongly as the
present. A ratio that grows without bound says the derivative is dominated
by whatever happened earliest, and in floating point by rounding.

Writes `results/unrolled_growth.jsonl`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from benchmarks.common import machine
from culprit.attribution import Unrolled
from culprit.problems import PROBLEMS
from experiments.compare import on_device

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--queries", type=int, default=16)
    parser.add_argument("--problems", nargs="*", default=["digits-sgd", "digits", "facts"])
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for name in args.problems:
        problem = on_device(PROBLEMS[name](0), args.device)
        task = problem.task
        initial = {key: value.clone() for key, value in task.params().items()}
        attributor = Unrolled(task, problem.config, initial, segment=50).fit(*problem.train)
        queries = (problem.queries[0][: args.queries], problem.queries[1][: args.queries])
        scores = attributor.score(*queries)
        total, end = attributor.adjoint_norms[0]
        for step, norm in attributor.adjoint_norms:
            records.append(
                {
                    "problem": name,
                    "optimizer": problem.config.optimizer,
                    "steps": total,
                    "steps_before_the_end": total - step,
                    "norm_ratio": norm / end,
                    "largest_score": float(scores.abs().max()),
                    "queries": args.queries,
                    "device": args.device,
                    "machine": machine(),
                    "torch": torch.__version__,
                }
            )
        last = records[-1]
        print(
            f"{name}: {total} steps; adjoint at the start of training is "
            f"{last['norm_ratio']:.3g} times its size at the end; "
            f"largest score {last['largest_score']:.3g}",
            flush=True,
        )
    path = RESULTS / "unrolled_growth.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    print(path)


if __name__ == "__main__":
    main()
