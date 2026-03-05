"""What half precision does to attribution scores on the GPU.

    python -m benchmarks.autocast

writes `results/benchmarks/autocast.jsonl`. Each method scores the same
queries three ways: in float32; under `torch.autocast` in float16; and
under autocast with the loss and the measurement multiplied by a power of
two first, which is what a gradient scaler does for training. The scores
are compared with the float32 ones by rank, query by query.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor

from benchmarks.common import synchronize, write_records
from culprit.attribution import make_attributor
from culprit.evaluate import spearman
from culprit.losses import CrossEntropy, Margin
from culprit.problems import PROBLEMS
from culprit.task import Task
from culprit.train import train

DEVICE = "mps"
SCALE = 4096.0
METHODS: dict[str, tuple[str, dict[str, Any]]] = {
    "gradient dot product": ("graddot", {}),
    "EK-FAC": ("ekfac", {"damping": 1e-3}),
    "TRAK": ("trak", {"projection_size": 2048, "damping": 1e-3}),
}


@dataclass(frozen=True)
class ScaledCrossEntropy(CrossEntropy):
    factor: float = 1.0

    def __call__(self, outputs: Tensor, targets: Tensor) -> Tensor:
        return self.factor * super().__call__(outputs, targets)


@dataclass(frozen=True)
class ScaledMargin(Margin):
    factor: float = 1.0

    def __call__(self, outputs: Tensor, targets: Tensor) -> Tensor:
        return self.factor * super().__call__(outputs, targets)


def scores(task: Task, method: str, options: dict[str, Any], data: Any, half: bool) -> Tensor:
    train_set, queries = data
    with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=half):
        attributor = make_attributor(method, task, batch_size=200, **options).fit(*train_set)
        return attributor.score(*queries)


def main() -> None:
    records: list[dict[str, Any]] = []
    for name in ("digits", "facts"):
        problem = PROBLEMS[name](0)
        problem.task.model.to(DEVICE)
        train_set = (problem.train[0].to(DEVICE), problem.train[1].to(DEVICE))
        queries = (problem.queries[0][:64].to(DEVICE), problem.queries[1][:64].to(DEVICE))
        train(problem.task, *train_set, problem.config)
        data = (train_set, queries)
        scaled = replace(problem.task, loss=ScaledCrossEntropy(SCALE), measure=ScaledMargin(SCALE))
        for label, (method, options) in METHODS.items():
            reference = scores(problem.task, method, options, data, half=False)
            # Scaling the loss by s scales the curvature by s squared, so an
            # absolute damping has to follow it to mean the same thing.
            scaled_options = dict(options)
            if method == "ekfac":
                scaled_options["damping"] = options["damping"] * SCALE**2
            settings = {
                "float32": (problem.task, options, False),
                "autocast to float16": (problem.task, options, True),
                f"autocast to float16, losses times {SCALE:g}": (scaled, scaled_options, True),
            }
            for setting, (task, setting_options, half) in settings.items():
                record: dict[str, Any] = {"problem": name, "method": label, "setting": setting}
                try:
                    # Once unmeasured: the first half-precision call compiles
                    # its kernels, which is not what is being timed.
                    scores(task, method, setting_options, data, half)
                    synchronize(DEVICE)
                    started = time.perf_counter()
                    result = scores(task, method, setting_options, data, half)
                    synchronize(DEVICE)
                    record["seconds"] = time.perf_counter() - started
                    finite = torch.isfinite(result)
                    record["finite"] = float(finite.double().mean())
                    cleaned = torch.where(finite, result, torch.zeros_like(result))
                    record["rank_agreement"] = float(spearman(cleaned.T, reference.T).mean())
                    record["zero"] = float((cleaned == 0).double().mean())
                except Exception as error:  # noqa: BLE001 - the error is the measurement
                    message = str(error).strip().splitlines()[0] if str(error).strip() else ""
                    record["error"] = f"{type(error).__name__}: {message}"[:200]
                records.append(record)
                print(record, flush=True)
    print(write_records("autocast", records))


if __name__ == "__main__":
    main()
