"""The ground truth: retrain on random subsets and see what the models then say.

An attribution method claims to know how a model's output on a query
depends on which examples it was trained on. The direct test is to train on
many different subsets of the examples and record the output each time.
`retrain` does that and keeps two tables: which examples each model saw,
and what each model measured on each query. Everything the evaluation needs
follows from those two.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from culprit.device import cpu_double
from culprit.ensemble import ensemble_measures, pools, random_subsets, train_ensemble
from culprit.task import Task
from culprit.train import TrainConfig


@dataclass(frozen=True)
class RetrainPlan:
    models: int = 100
    fraction: float = 0.5  # of the training set that each model sees
    seed: int = 0  # decides the subsets
    repeat: int = 0  # decides, with the seed, how each model is initialised and ordered
    together: int = 64  # how many models share a pass; bounds memory, not the result

    def model_seed(self, model: int) -> int:
        """The seed of one model's initialisation and of the order of its examples.

        Two plans that differ only in `repeat` train on the same subsets from
        different seeds. Comparing them shows how much of what a model
        measures is decided by its subset and how much by its seed.
        """
        return ((self.seed + 1) * 1_009 + self.repeat) * 100_003 + model


@dataclass
class Retraining:
    plan: RetrainPlan
    config: TrainConfig
    masks: Tensor  # [models, examples], true where the model trained on the example
    measurements: Tensor  # [models, queries], float64
    records: list[dict[str, Any]]  # one per model

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        # Eight examples to the byte: the masks are most of what is stored.
        packed = torch.from_numpy(np.packbits(self.masks.numpy(), axis=1))
        save_file(
            {"masks": packed, "measurements": self.measurements.contiguous()},
            directory / "retraining.safetensors",
        )
        header = {
            "plan": asdict(self.plan),
            "config": asdict(self.config),
            "examples": self.masks.shape[1],
        }
        (directory / "plan.json").write_text(json.dumps(header, indent=1))
        (directory / "models.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in self.records)
        )

    @classmethod
    def load(cls, directory: Path) -> Retraining:
        header = json.loads((directory / "plan.json").read_text())
        tensors = load_file(directory / "retraining.safetensors")
        unpacked = np.unpackbits(tensors["masks"].numpy(), axis=1, count=header["examples"])
        records = [
            json.loads(line) for line in (directory / "models.jsonl").read_text().splitlines()
        ]
        return cls(
            plan=RetrainPlan(**header["plan"]),
            config=TrainConfig(**header["config"]),
            masks=torch.from_numpy(unpacked).bool(),
            measurements=tensors["measurements"],
            records=records,
        )


def retrain(
    task: Task,
    train: tuple[Tensor, Tensor],
    queries: tuple[Tensor, Tensor],
    config: TrainConfig,
    plan: RetrainPlan,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> Retraining:
    """Train `plan.models` models, each on its own random subset, and measure the queries.

    The models are trained `plan.together` at a time. The task's own model
    is not changed.
    """
    inputs, targets = train
    masks = random_subsets(len(inputs), plan.models, plan.fraction, plan.seed)
    subsets = pools(masks)
    measurements = torch.empty(plan.models, len(queries[0]), dtype=torch.float64)
    records: list[dict[str, Any]] = []
    for start in range(0, plan.models, plan.together):
        members = range(start, min(start + plan.together, plan.models))
        seeds = [plan.model_seed(model) for model in members]
        run = train_ensemble(
            task,
            inputs,
            targets,
            config,
            pools=subsets[members.start : members.stop],
            init_seeds=seeds,
            order_seeds=seeds,
        )
        measured = cpu_double(ensemble_measures(task, run.params, *queries))
        measurements[members.start : members.stop] = measured
        tail = max(1, len(run.losses) // 10)
        final_losses = run.losses[-tail:].mean(0)
        for offset, model in enumerate(members):
            records.append(
                {
                    "model": model,
                    "seed": seeds[offset],
                    "examples": int(masks[model].sum()),
                    "final_loss": float(final_losses[offset]),
                    "mean_measurement": float(measured[offset].mean()),
                }
            )
        if progress is not None:
            progress(members.stop, plan.models)
    return Retraining(plan, config, masks, measurements, records)
