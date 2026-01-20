"""A training loop that gives the same parameters every time it is run.

Attribution is checked against retraining, so retraining has to be a
function of its inputs: the initial parameters, the set of examples and one
seed for the order they are visited in. Nothing here reads the global random
state.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from culprit.task import Params, Task


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adamw"  # or "sgd"
    momentum: float = 0.0  # sgd only
    schedule: str = "linear"  # decay to zero after warmup, or "constant"
    warmup_steps: int = 0
    seed: int = 0  # decides the order of the examples and nothing else
    checkpoint_every: int = 0  # in steps; 0 keeps only the final parameters


@dataclass(frozen=True)
class Checkpoint:
    step: int  # number of updates applied
    lr: float  # learning rate of the most recent update
    params: Params


@dataclass
class TrainRun:
    config: TrainConfig
    losses: Tensor  # [steps], mean loss of each batch before its update
    checkpoints: list[Checkpoint] = field(default_factory=list)

    @property
    def final(self) -> Params:
        return self.checkpoints[-1].params

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for checkpoint in self.checkpoints:
            save_file(
                {name: value.cpu().contiguous() for name, value in checkpoint.params.items()},
                directory / f"step-{checkpoint.step:07d}.safetensors",
            )
        manifest = {
            "config": asdict(self.config),
            "losses": self.losses.tolist(),
            "checkpoints": [{"step": c.step, "lr": c.lr} for c in self.checkpoints],
        }
        (directory / "run.json").write_text(json.dumps(manifest, indent=1))

    @classmethod
    def load(cls, directory: Path) -> TrainRun:
        manifest = json.loads((directory / "run.json").read_text())
        checkpoints = [
            Checkpoint(
                step=entry["step"],
                lr=entry["lr"],
                params=load_file(directory / f"step-{entry['step']:07d}.safetensors"),
            )
            for entry in manifest["checkpoints"]
        ]
        return cls(
            config=TrainConfig(**manifest["config"]),
            losses=torch.tensor(manifest["losses"]),
            checkpoints=checkpoints,
        )


def batch_schedule(pool: Tensor | int, batch_size: int, epochs: int, seed: int) -> Tensor:
    """Which examples each step trains on, as `[steps, batch_size]` indices.

    Every epoch is a fresh permutation of `pool` cut into full batches; the
    remainder of each permutation is dropped, so the result is rectangular.
    `pool` is a number of examples or the indices of a subset of them.
    """
    indices = torch.arange(pool) if isinstance(pool, int) else pool.cpu()
    per_epoch = len(indices) // batch_size
    if per_epoch == 0:
        raise ValueError(f"{len(indices)} examples do not fill a batch of {batch_size}")
    generator = torch.Generator().manual_seed(seed)
    epochs_rows = []
    for _ in range(epochs):
        order = indices[torch.randperm(len(indices), generator=generator)]
        epochs_rows.append(order[: per_epoch * batch_size].view(per_epoch, batch_size))
    return torch.cat(epochs_rows)


def learning_rate(step: int, total_steps: int, config: TrainConfig) -> float:
    """The learning rate of update number `step`, counting from zero."""
    if step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps
    if config.schedule == "constant":
        return config.lr
    if config.schedule == "linear":
        remaining = total_steps - step
        return config.lr * remaining / max(1, total_steps - config.warmup_steps)
    raise ValueError(f"unknown schedule {config.schedule!r}")


def _optimizer(params: Sequence[Tensor], config: TrainConfig) -> torch.optim.Optimizer:
    if config.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
        )
    raise ValueError(f"unknown optimizer {config.optimizer!r}")


def _snapshot(task: Task) -> Params:
    return {name: value.clone() for name, value in task.params().items()}


def train(
    task: Task,
    inputs: Tensor,
    targets: Tensor,
    config: TrainConfig,
    *,
    pool: Tensor | None = None,
) -> TrainRun:
    """Train `task.model` in place on `inputs[pool]`, or on everything.

    The parameters named by the task are updated and the rest are left alone.
    The last checkpoint of the returned run always holds the final parameters.
    """
    named = dict(task.model.named_parameters())
    trained = [named[name] for name in task.params()]
    optimizer = _optimizer(trained, config)
    schedule = batch_schedule(
        len(inputs) if pool is None else pool, config.batch_size, config.epochs, config.seed
    ).to(inputs.device)
    total = len(schedule)

    losses = torch.empty(total)
    checkpoints: list[Checkpoint] = []
    task.model.train()
    for step, batch in enumerate(schedule):
        lr = learning_rate(step, total, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss = task.loss(task.model(inputs[batch]), targets[batch]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        # Moved before it is stored: writing a GPU tensor straight into a
        # double-precision CPU tensor silently writes nothing (see pitfalls.md).
        losses[step] = loss.detach().cpu()
        done = step + 1
        if done == total or (config.checkpoint_every and done % config.checkpoint_every == 0):
            checkpoints.append(Checkpoint(step=done, lr=lr, params=_snapshot(task)))
    task.model.eval()
    return TrainRun(config=config, losses=losses, checkpoints=checkpoints)
