"""How many model updates per second, training models together against one at a time.

    python -m benchmarks.ensemble

writes `results/benchmarks/ensemble.jsonl`.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from benchmarks.common import devices, seconds, write_records
from culprit.ensemble import train_ensemble
from culprit.models import GPT, MLP, GPTConfig
from culprit.task import Task
from culprit.train import TrainConfig, train

GPT_CONFIG = GPTConfig(vocab_size=107, block_size=14, n_layer=2, n_head=2, d_model=64)
STEPS, BATCH = 20, 64
CONFIG = TrainConfig(epochs=1, batch_size=BATCH, lr=1e-3)
SIZES = (1, 8, 64, 512)


def problems() -> dict[str, tuple[Task, Tensor, Tensor]]:
    generator = torch.Generator().manual_seed(0)
    n = STEPS * BATCH
    tokens = torch.randint(
        GPT_CONFIG.vocab_size, (n, GPT_CONFIG.block_size + 1), generator=generator
    )
    return {
        "mlp 196-32-32-10": (
            Task(MLP([196, 32, 32, 10])),
            torch.randn(n, 196, generator=generator),
            torch.randint(10, (n,), generator=generator),
        ),
        "gpt 2x64": (Task(GPT(GPT_CONFIG)), tokens[:, :-1], tokens[:, 1:]),
    }


def main() -> None:
    records: list[dict[str, Any]] = []
    for device in devices():
        for model, (task, x, y) in problems().items():
            task.model.to(device)
            x, y = x.to(device), y.to(device)

            def alone(task: Task = task, x: Tensor = x, y: Tensor = y) -> None:
                task.model.reset_parameters(0)  # type: ignore[operator]
                train(task, x, y, CONFIG)

            runs = {"one at a time": (1, alone)}
            for size in SIZES:
                pools = torch.arange(len(x)).repeat(size, 1)

                def together(
                    task: Task = task, x: Tensor = x, y: Tensor = y, pools: Tensor = pools
                ) -> None:
                    count = len(pools)
                    seeds = list(range(count))
                    train_ensemble(
                        task, x, y, CONFIG, pools=pools, init_seeds=seeds, order_seeds=seeds
                    )

                runs[f"{size} together"] = (size, together)
            for label, (size, fn) in runs.items():
                per_step = seconds(fn, device, repeats=3, warmup=1) / STEPS
                record = {
                    "model": model,
                    "device": device,
                    "how": label,
                    "models": size,
                    "batch": BATCH,
                    "seconds_per_step": per_step,
                    "updates_per_second": size / per_step,
                }
                records.append(record)
                print(record)
    print(write_records("ensemble", records))


if __name__ == "__main__":
    main()
