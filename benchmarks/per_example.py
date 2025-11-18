"""How long per-example gradients take, three ways.

    python -m benchmarks.per_example

writes `results/benchmarks/per_example.jsonl`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from benchmarks.common import devices, seconds, write_records
from culprit.grads import per_example_grads
from culprit.grads.hooks import hook_per_example_grads
from culprit.models import GPT, MLP, GPTConfig
from culprit.task import Task

GPT_CONFIG = GPTConfig(vocab_size=107, block_size=14, n_layer=2, n_head=2, d_model=64)


def problems(batch: int) -> dict[str, tuple[Task, torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(
        GPT_CONFIG.vocab_size, (batch, GPT_CONFIG.block_size + 1), generator=generator
    )
    return {
        "mlp 196-32-32-10": (
            Task(MLP([196, 32, 32, 10])),
            torch.randn(batch, 196, generator=generator),
            torch.randint(10, (batch,), generator=generator),
        ),
        "gpt 2x64": (Task(GPT(GPT_CONFIG)), tokens[:, :-1], tokens[:, 1:]),
    }


def loop(task: Task, x: torch.Tensor, y: torch.Tensor) -> None:
    params = {name: value.clone().requires_grad_() for name, value in task.params().items()}
    for i in range(len(x)):
        value = task.losses(params, x[i : i + 1], y[i : i + 1]).sum()
        torch.autograd.grad(value, list(params.values()))


def main() -> None:
    records: list[dict[str, Any]] = []
    for device in devices():
        for batch in (64, 256):
            for model, (task, x, y) in problems(batch).items():
                task.model.to(device)
                x, y = x.to(device), y.to(device)
                params = task.params()
                methods: dict[str, Callable[[], Any]] = {
                    "loop": lambda task=task, x=x, y=y: loop(task, x, y),
                    "vmap": lambda task=task, params=params, x=x, y=y: per_example_grads(
                        task.example_loss, params, x, y
                    ),
                    "hooks": lambda task=task, params=params, x=x, y=y: hook_per_example_grads(
                        task, params, x, y
                    ),
                }
                for method, fn in methods.items():
                    # The loop is slow enough that a few repeats give a stable median.
                    repeats = 3 if method == "loop" else 7
                    record = {
                        "model": model,
                        "parameters": sum(value.numel() for value in params.values()),
                        "device": device,
                        "batch": batch,
                        "method": method,
                        "seconds": seconds(fn, device, repeats=repeats),
                    }
                    records.append(record)
                    print(record)
    print(write_records("per_example", records))


if __name__ == "__main__":
    main()
