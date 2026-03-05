"""What `torch.compile` does for the hot paths, and where it gives up.

    python -m benchmarks.compile

writes `results/benchmarks/compile.jsonl`. For each function it records the
time of the eager version, the time of the first compiled call (which
includes compilation) and the steady time afterwards, or the error if the
compiler could not handle the function.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.func import functional_call, grad, jvp, vmap

from benchmarks.common import devices, seconds, synchronize, write_records
from culprit.models import GPT, GPTConfig
from culprit.task import Task

CONFIG = GPTConfig(vocab_size=107, block_size=14, n_layer=2, n_head=2, d_model=64)
BATCH = 256


def functions(device: str) -> dict[str, tuple[Callable[..., Any], tuple[Any, ...]]]:
    """The functions worth compiling, each with the arguments to call it with.

    They are written against plain tensors and dictionaries, without the
    library's own wrappers, so that what is measured is the compiler's
    handling of the transform and not of the code around it.
    """
    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(CONFIG.vocab_size, (BATCH, CONFIG.block_size + 1), generator=generator)
    x, y = tokens[:, :-1].to(device), tokens[:, 1:].to(device)
    task = Task(GPT(CONFIG).to(device))
    model, params = task.model, task.params()
    direction = {name: torch.randn_like(value) for name, value in params.items()}

    def forward(params: dict[str, Tensor], x: Tensor) -> Tensor:
        out: Tensor = functional_call(model, params, (x,))
        return out

    def mean_loss(params: dict[str, Tensor], x: Tensor, y: Tensor) -> Tensor:
        return task.loss(forward(params, x), y).mean()

    def example_loss(params: dict[str, Tensor], x: Tensor, y: Tensor) -> Tensor:
        return task.loss(forward(params, x.unsqueeze(0)), y.unsqueeze(0)).squeeze(0)

    def gradient(params: dict[str, Tensor], x: Tensor, y: Tensor) -> dict[str, Tensor]:
        out: dict[str, Tensor] = grad(mean_loss)(params, x, y)
        return out

    def per_example(params: dict[str, Tensor], x: Tensor, y: Tensor) -> dict[str, Tensor]:
        out: dict[str, Tensor] = vmap(grad(example_loss), in_dims=(None, 0, 0))(params, x, y)
        return out

    def hvp(
        params: dict[str, Tensor], x: Tensor, y: Tensor, v: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        out: dict[str, Tensor] = jvp(lambda p: grad(mean_loss)(p, x, y), (params,), (v,))[1]
        return out

    return {
        "forward pass": (forward, (params, x)),
        "gradient of the mean loss": (gradient, (params, x, y)),
        "per-example gradients, vmap over grad": (per_example, (params, x, y)),
        "Hessian-vector product, jvp over grad": (hvp, (params, x, y, direction)),
    }


def main() -> None:
    records: list[dict[str, Any]] = []
    for device in devices():
        for name, (fn, args) in functions(device).items():
            record: dict[str, Any] = {"function": name, "device": device, "batch": BATCH}
            record["eager_seconds"] = seconds(lambda fn=fn, args=args: fn(*args), device)
            try:
                compiled = torch.compile(fn)
                started = time.perf_counter()
                compiled(*args)
                synchronize(device)
                record["first_call_seconds"] = time.perf_counter() - started
                record["compiled_seconds"] = seconds(
                    lambda compiled=compiled, args=args: compiled(*args), device
                )
            except Exception as error:  # noqa: BLE001 - the error is the measurement
                message = str(error).strip().splitlines()[0] if str(error).strip() else ""
                record["error"] = f"{type(error).__name__}: {message}"[:200]
            records.append(record)
            print(record, flush=True)
    print(write_records("compile", records))


if __name__ == "__main__":
    main()
