"""What single precision costs, measured against double precision on the CPU.

    python -m benchmarks.precision

writes `results/benchmarks/precision.jsonl`. MPS has no float64, so the
question is which computations can stay on the GPU in float32 and which have
to be taken to the CPU.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import Tensor

from benchmarks.common import devices, write_records
from culprit.curvature.products import ggn_vp, hvp_forward_over_reverse
from culprit.device import cpu_double
from culprit.flat import Layout
from culprit.grads import gradient_matrix
from culprit.linalg import CurvatureOperator
from culprit.linalg.solvers import conjugate_gradient
from culprit.models import GPT, MLP, GPTConfig
from culprit.task import Task

GPT_CONFIG = GPTConfig(vocab_size=107, block_size=14, n_layer=2, n_head=2, d_model=64)
BATCH = 256
DAMPINGS = (1e-1, 1e-2, 1e-3, 1e-4)


def problems() -> dict[str, tuple[Task, Tensor, Tensor]]:
    """Models and data created in double precision; everything else is a copy of these."""
    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(
        GPT_CONFIG.vocab_size, (BATCH, GPT_CONFIG.block_size + 1), generator=generator
    )
    return {
        "mlp 196-32-32-10": (
            Task(MLP([196, 32, 32, 10]).double()),
            torch.randn(BATCH, 196, generator=generator, dtype=torch.float64),
            torch.randint(10, (BATCH,), generator=generator),
        ),
        "gpt 2x64": (Task(GPT(GPT_CONFIG).double()), tokens[:, :-1], tokens[:, 1:]),
    }


def moved(task: Task, x: Tensor, y: Tensor, device: str) -> tuple[Task, Tensor, Tensor]:
    model = copy.deepcopy(task.model).to(device, torch.float32)
    x = x.to(device, torch.float32) if x.is_floating_point() else x.to(device)
    return Task(model, task.loss, task.measure), x, y.to(device)


def quantities(task: Task, x: Tensor, y: Tensor, flat_direction: Tensor) -> dict[str, Tensor]:
    """Everything whose precision is in question, computed where the task lives."""
    params = task.params()
    layout = Layout.of(params)
    first = next(iter(params.values()))
    direction = layout.unflatten(flat_direction.to(first.device, first.dtype))
    out = {
        "per-example gradients": gradient_matrix(task, params, x, y),
        "Gauss-Newton-vector product": layout.flatten(ggn_vp(task, params, x, y, direction)),
        "Hessian-vector product": layout.flatten(
            hvp_forward_over_reverse(task, params, x, y, direction)
        ),
    }
    # The right-hand side of an influence computation is a gradient.
    rhs = out["per-example gradients"][0]
    tol = 1e-10 if first.dtype == torch.float64 else 1e-6
    for damping in DAMPINGS:
        operator = CurvatureOperator(task, params, x, y).damped(damping)
        solution, info = conjugate_gradient(operator, rhs, tol=tol, max_iterations=2000)
        out[f"CG solve, damping {damping:g}"] = solution
        out[f"CG solve, damping {damping:g} iterations"] = torch.tensor(float(info.iterations))
        out[f"CG solve, damping {damping:g} residual"] = info.residual
    return out


def relative_error(value: Tensor, reference: Tensor) -> float:
    return float((cpu_double(value) - reference).norm() / reference.norm())


def main() -> None:
    records: list[dict[str, Any]] = []
    for model, (task, x, y) in problems().items():
        size = Layout.of(task.params()).size
        direction = torch.randn(size, generator=torch.Generator().manual_seed(1)).double()
        reference = quantities(task, x, y, direction)
        for device in devices():
            single = quantities(*moved(task, x, y, device), direction)
            for name, value in reference.items():
                if name.endswith(("iterations", "residual")):
                    continue
                record: dict[str, Any] = {
                    "model": model,
                    "parameters": size,
                    "device": device,
                    "quantity": name,
                    "relative_error": relative_error(single[name], value),
                }
                if name.startswith("CG"):
                    record["iterations"] = int(single[f"{name} iterations"])
                    record["iterations_double"] = int(reference[f"{name} iterations"])
                    record["residual"] = float(single[f"{name} residual"])
                records.append(record)
                print(record)
    print(write_records("precision", records))


if __name__ == "__main__":
    main()
