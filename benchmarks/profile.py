"""Where the time and the memory go, stage by stage.

    python -m benchmarks.profile

writes `results/benchmarks/profile.jsonl`. Every stage is timed on each
device, and profiled once on the CPU with `torch.profiler` for its peak
memory and its busiest operators.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import torch
from torch import Tensor
from torch.profiler import ProfilerActivity, profile

from benchmarks.common import devices, seconds, write_records
from culprit.curvature.kfac import collect_factors, ekfac
from culprit.curvature.products import ggn_vp, hvp_forward_over_reverse
from culprit.flat import Layout
from culprit.grads import gradient_matrix, per_example_grads
from culprit.grads.ghost import capture, gradient_dots
from culprit.grads.hooks import hook_per_example_grads
from culprit.models import GPT, GPTConfig
from culprit.models.rmsnorm import RMSNorm
from culprit.projection import SparseProjector
from culprit.task import Task

CONFIG = GPTConfig(vocab_size=107, block_size=14, n_layer=2, n_head=2, d_model=64)
BATCH, QUERIES = 256, 16


def peak_and_operators(fn: Callable[[], Any]) -> tuple[int, str]:
    """Peak bytes allocated while `fn` runs on the CPU, and where its time went.

    The profiler attributes each allocation to the operator that made it and
    records frees that happen between operators as events of their own.
    Adding those up in order of time gives the allocation curve to the
    resolution of one operator, and its maximum is the peak.
    """
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as recorded:
        fn()
    running = peak = 0
    for event in sorted(recorded.events(), key=lambda e: e.time_range.start):
        running += event.self_cpu_memory_usage
        peak = max(peak, running)
    averages = recorded.key_averages()
    total = sum(event.self_cpu_time_total for event in averages) or 1
    busiest = sorted(averages, key=lambda e: e.self_cpu_time_total, reverse=True)[:3]
    names = ", ".join(
        f"{event.key.removeprefix('aten::')} {100 * event.self_cpu_time_total / total:.0f}%"
        for event in busiest
    )
    return peak, names


def stages(device: str) -> dict[str, Callable[[], Any]]:
    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(CONFIG.vocab_size, (BATCH, CONFIG.block_size + 1), generator=generator)
    x, y = tokens[:, :-1].to(device), tokens[:, 1:].to(device)
    task = Task(GPT(CONFIG).to(device))
    params = task.params()
    layout = Layout.of(params)
    direction = layout.unflatten(torch.randn(layout.size, generator=generator).to(device))
    projector = SparseProjector(layout.size, 2048)
    grads = gradient_matrix(task, params, x, y)

    def training_step(of: Task = task) -> None:
        of.model.zero_grad(set_to_none=True)
        of.loss(of.model(x), y).mean().backward()

    def dots(strategy: str) -> Tensor:
        queries = capture(task, params, x[:QUERIES], y[:QUERIES], measure=True)
        return gradient_dots(task, params, queries, capture(task, params, x, y), strategy=strategy)  # type: ignore[arg-type]

    def with_rms_norm(fused: bool) -> Task:
        model = GPT(replace(CONFIG, norm="rmsnorm")).to(device)
        for module in model.modules():
            if isinstance(module, RMSNorm):
                module.fused = fused
        return Task(model)

    fused, unfused = with_rms_norm(True), with_rms_norm(False)
    return {
        "forward and backward": training_step,
        "per-example gradients, vmap": lambda: per_example_grads(task.example_loss, params, x, y),
        "per-example gradients, hooks": lambda: hook_per_example_grads(task, params, x, y),
        f"gradient dot products with {QUERIES} queries, as sums over positions": lambda: dots(
            "ghost"
        ),
        f"gradient dot products with {QUERIES} queries, materialised": lambda: dots("materialise"),
        "Gauss-Newton-vector product": lambda: ggn_vp(task, params, x, y, direction),
        "Hessian-vector product": lambda: hvp_forward_over_reverse(task, params, x, y, direction),
        "K-FAC factors": lambda: collect_factors(task, params, x, y),
        "EK-FAC, factors and eigenvalues": lambda: ekfac(task, params, x, y),
        "sparse projection of the gradients to 2048": lambda: projector.project(grads),
        "forward and backward, RMS norm fused": lambda: training_step(fused),
        "forward and backward, RMS norm from plain operations": lambda: training_step(unfused),
    }


def main() -> None:
    records: list[dict[str, Any]] = []
    for device in devices():
        for name, fn in stages(device).items():
            record: dict[str, Any] = {
                "stage": name,
                "device": device,
                "batch": BATCH,
                "seconds": seconds(fn, device),
            }
            if device == "cpu":
                record["peak_bytes"], record["operators"] = peak_and_operators(fn)
            records.append(record)
            print(record)
    print(write_records("profile", records))


if __name__ == "__main__":
    main()
