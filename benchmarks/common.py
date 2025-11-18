"""Timing and record keeping shared by the benchmark scripts."""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

RESULTS = Path(__file__).resolve().parent.parent / "results" / "benchmarks"


def devices() -> list[str]:
    return ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]


def synchronize(device: str) -> None:
    """Wait for queued GPU work, without which a timer measures only the dispatch."""
    if device == "mps":
        torch.mps.synchronize()


def seconds(fn: Callable[[], Any], device: str, repeats: int = 7, warmup: int = 2) -> float:
    """Median wall-clock time of `fn` after a few untimed calls."""
    for _ in range(warmup):
        fn()
    synchronize(device)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        synchronize(device)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def machine() -> str:
    if platform.system() == "Darwin":
        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        )
        if brand.returncode == 0:
            return brand.stdout.strip()
    return platform.processor() or platform.machine()


def write_records(name: str, records: list[dict[str, Any]]) -> Path:
    """Replace the records of one benchmark with the ones just measured."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    context = {"torch": torch.__version__, "machine": machine()}
    path = RESULTS / f"{name}.jsonl"
    path.write_text("".join(json.dumps({**record, **context}) + "\n" for record in records))
    return path
