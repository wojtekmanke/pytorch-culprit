"""Whether K-FAC and EK-FAC help conjugate gradient, counted in iterations.

    python -m benchmarks.preconditioning

writes `results/benchmarks/preconditioning.jsonl`. Everything is in double
precision on the CPU and the models are at their random initialisation, so
the counts depend on nothing but the seeds.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from benchmarks.common import write_records
from culprit.curvature import dense_ggn, kfac
from culprit.data.blobs import gaussian_blobs
from culprit.flat import Layout
from culprit.grads import gradient_matrix
from culprit.linalg import DenseOperator
from culprit.linalg.solvers import conjugate_gradient
from culprit.losses import IGNORE
from culprit.models import GPT, MLP, GPTConfig
from culprit.task import Task

GPT_CONFIG = GPTConfig(vocab_size=11, block_size=6, n_layer=2, n_head=2, d_model=8, mlp_ratio=2)
DAMPINGS = (1e-1, 1e-2, 1e-3, 1e-4)


def gpt_problem(n: int) -> tuple[Task, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(
        GPT_CONFIG.vocab_size, (n, GPT_CONFIG.block_size + 1), generator=generator
    )
    x, y = tokens[:, :-1], tokens[:, 1:].clone()
    y[0, 3:] = IGNORE
    return Task(GPT(GPT_CONFIG).double()), x, y


def mlp_problem(n: int, scaled: bool) -> tuple[Task, Tensor, Tensor]:
    x, y = gaussian_blobs(n, dim=10, classes=3)
    x = x.double()
    if scaled:
        # Features whose scales span two orders of magnitude: the kind of
        # ill-conditioning that lives in the input factor of K-FAC.
        x = x * torch.logspace(-1, 1, 10, dtype=torch.float64)
    return Task(MLP([10, 16, 3]).double()), x, y


PROBLEMS = {
    "gpt, 12 examples": lambda: gpt_problem(12),
    "gpt, 60 examples": lambda: gpt_problem(60),
    "mlp, 200 examples": lambda: mlp_problem(200, scaled=False),
    "mlp, 200 examples, badly scaled inputs": lambda: mlp_problem(200, scaled=True),
    "mlp, 2000 examples, badly scaled inputs": lambda: mlp_problem(2000, scaled=True),
}


def exact_blocks_inverse(
    ggn: Tensor, layout: Layout, groups: list[tuple[str, ...]], damping: float
) -> Tensor:
    """The inverse of the damped matrix with everything between layers set to zero.

    No block-diagonal approximation can do better than the true blocks, so
    this separates what K-FAC loses by approximating a block from what any
    method loses by dropping the interactions between layers.
    """
    inverse = torch.zeros_like(ggn)
    slices = layout.slices()
    for names in groups:
        index = torch.cat([torch.arange(slices[n].start, slices[n].stop) for n in names])
        block = ggn[index][:, index] + damping * torch.eye(len(index), dtype=ggn.dtype)
        inverse[index[:, None], index[None, :]] = torch.linalg.inv(block)
    return inverse


def main() -> None:
    records: list[dict[str, Any]] = []
    for name, make in PROBLEMS.items():
        task, x, y = make()
        params = task.params()
        ggn = dense_ggn(task, params, x, y)
        eigenvalues = torch.linalg.eigvalsh(ggn)
        rank = int((eigenvalues > 1e-10 * eigenvalues[-1]).sum())
        rhs = gradient_matrix(task, params, x[:3], y[:3], measure=True).T
        factored = {
            "K-FAC": kfac(task, params, x, y),
            "EK-FAC": kfac(task, params, x, y, eigenvalue_correction=True),
        }
        groups = [block.names for block in factored["K-FAC"].blocks.values()]
        for damping in DAMPINGS:
            operator = DenseOperator(ggn + damping * torch.eye(len(ggn), dtype=torch.float64))
            preconditioners = {
                "none": None,
                **{label: curvature.inverse(damping) for label, curvature in factored.items()},
                "exact blocks": DenseOperator(
                    exact_blocks_inverse(ggn, Layout.of(params), groups, damping)
                ),
            }
            for label, preconditioner in preconditioners.items():
                _, info = conjugate_gradient(
                    operator, rhs, tol=1e-8, max_iterations=20_000, preconditioner=preconditioner
                )
                record = {
                    "problem": name,
                    "parameters": len(ggn),
                    "rank": rank,
                    "damping": damping,
                    "preconditioner": label,
                    "iterations": info.iterations,
                    "converged": info.converged,
                }
                records.append(record)
                print(record)
    print(write_records("preconditioning", records))


if __name__ == "__main__":
    main()
