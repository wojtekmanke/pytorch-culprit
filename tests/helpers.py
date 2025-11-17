"""Small problems shared by the tests, and the slow references they check against."""

import torch

from culprit.data.blobs import gaussian_blobs
from culprit.losses import IGNORE
from culprit.models import GPT, MLP, GPTConfig
from culprit.task import Task

GPT_CONFIG = GPTConfig(vocab_size=11, block_size=6, n_layer=2, n_head=2, d_model=8, mlp_ratio=2)


def mlp_problem(n=12, seed=0, **task_options):
    """A classifier with a few hundred parameters and `n` examples."""
    x, y = gaussian_blobs(n, dim=5, classes=3, seed=seed)
    return Task(MLP([5, 4, 3], seed=seed), **task_options), x, y


def gpt_problem(n=5, seed=0, **task_options):
    """A language model on random tokens, with some targets masked out."""
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(
        GPT_CONFIG.vocab_size, (n, GPT_CONFIG.block_size + 1), generator=generator
    )
    x, y = tokens[:, :-1], tokens[:, 1:].clone()
    y[0, 3:] = IGNORE  # one example is shorter than the others
    return Task(GPT(GPT_CONFIG, seed=seed), **task_options), x, y


def loop_grads(task, x, y, measure=False):
    """Per-example gradients the obvious way: one backward pass per example."""
    params = {name: value.clone().requires_grad_() for name, value in task.params().items()}
    rows = []
    for i in range(len(x)):
        fn = task.measures if measure else task.losses
        value = fn(params, x[i : i + 1], y[i : i + 1]).sum()
        rows.append(torch.autograd.grad(value, list(params.values())))
    return {name: torch.stack([row[j] for row in rows]) for j, name in enumerate(params)}
