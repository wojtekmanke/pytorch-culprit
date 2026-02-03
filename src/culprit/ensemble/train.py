"""Training many copies of a model at once.

Checking attribution against retraining means retraining hundreds of times.
For models this small a GPU spends most of a training step on overhead, so
hundreds of separate runs waste it. Here the copies are one computation:
their parameters are stacked along a new first dimension, and `vmap` turns
the function that computes one model's gradient into one that computes all
of them. Each model still sees its own examples in its own order and starts
from its own initialisation; nothing is shared but the wall clock.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.func import grad_and_value, stack_module_state, vmap

from culprit.ensemble.optim import optimizer_for
from culprit.task import Params, Task
from culprit.train import TrainConfig, batch_schedule, learning_rate


@dataclass(frozen=True)
class EnsembleRun:
    params: Params  # final parameters, each `[models, *shape]`
    losses: Tensor  # `[steps, models]`, mean loss of each model's batch before its update


def initial_parameters(task: Task, seeds: Sequence[int]) -> Params:
    """The task's parameters as `reset_parameters(seed)` leaves them, one row per seed.

    The task's own model is left as it was.
    """
    models = []
    for seed in seeds:
        model = copy.deepcopy(task.model)
        model.reset_parameters(seed)  # type: ignore[operator]
        models.append(model)
    stacked, _ = stack_module_state(models)
    return {name: stacked[name].detach() for name in task.params()}


def train_ensemble(
    task: Task,
    inputs: Tensor,
    targets: Tensor,
    config: TrainConfig,
    *,
    pools: Tensor,
    init_seeds: Sequence[int],
    order_seeds: Sequence[int],
) -> EnsembleRun:
    """Train one model per row of `pools`, all at once.

    Row `m` of `pools` holds the indices of the examples model `m` may train
    on; the rows have the same length, so every model takes the same number
    of steps. Model `m` is initialised from `init_seeds[m]` and visits its
    pool in the order `order_seeds[m]` gives. The result for each model is
    what `culprit.train.train` gives for the same pool and seeds, up to
    rounding; `config.seed` is not used.
    """
    count = len(pools)
    if not (len(init_seeds) == len(order_seeds) == count):
        raise ValueError("pools, init_seeds and order_seeds must have one entry per model")
    params = initial_parameters(task, init_seeds)
    schedule = torch.stack(
        [
            batch_schedule(pool, config.batch_size, config.epochs, seed)
            for pool, seed in zip(pools, order_seeds, strict=True)
        ],
        dim=1,
    ).to(inputs.device)  # [steps, models, batch]

    def mean_loss(model_params: Params, x: Tensor, y: Tensor) -> Tensor:
        return task.losses(model_params, x, y).mean()

    # One model's gradient function, batched over models: every argument
    # carries a leading model dimension, including the data.
    gradients = vmap(grad_and_value(mean_loss))

    optimizer = optimizer_for(config)
    state = optimizer.init(params)
    total = len(schedule)
    losses = torch.empty(total, count)
    for step, batch in enumerate(schedule):
        grads, loss = gradients(params, inputs[batch], targets[batch])
        params, state = optimizer.step(params, grads, state, learning_rate(step, total, config))
        # Moved before it is stored: writing a GPU tensor straight into a
        # double-precision CPU tensor silently writes nothing (see pitfalls.md).
        losses[step] = loss.detach().cpu()
    return EnsembleRun(params=params, losses=losses)


def ensemble_measures(task: Task, params: Params, inputs: Tensor, targets: Tensor) -> Tensor:
    """The measurement of every example under every model, `[models, examples]`."""
    out: Tensor = vmap(task.measures, in_dims=(0, None, None))(params, inputs, targets)
    return out
