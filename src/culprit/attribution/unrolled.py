"""Differentiate a measurement through the whole training run.

Influence functions ask how the *minimiser* of the training loss moves when
an example is reweighted. A network is not at a minimiser; it is wherever
its optimiser left it. Unrolling asks the question about that instead.
Training is a deterministic function of the examples' weights:

    theta_0 -> step -> theta_1 -> step -> ... -> theta_T -> measurement,

where each step computes a weighted mini-batch gradient and hands it to the
optimiser. The score of example `i` for query `q` is the derivative of the
measurement of `q` at `theta_T` with respect to the weight of example `i`,
taken at weights of one. Nothing is assumed about convergence or convexity,
and the optimiser's own state, its momentum or its running moments, is
differentiated through like everything else.

The derivative is taken in reverse. An adjoint of the parameters and of the
optimiser state starts at the gradient of the measurement and is pulled
back one step at a time; at each step the same pullback gives the
derivative with respect to that step's example weights.

Reverse mode needs the parameters at every step, in reverse order. Storing
all of them costs memory in proportion to the length of training. They are
kept only every `segment` steps; going backwards, each segment is replayed
from its stored start, which training being deterministic makes exact. That
is `steps / segment + segment` stored states for one extra forward pass.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

import torch
from torch import Tensor
from torch.func import grad, grad_and_value, vjp, vmap

from culprit.attribution.base import Attributor
from culprit.device import cpu_double
from culprit.ensemble.optim import FunctionalOptimizer, optimizer_for
from culprit.grads import per_example_grads
from culprit.task import Params, Task
from culprit.train import TrainConfig, batch_schedule, learning_rate


def weighted_step(
    task: Task,
    optimizer: FunctionalOptimizer,
    params: Mapping[str, Tensor],
    state: Any,
    weights: Tensor,
    inputs: Tensor,
    targets: Tensor,
    lr: float,
) -> tuple[Params, Any, Tensor]:
    """One training step on a batch whose examples carry `weights`.

    With weights of one this is the step `culprit.train.train` takes.
    Returns the new parameters, the new optimiser state and the loss.
    """

    def loss(at: Params) -> Tensor:
        return (weights * task.losses(at, inputs, targets)).mean()

    grads, value = grad_and_value(loss)(dict(params))
    new_params, new_state = optimizer.step(params, grads, state, lr)
    return new_params, new_state, value


@dataclass(frozen=True)
class FunctionalRun:
    final: Params
    losses: Tensor  # [steps]
    schedule: Tensor  # [steps, batch], the examples each step trained on
    snapshots: dict[int, tuple[Params, Any]]  # parameters and state before the given step


def functional_train(
    task: Task,
    inputs: Tensor,
    targets: Tensor,
    config: TrainConfig,
    initial: Mapping[str, Tensor],
    *,
    weights: Tensor | None = None,
    pool: Tensor | None = None,
    keep_every: int | None = None,
) -> FunctionalRun:
    """`culprit.train.train` as a function: nothing is modified, everything is returned.

    Starts from `initial` and visits the examples in the order `config.seed`
    gives, so with `weights` left at one it arrives where `train` does, up
    to rounding. `weights` holds one weight per training example.
    `keep_every` stores the parameters and optimiser state before every
    that-many-th step, for a later pass in reverse.
    """
    optimizer = optimizer_for(config)
    schedule = batch_schedule(
        len(inputs) if pool is None else pool, config.batch_size, config.epochs, config.seed
    ).to(inputs.device)
    if weights is None:
        weights = torch.ones(len(inputs), dtype=next(iter(initial.values())).dtype)
    weights = weights.to(inputs.device)
    params: Params = dict(initial)
    state = optimizer.init(params)
    snapshots: dict[int, tuple[Params, Any]] = {}
    losses = torch.empty(len(schedule))
    for step, batch in enumerate(schedule):
        if keep_every and step % keep_every == 0:
            snapshots[step] = (params, state)
        lr = learning_rate(step, len(schedule), config)
        params, state, loss = weighted_step(
            task, optimizer, params, state, weights[batch], inputs[batch], targets[batch], lr
        )
        losses[step] = loss.detach().cpu()
    return FunctionalRun(final=params, losses=losses, schedule=schedule, snapshots=snapshots)


def weighted_runs(
    task: Task,
    inputs: Tensor,
    targets: Tensor,
    config: TrainConfig,
    initial: Mapping[str, Tensor],
    weights: Tensor,
) -> Params:
    """The final parameters of many runs that differ only in their example weights.

    `weights` is `[runs, examples]`. Every run starts from `initial` and
    visits the examples in the same order, so the runs differ in nothing but
    the weights: a weight of zero removes an example without changing which
    batch it would have been in. That is retraining matched to one run, the
    counterfactual a derivative taken through that run is a statement about.
    The runs are trained together; the result has a leading run dimension.
    """
    optimizer = optimizer_for(config)
    schedule = batch_schedule(len(inputs), config.batch_size, config.epochs, config.seed)
    schedule = schedule.to(inputs.device)
    weights = weights.to(inputs.device)
    params: Params = {
        name: value.expand(len(weights), *value.shape).clone() for name, value in initial.items()
    }
    state = optimizer.init(params)

    def loss(at: Params, batch_weights: Tensor, x: Tensor, y: Tensor) -> Tensor:
        return (batch_weights * task.losses(at, x, y)).mean()

    gradients = vmap(grad(loss), in_dims=(0, 0, None, None))
    for step, batch in enumerate(schedule):
        grads = gradients(params, weights[:, batch], inputs[batch], targets[batch])
        params, state = optimizer.step(
            params, grads, state, learning_rate(step, len(schedule), config)
        )
    return params


class Unrolled(Attributor):
    """The derivative of each query's measurement with respect to each example's weight.

    Unlike the other methods this one explains a training run, not a set of
    parameters, so it is given the run: the configuration and the parameters
    it started from. `fit` repeats the run, which reproduces the model being
    explained, and `score` differentiates through it.

    One reverse pass serves every query at once: the adjoints of all the
    queries are stacked and each step's pullback is applied to the stack
    with `vmap`. `segment` is the number of steps between stored states.
    """

    def __init__(
        self,
        task: Task,
        config: TrainConfig,
        initial: Mapping[str, Tensor],
        *,
        segment: int = 50,
        pool: Tensor | None = None,
    ) -> None:
        super().__init__(task, initial)
        self.config, self.initial, self.segment, self.pool = config, dict(initial), segment, pool
        self.optimizer = optimizer_for(config)
        self._train: tuple[Tensor, Tensor] | None = None
        self.run: FunctionalRun | None = None

    def fit(self, inputs: Tensor, targets: Tensor) -> Self:
        self._train = (inputs, targets)
        self.run = functional_train(
            self.task,
            inputs,
            targets,
            self.config,
            self.initial,
            pool=self.pool,
            keep_every=self.segment,
        )
        self.params = self.run.final
        return self

    def _replay(self, start: int, stop: int, ones: Tensor) -> list[tuple[Params, Any]]:
        """The parameters and state before each of the steps `start` to `stop - 1`."""
        assert self.run is not None and self._train is not None
        inputs, targets = self._train
        params, state = self.run.snapshots[start]
        states = []
        for step in range(start, stop):
            states.append((params, state))
            batch = self.run.schedule[step]
            lr = learning_rate(step, len(self.run.schedule), self.config)
            params, state, _ = weighted_step(
                self.task, self.optimizer, params, state, ones, inputs[batch], targets[batch], lr
            )
        return states

    def score(self, inputs: Tensor, targets: Tensor) -> Tensor:
        if self.run is None or self._train is None:
            raise RuntimeError("call fit with the training examples before score")
        train_inputs, train_targets = self._train
        final = self.run.final
        first = next(iter(final.values()))
        total = len(self.run.schedule)

        # The adjoints at the end of training: the gradient of each query's
        # measurement for the parameters, and zero for the optimiser state,
        # which the measurement does not depend on.
        adjoint: Params = per_example_grads(self.task.example_measure, final, inputs, targets)
        _, last_state = self.run.snapshots[max(self.run.snapshots)]
        state_adjoint = tuple(
            {
                name: torch.zeros(len(inputs), *value.shape, dtype=first.dtype, device=first.device)
                for name, value in tree.items()
            }
            for tree in last_state.tensors()
        )
        scores = torch.zeros(len(inputs), len(train_inputs), dtype=first.dtype, device=first.device)
        ones = torch.ones(self.config.batch_size, dtype=first.dtype, device=first.device)

        for start in sorted(self.run.snapshots, reverse=True):
            stop = min(start + self.segment, total)
            states = self._replay(start, stop, ones)
            for step in range(stop - 1, start - 1, -1):
                params, state = states[step - start]
                batch = self.run.schedule[step]
                lr = learning_rate(step, total, self.config)

                def step_fn(
                    at: Params,
                    state_tensors: tuple[Params, ...],
                    weights: Tensor,
                    state: Any = state,
                    batch: Tensor = batch,
                    lr: float = lr,
                ) -> tuple[Params, tuple[Params, ...]]:
                    new_params, new_state, _ = weighted_step(
                        self.task,
                        self.optimizer,
                        at,
                        state.with_tensors(state_tensors),
                        weights,
                        train_inputs[batch],
                        train_targets[batch],
                        lr,
                    )
                    return new_params, new_state.tensors()

                pullback = vjp(step_fn, params, state.tensors(), ones)[1]
                adjoint, state_adjoint, weight_adjoint = vmap(pullback)((adjoint, state_adjoint))
                # A batch is cut from a permutation, so its indices are distinct.
                scores[:, batch] += weight_adjoint
        return cpu_double(scores)
