"""Kronecker-factored curvature: one small block per layer.

The Fisher information of one linear layer is the expected outer product of
that layer's per-example gradient with itself. The gradient is the outer
product of the layer's input `a` and the backprop `g` at its output, so the
Fisher is an expectation of `(a a^T) kron (g g^T)`. K-FAC replaces the
expectation of the product by the product of the expectations,

    F  ~  E[a a^T] kron E[g g^T]  =  A kron S,

which turns a matrix with `(in * out)^2` entries into two with `in^2` and
`out^2`, and turns its inverse into the inverses of those two.

The backprops are taken with labels drawn from the model's own predictions,
not the labels in the data. That makes the expectation the Fisher
information, which for cross-entropy is the Gauss-Newton matrix; with the
data's labels it would be the empirical Fisher, a different matrix.

This module collects the factors. `Recorder` supplies `a` and `g` for a
whole batch from one forward and one backward pass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from culprit.batching import batches
from culprit.device import cpu_double
from culprit.grads.hooks import Recorder
from culprit.losses import CrossEntropy
from culprit.task import Task

FisherKind = Literal["sampled", "exact", "empirical"]


class Block(ABC):
    """The curvature of the parameters of one layer.

    `names` are the parameters it covers, as the task names them. Statistics
    are accumulated on the CPU in double precision whatever the model's
    device: they are small, and sums over a dataset are where single
    precision loses digits.
    """

    names: tuple[str, ...]

    @abstractmethod
    def update_inputs(self, a: Tensor) -> None:
        """Accumulate what depends only on the layer's input, once per forward pass."""

    @abstractmethod
    def update_backprops(self, a: Tensor, g: Tensor) -> None:
        """Accumulate what depends on a backprop, once per backward pass."""


class LinearBlock(Block):
    """`A kron S` for a linear layer, with the bias folded in.

    The bias is the weight of a constant input of one, so the layer's
    parameters form a single `[out, in + 1]` matrix and `a` gains a trailing
    one. A layer applied at every position of a sequence treats each
    position as a separate input: `A` is averaged over positions and `S`
    is summed over them and averaged over examples.
    """

    def __init__(self, name: str, module: nn.Linear, covered: frozenset[str]) -> None:
        self.weight, self.bias = f"{name}.weight", f"{name}.bias"
        self.with_bias = module.bias is not None and self.bias in covered
        self.names = (self.weight, self.bias) if self.with_bias else (self.weight,)
        width = module.in_features + int(self.with_bias)
        self.a_sum = torch.zeros(width, width, dtype=torch.float64)
        self.s_sum = torch.zeros(module.out_features, module.out_features, dtype=torch.float64)
        self.positions = 0

    def _inputs(self, a: Tensor) -> Tensor:
        a = a.reshape(-1, a.shape[-1])
        if self.with_bias:
            a = torch.cat([a, a.new_ones(a.shape[0], 1)], dim=1)
        return a

    def update_inputs(self, a: Tensor) -> None:
        a = self._inputs(a)
        self.a_sum += cpu_double(a.T @ a)
        self.positions += a.shape[0]

    def update_backprops(self, a: Tensor, g: Tensor) -> None:
        g = g.reshape(-1, g.shape[-1])
        self.s_sum += cpu_double(g.T @ g)

    def factors(self, draws: int) -> tuple[Tensor, Tensor]:
        """`A` of shape `[in (+1), in (+1)]` and `S` of shape `[out, out]`."""
        return self.a_sum / self.positions, self.s_sum / draws


class EmbeddingBlock(Block):
    """`A kron S` for an embedding, whose `A` is diagonal.

    An embedding is a linear layer applied to a one-hot vector, and the
    outer product of a one-hot vector with itself has a single one on the
    diagonal. `A` is therefore the frequency of each token and is kept as a
    vector, however large the vocabulary.
    """

    def __init__(self, name: str, module: nn.Embedding) -> None:
        self.names = (f"{name}.weight",)
        self.a_sum = torch.zeros(module.num_embeddings, dtype=torch.float64)
        self.s_sum = torch.zeros(module.embedding_dim, module.embedding_dim, dtype=torch.float64)
        self.positions = 0

    def update_inputs(self, a: Tensor) -> None:
        counts = torch.bincount(a.reshape(-1).cpu(), minlength=len(self.a_sum))
        self.a_sum += counts.double()
        self.positions += a.numel()

    def update_backprops(self, a: Tensor, g: Tensor) -> None:
        g = g.reshape(-1, g.shape[-1])
        self.s_sum += cpu_double(g.T @ g)

    def factors(self, draws: int) -> tuple[Tensor, Tensor]:
        """The diagonal of `A`, shape `[vocab]`, and `S` of shape `[width, width]`."""
        return self.a_sum / self.positions, self.s_sum / draws


class LayerNormBlock(Block):
    """The full Fisher of a layer norm's weight and bias.

    The gradient of an elementwise scale is not an outer product, so there
    is nothing to factor. There are only `2 * width` parameters, though, and
    their Fisher is small enough to keep whole.
    """

    def __init__(self, name: str, module: nn.LayerNorm, covered: frozenset[str]) -> None:
        self.module = module
        self.weight, self.bias = f"{name}.weight", f"{name}.bias"
        self.with_bias = module.bias is not None and self.bias in covered
        self.names = (self.weight, self.bias) if self.with_bias else (self.weight,)
        size = sum(module.normalized_shape) * len(self.names)
        self.f_sum = torch.zeros(size, size, dtype=torch.float64)

    def update_inputs(self, a: Tensor) -> None:
        pass

    def per_example(self, a: Tensor, g: Tensor) -> Tensor:
        """Per-example gradients of weight then bias, `[batch, size]`."""
        normalised = F.layer_norm(a, self.module.normalized_shape, None, None, self.module.eps)
        shape = (g.shape[0], -1, *self.module.normalized_shape)
        parts = [(g * normalised).reshape(shape).sum(1).flatten(1)]
        if self.with_bias:
            parts.append(g.reshape(shape).sum(1).flatten(1))
        return torch.cat(parts, dim=1)

    def update_backprops(self, a: Tensor, g: Tensor) -> None:
        grads = self.per_example(a, g)
        self.f_sum += cpu_double(grads.T @ grads)

    def fisher(self, draws: int) -> Tensor:
        return self.f_sum / draws


def _blocks(task: Task, params: Mapping[str, Tensor]) -> dict[str, Block]:
    """One block per layer that owns any of the task's parameters, keyed by layer name."""
    covered = frozenset(params)
    blocks: dict[str, Block] = {}
    for name, module in task.model.named_modules():
        if f"{name}.weight" not in covered:
            continue
        if isinstance(module, nn.Linear):
            blocks[name] = LinearBlock(name, module, covered)
        elif isinstance(module, nn.Embedding):
            blocks[name] = EmbeddingBlock(name, module)
        elif isinstance(module, nn.LayerNorm):
            blocks[name] = LayerNormBlock(name, module, covered)
    owned = {name for block in blocks.values() for name in block.names}
    missing = [name for name in params if name not in owned]
    if missing:
        raise ValueError(f"no Kronecker block for: {', '.join(missing)}")
    return blocks


def _pseudo_losses(
    task: Task,
    outputs: Tensor,
    targets: Tensor,
    fisher: FisherKind,
    generator: torch.Generator,
    samples: int,
) -> Iterator[Tensor]:
    """Scalars whose backprops, squared and added up, give the Fisher of this batch."""
    if fisher == "empirical":
        yield task.loss(outputs, targets).sum()
        return
    if not isinstance(task.loss, CrossEntropy):
        raise TypeError("the Fisher needs a loss whose labels can be drawn from the model")
    if fisher == "sampled":
        for _ in range(samples):
            yield task.loss(outputs, task.loss.sample(outputs, targets, generator)).sum()
        return
    # Exact: for each class, the loss of that class weighted by the square
    # root of its probability. The squared backprops then add up to the
    # expectation over classes, with no sampling noise.
    if outputs.dim() != 2:
        raise ValueError(
            "the exact Fisher is only available for one prediction per example; "
            "a sequence has too many joint labellings to enumerate"
        )
    weights = outputs.detach().softmax(-1).sqrt()
    for label in range(outputs.shape[1]):
        yield (weights[:, label] * task.loss(outputs, torch.full_like(targets, label))).sum()


class KFACFactors:
    """The accumulated statistics of every layer, before anything is inverted."""

    def __init__(self, blocks: dict[str, Block], draws: int) -> None:
        self.blocks, self.draws = blocks, draws


def collect_factors(
    task: Task,
    params: Mapping[str, Tensor],
    inputs: Tensor,
    targets: Tensor,
    *,
    fisher: FisherKind = "sampled",
    samples: int = 1,
    seed: int = 0,
    batch_size: int | None = None,
) -> KFACFactors:
    """One pass over the data that gathers `A` and `S` for every layer.

    `fisher` chooses the labels the backprops are taken with: `"sampled"`
    draws `samples` labellings per example from the model, `"exact"`
    enumerates the classes and has no sampling noise but needs one backward
    pass per class, and `"empirical"` uses the labels in the data.
    """
    blocks = _blocks(task, params)
    generator = torch.Generator().manual_seed(seed)
    draws = 0
    for x, y in batches(inputs, targets, batch_size):
        with Recorder(task.model) as recorder:
            outputs = task.outputs(params, x)
            records = {name: recorder.records[name] for name in blocks}
            for name, block in blocks.items():
                block.update_inputs(records[name].input)
            losses = list(_pseudo_losses(task, outputs, y, fisher, generator, samples))
            for index, loss in enumerate(losses):
                backprops = recorder.backprops(loss, retain_graph=index < len(losses) - 1)
                for name, block in blocks.items():
                    block.update_backprops(records[name].input, backprops[name])
        draws += len(x) * (samples if fisher == "sampled" else 1)
    return KFACFactors(blocks, draws)
