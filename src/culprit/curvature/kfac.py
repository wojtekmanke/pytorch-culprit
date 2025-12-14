"""Kronecker-factored curvature: one small block per layer.

The Fisher information of one linear layer is the expected outer product of
that layer's per-example gradient with itself. The gradient is the outer
product of the layer's input `a` and the backprop `g` at its output, so the
Fisher is an expectation of `(a a^T) kron (g g^T)`. K-FAC replaces the
expectation of the product by the product of the expectations,

    F  ~  E[a a^T] kron E[g g^T]  =  A kron S,

which turns a matrix with `(in * out)^2` entries into two with `in^2` and
`out^2`. The eigenvectors of `A kron S` are the Kronecker products of the
eigenvectors of `A` and of `S`, so the big matrix can be applied, inverted
or damped by rotating into that basis, scaling, and rotating back.

EK-FAC keeps that basis and corrects the scaling. In the basis the K-FAC
eigenvalues are products of the two factors' eigenvalues; EK-FAC replaces
each with the second moment of the per-example gradients along that basis
vector, which is the best diagonal there is for that basis (George et al.,
2018).

The backprops are taken with labels drawn from the model's own predictions,
not the labels in the data. That makes the expectation the Fisher
information, which for cross-entropy is the Gauss-Newton matrix; with the
data's labels it would be the empirical Fisher, a different matrix.

`Recorder` supplies `a` and `g` for a whole batch from one forward and one
backward pass.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from culprit.batching import batches
from culprit.device import cpu_double
from culprit.flat import Layout
from culprit.grads.hooks import Recorder
from culprit.linalg.operator import LinearOperator
from culprit.losses import CrossEntropy
from culprit.task import Params, Task

FisherKind = Literal["sampled", "exact", "empirical"]
SequenceKind = Literal["expand", "reduce"]
Spectral = Callable[[Tensor], Tensor]


def _eigh(matrix: Tensor) -> tuple[Tensor, Tensor]:
    """Eigenvalues clipped at zero, and eigenvectors, of a double-precision matrix.

    The factors are positive semi-definite by construction; what `eigh`
    returns below zero is rounding, and would turn into a negative curvature
    once multiplied by an eigenvalue of the other factor.
    """
    values, vectors = torch.linalg.eigh(matrix)
    return values.clamp_min(0.0), vectors


class Block(ABC):
    """The curvature of the parameters of one layer.

    `names` are the parameters it covers, as the task names them. Statistics
    are accumulated on the CPU in double precision whatever the model's
    device: they are small, and sums over a dataset are where single
    precision loses digits. `finalize` decomposes them and moves the result
    to where the model is.
    """

    names: tuple[str, ...]
    eigenvalues: Tensor  # in the block's eigenbasis, set by finalize

    @abstractmethod
    def update_inputs(self, a: Tensor) -> None:
        """Accumulate what depends only on the layer's input, once per forward pass."""

    @abstractmethod
    def update_backprops(self, a: Tensor, g: Tensor) -> None:
        """Accumulate what depends on a backprop, once per backward pass."""

    @abstractmethod
    def finalize(self, draws: int, like: Tensor) -> None:
        """Turn the statistics into an eigenbasis and eigenvalues, placed like `like`."""

    @abstractmethod
    def rotate(self, tree: Mapping[str, Tensor]) -> Tensor:
        """This block's parameters, from `tree`, expressed in the eigenbasis.

        The values in `tree` may carry leading batch dimensions.
        """

    @abstractmethod
    def unrotate(self, rotated: Tensor) -> Params:
        """Inverse of `rotate`."""

    @abstractmethod
    def per_example(self, a: Tensor, g: Tensor) -> Params:
        """Per-example gradients of this block's parameters, `[batch, *shape]`."""

    def apply(self, tree: Mapping[str, Tensor], fn: Spectral) -> Params:
        """Apply `fn` of this block's curvature to its parameters."""
        return self.unrotate(fn(self.eigenvalues) * self.rotate(tree))

    def start_eigenvalue_fit(self) -> None:
        self._moment_sum = torch.zeros(self.eigenvalues.shape, dtype=torch.float64)

    def update_eigenvalues(self, a: Tensor, g: Tensor) -> None:
        """Accumulate the squared per-example gradients in the eigenbasis."""
        self._moment_sum += cpu_double(self.rotate(self.per_example(a, g)).square().sum(0))

    def finish_eigenvalue_fit(self, draws: int) -> None:
        self.eigenvalues = (self._moment_sum / draws).to(self.eigenvalues)


class LinearBlock(Block):
    """`A kron S` for a linear layer, with the bias folded in.

    The bias is the weight of a constant input of one, so the layer's
    parameters form a single `[out, in + 1]` matrix and `a` gains a trailing
    one.

    When the layer is applied at every position of a sequence, the gradient
    of an example is a sum over positions of outer products, and there are
    two ways to keep a Kronecker structure (Eschenhagen et al., 2023):

    - `"expand"` treats every position as an example of its own. `A` is the
      mean of `a a^T` over all positions; `S` is `g g^T` summed over the
      positions of an example and averaged over examples. It ignores how the
      positions of one example are correlated.
    - `"reduce"` first collapses each example to a single outer product:
      the mean of its inputs with the sum of its backprops. It is exact when
      an example's inputs are the same at every position and coarser the
      more they vary.

    Positions that are masked out of the loss still count towards `A` under
    either choice; the block sees inputs, not targets.
    """

    def __init__(
        self,
        name: str,
        module: nn.Linear,
        covered: frozenset[str],
        sequence: SequenceKind = "expand",
    ) -> None:
        self.weight, self.bias = f"{name}.weight", f"{name}.bias"
        self.with_bias = module.bias is not None and self.bias in covered
        self.names = (self.weight, self.bias) if self.with_bias else (self.weight,)
        self.sequence = sequence
        width = module.in_features + int(self.with_bias)
        self.a_sum = torch.zeros(width, width, dtype=torch.float64)
        self.s_sum = torch.zeros(module.out_features, module.out_features, dtype=torch.float64)
        self.positions = 0

    def _with_one(self, a: Tensor) -> Tensor:
        if not self.with_bias:
            return a
        return torch.cat([a, a.new_ones(*a.shape[:-1], 1)], dim=-1)

    def update_inputs(self, a: Tensor) -> None:
        if self.sequence == "reduce":
            a = a.reshape(a.shape[0], -1, a.shape[-1]).mean(1)
        a = self._with_one(a.reshape(-1, a.shape[-1]))
        self.a_sum += cpu_double(a.T @ a)
        self.positions += a.shape[0]

    def update_backprops(self, a: Tensor, g: Tensor) -> None:
        if self.sequence == "reduce":
            g = g.reshape(g.shape[0], -1, g.shape[-1]).sum(1)
        g = g.reshape(-1, g.shape[-1])
        self.s_sum += cpu_double(g.T @ g)

    def factors(self, draws: int) -> tuple[Tensor, Tensor]:
        """`A` of shape `[in (+1), in (+1)]` and `S` of shape `[out, out]`."""
        return self.a_sum / self.positions, self.s_sum / draws

    def finalize(self, draws: int, like: Tensor) -> None:
        a, s = self.factors(draws)
        a_values, a_vectors = _eigh(a)
        s_values, s_vectors = _eigh(s)
        self.q_a, self.q_s = a_vectors.to(like), s_vectors.to(like)
        self.eigenvalues = torch.outer(s_values, a_values).to(like)

    def rotate(self, tree: Mapping[str, Tensor]) -> Tensor:
        packed = tree[self.weight]
        if self.with_bias:
            packed = torch.cat([packed, tree[self.bias].unsqueeze(-1)], dim=-1)
        return self.q_s.T @ packed @ self.q_a

    def unrotate(self, rotated: Tensor) -> Params:
        packed = self.q_s @ rotated @ self.q_a.T
        if not self.with_bias:
            return {self.weight: packed}
        return {self.weight: packed[..., :-1], self.bias: packed[..., -1]}

    def per_example(self, a: Tensor, g: Tensor) -> Params:
        packed = torch.einsum("b...o,b...i->boi", g, self._with_one(a))
        if not self.with_bias:
            return {self.weight: packed}
        return {self.weight: packed[..., :-1], self.bias: packed[..., -1]}


class EmbeddingBlock(Block):
    """`A kron S` for an embedding, whose `A` is diagonal.

    An embedding is a linear layer applied to a one-hot vector, and the
    outer product of a one-hot vector with itself has a single one on the
    diagonal. `A` is therefore the frequency of each token and is kept as a
    vector, however large the vocabulary. Being diagonal it is its own
    eigenbasis, so only `S` is rotated.

    Positions are always treated as separate examples here. Averaging the
    one-hot inputs of a sequence, as `"reduce"` would, gives a vector of
    token counts whose second moment is no longer diagonal.
    """

    def __init__(self, name: str, module: nn.Embedding) -> None:
        self.weight = f"{name}.weight"
        self.names = (self.weight,)
        self.rows = module.num_embeddings
        self.a_sum = torch.zeros(self.rows, dtype=torch.float64)
        self.s_sum = torch.zeros(module.embedding_dim, module.embedding_dim, dtype=torch.float64)
        self.positions = 0

    def update_inputs(self, a: Tensor) -> None:
        self.a_sum += torch.bincount(a.reshape(-1).cpu(), minlength=self.rows).double()
        self.positions += a.numel()

    def update_backprops(self, a: Tensor, g: Tensor) -> None:
        g = g.reshape(-1, g.shape[-1])
        self.s_sum += cpu_double(g.T @ g)

    def factors(self, draws: int) -> tuple[Tensor, Tensor]:
        """The diagonal of `A`, shape `[vocab]`, and `S` of shape `[width, width]`."""
        return self.a_sum / self.positions, self.s_sum / draws

    def finalize(self, draws: int, like: Tensor) -> None:
        frequency, s = self.factors(draws)
        s_values, s_vectors = _eigh(s)
        self.q_s = s_vectors.to(like)
        self.eigenvalues = torch.outer(frequency, s_values).to(like)

    def rotate(self, tree: Mapping[str, Tensor]) -> Tensor:
        return tree[self.weight] @ self.q_s

    def unrotate(self, rotated: Tensor) -> Params:
        return {self.weight: rotated @ self.q_s.T}

    def per_example(self, a: Tensor, g: Tensor) -> Params:
        batch, width = g.shape[0], g.shape[-1]
        index = a.reshape(batch, -1, 1).expand(-1, -1, width)
        grad = g.new_zeros(batch, self.rows, width)
        grad.scatter_add_(1, index, g.reshape(batch, -1, width))
        return {self.weight: grad}


class LayerNormBlock(Block):
    """The full Fisher of a layer norm's weight and bias.

    The gradient of an elementwise scale is not an outer product, so there
    is nothing to factor. There are only `2 * width` parameters, though, and
    their Fisher is small enough to keep whole and decompose directly.
    """

    def __init__(self, name: str, module: nn.LayerNorm, covered: frozenset[str]) -> None:
        self.module = module
        self.weight, self.bias = f"{name}.weight", f"{name}.bias"
        self.with_bias = module.bias is not None and self.bias in covered
        self.names = (self.weight, self.bias) if self.with_bias else (self.weight,)
        self.shape = tuple(module.normalized_shape)
        self.width = math.prod(self.shape)
        size = self.width * len(self.names)
        self.f_sum = torch.zeros(size, size, dtype=torch.float64)

    def update_inputs(self, a: Tensor) -> None:
        pass

    def _pack(self, tree: Mapping[str, Tensor]) -> Tensor:
        batch_dims = tree[self.weight].dim() - len(self.shape)
        parts = [tree[name].flatten(batch_dims) for name in self.names]
        return torch.cat(parts, dim=-1)

    def per_example(self, a: Tensor, g: Tensor) -> Params:
        normalised = F.layer_norm(a, self.shape, None, None, self.module.eps)
        shape = (g.shape[0], -1, *self.shape)
        grads = {self.weight: (g * normalised).reshape(shape).sum(1)}
        if self.with_bias:
            grads[self.bias] = g.reshape(shape).sum(1)
        return grads

    def update_backprops(self, a: Tensor, g: Tensor) -> None:
        grads = self._pack(self.per_example(a, g))
        self.f_sum += cpu_double(grads.T @ grads)

    def fisher(self, draws: int) -> Tensor:
        return self.f_sum / draws

    def finalize(self, draws: int, like: Tensor) -> None:
        values, vectors = _eigh(self.fisher(draws))
        self.q, self.eigenvalues = vectors.to(like), values.to(like)

    def rotate(self, tree: Mapping[str, Tensor]) -> Tensor:
        return self._pack(tree) @ self.q

    def unrotate(self, rotated: Tensor) -> Params:
        packed = rotated @ self.q.T
        parts = packed.split(self.width, dim=-1)
        return {
            name: part.reshape(*packed.shape[:-1], *self.shape)
            for name, part in zip(self.names, parts, strict=True)
        }


def _blocks(task: Task, params: Mapping[str, Tensor], sequence: SequenceKind) -> dict[str, Block]:
    """One block per layer that owns any of the task's parameters, keyed by layer name."""
    covered = frozenset(params)
    blocks: dict[str, Block] = {}
    for name, module in task.model.named_modules():
        if f"{name}.weight" not in covered:
            continue
        if isinstance(module, nn.Linear):
            blocks[name] = LinearBlock(name, module, covered, sequence)
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


class _Sweep:
    """The loop every statistic shares: forward once, backprop each pseudo-loss."""

    def __init__(
        self,
        task: Task,
        params: Mapping[str, Tensor],
        *,
        fisher: FisherKind,
        samples: int,
        seed: int,
        batch_size: int | None,
    ) -> None:
        self.task, self.params = task, params
        self.fisher, self.samples, self.seed, self.batch_size = fisher, samples, seed, batch_size

    def run(
        self,
        blocks: Mapping[str, Block],
        inputs: Tensor,
        targets: Tensor,
        on_inputs: Callable[[Block, Tensor], None],
        on_backprops: Callable[[Block, Tensor, Tensor], None],
    ) -> int:
        """Call the two callbacks for every block and batch; return the number of draws.

        A run with the same seed draws the same labels, which is what lets a
        second sweep fit eigenvalues to the very gradients the first sweep
        built the basis from.
        """
        generator = torch.Generator().manual_seed(self.seed)
        draws = 0
        for x, y in batches(inputs, targets, self.batch_size):
            with Recorder(self.task.model) as recorder:
                outputs = self.task.outputs(self.params, x)
                seen = {name: recorder.records[name].input for name in blocks}
                for name, block in blocks.items():
                    on_inputs(block, seen[name])
                losses = list(
                    _pseudo_losses(self.task, outputs, y, self.fisher, generator, self.samples)
                )
                for index, loss in enumerate(losses):
                    backprops = recorder.backprops(loss, retain_graph=index < len(losses) - 1)
                    for name, block in blocks.items():
                        on_backprops(block, seen[name], backprops[name])
            draws += len(x) * (self.samples if self.fisher == "sampled" else 1)
        return draws


class KFACFactors:
    """The accumulated statistics of every layer, before anything is decomposed."""

    def __init__(self, blocks: dict[str, Block], draws: int) -> None:
        self.blocks, self.draws = blocks, draws


class FactoredCurvature(LinearOperator):
    """A block-diagonal curvature approximation that knows its own eigenbasis.

    Because every block is stored as an eigenbasis and eigenvalues, any
    function of the matrix costs the same as a product: `inverse` gives the
    damped inverse as another operator, with no solve.
    """

    def __init__(
        self,
        layout: Layout,
        blocks: Mapping[str, Block],
        like: Tensor,
        fn: Spectral | None = None,
    ) -> None:
        self.layout, self.blocks = layout, dict(blocks)
        self.size, self.dtype, self.device = layout.size, like.dtype, like.device
        self._placement = like
        self._fn: Spectral = fn if fn is not None else (lambda eigenvalues: eigenvalues)

    def _apply(self, tree: Mapping[str, Tensor]) -> Params:
        out: Params = {}
        for block in self.blocks.values():
            out.update(block.apply(tree, self._fn))
        return out

    def matvec(self, vector: Tensor) -> Tensor:
        return self.layout.flatten(self._apply(self.layout.unflatten(vector)))

    def matmat(self, matrix: Tensor) -> Tensor:
        # Columns become a leading batch dimension, which every block's
        # rotation broadcasts over.
        return self.layout.flatten(self._apply(self.layout.unflatten(matrix.T))).T

    def inverse(self, damping: float) -> FactoredCurvature:
        """`(self + damping * I)^-1`, in the same eigenbasis."""
        return FactoredCurvature(
            self.layout,
            self.blocks,
            self._placement,
            lambda eigenvalues: 1.0 / (eigenvalues + damping),
        )


def collect_factors(
    task: Task,
    params: Mapping[str, Tensor],
    inputs: Tensor,
    targets: Tensor,
    *,
    fisher: FisherKind = "sampled",
    sequence: SequenceKind = "expand",
    samples: int = 1,
    seed: int = 0,
    batch_size: int | None = None,
) -> KFACFactors:
    """One pass over the data that gathers `A` and `S` for every layer.

    `fisher` chooses the labels the backprops are taken with: `"sampled"`
    draws `samples` labellings per example from the model, `"exact"`
    enumerates the classes and has no sampling noise but needs one backward
    pass per class, and `"empirical"` uses the labels in the data. `sequence`
    chooses how a linear layer shared across positions is factored; see
    `LinearBlock`.
    """
    blocks = _blocks(task, params, sequence)
    sweep = _Sweep(task, params, fisher=fisher, samples=samples, seed=seed, batch_size=batch_size)
    draws = sweep.run(
        blocks,
        inputs,
        targets,
        lambda b, a: b.update_inputs(a),
        lambda b, a, g: b.update_backprops(a, g),
    )
    return KFACFactors(blocks, draws)


def kfac(
    task: Task,
    params: Mapping[str, Tensor],
    inputs: Tensor,
    targets: Tensor,
    *,
    fisher: FisherKind = "sampled",
    sequence: SequenceKind = "expand",
    samples: int = 1,
    seed: int = 0,
    batch_size: int | None = None,
    eigenvalue_correction: bool = False,
) -> FactoredCurvature:
    """The K-FAC approximation to the Fisher of the mean loss, as an operator.

    With `eigenvalue_correction` a second pass over the data refits the
    eigenvalues to the per-example gradients, which makes it EK-FAC; `ekfac`
    is a shorthand for that.
    """
    factors = collect_factors(
        task,
        params,
        inputs,
        targets,
        fisher=fisher,
        sequence=sequence,
        samples=samples,
        seed=seed,
        batch_size=batch_size,
    )
    like = next(iter(params.values()))
    for block in factors.blocks.values():
        block.finalize(factors.draws, like)
    if eigenvalue_correction:
        sweep = _Sweep(
            task, params, fisher=fisher, samples=samples, seed=seed, batch_size=batch_size
        )
        for block in factors.blocks.values():
            block.start_eigenvalue_fit()
        draws = sweep.run(
            factors.blocks,
            inputs,
            targets,
            lambda b, a: None,
            lambda b, a, g: b.update_eigenvalues(a, g),
        )
        for block in factors.blocks.values():
            block.finish_eigenvalue_fit(draws)
    return FactoredCurvature(Layout.of(params), factors.blocks, like)


def ekfac(
    task: Task,
    params: Mapping[str, Tensor],
    inputs: Tensor,
    targets: Tensor,
    *,
    fisher: FisherKind = "sampled",
    sequence: SequenceKind = "expand",
    samples: int = 1,
    seed: int = 0,
    batch_size: int | None = None,
) -> FactoredCurvature:
    """K-FAC's eigenbasis with eigenvalues refitted to the per-example gradients."""
    return kfac(
        task,
        params,
        inputs,
        targets,
        fisher=fisher,
        sequence=sequence,
        samples=samples,
        seed=seed,
        batch_size=batch_size,
        eigenvalue_correction=True,
    )
