"""Influence functions whose inverse curvature is applied without a dense solve."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import Literal, Self

import torch
from torch import Tensor

from culprit.attribution.base import Attributor
from culprit.batching import batches
from culprit.curvature.kfac import FactoredCurvature, FisherKind, SequenceKind, kfac
from culprit.device import cpu_double
from culprit.grads import gradient_matrix
from culprit.linalg.operator import CurvatureOperator, LinearOperator
from culprit.linalg.solvers import SolveInfo, conjugate_gradient, top_eigenpairs
from culprit.task import Task


class InfluenceFunction(Attributor):
    """Everything of the form `-1/n * grad m(q)^T C^-1 grad loss(i)`.

    Subclasses differ only in how they apply the inverse curvature. It is
    applied to the gradients of the queries, of which there are few, and the
    result is then matched against the gradients of the training examples a
    batch at a time, so no matrix with a row per training example and a
    column per parameter is ever held.
    """

    def __init__(
        self,
        task: Task,
        params: Mapping[str, Tensor] | None = None,
        *,
        batch_size: int | None = None,
    ) -> None:
        super().__init__(task, params)
        self.batch_size = batch_size
        self._train: tuple[Tensor, Tensor] | None = None

    def fit(self, inputs: Tensor, targets: Tensor) -> Self:
        self._train = (inputs, targets)
        self._prepare(inputs, targets)
        return self

    @abstractmethod
    def _prepare(self, inputs: Tensor, targets: Tensor) -> None:
        """Build whatever `_solve` needs from the training examples."""

    @abstractmethod
    def _solve(self, rhs: Tensor) -> Tensor:
        """An approximation of `C^-1 @ rhs` for the columns of `rhs`, `[size, k]`."""

    def score(self, inputs: Tensor, targets: Tensor) -> Tensor:
        if self._train is None:
            raise RuntimeError("call fit with the training examples before score")
        queries = gradient_matrix(
            self.task, self.params, inputs, targets, measure=True, batch_size=self.batch_size
        )
        solved = self._solve(queries.T)
        rows = [
            gradient_matrix(self.task, self.params, x, y) @ solved
            for x, y in batches(*self._train, self.batch_size)
        ]
        return -cpu_double(torch.cat(rows).T) / len(self._train[0])


class EKFACInfluence(InfluenceFunction):
    """Influence functions with an EK-FAC approximation to the curvature.

    The inverse is a rotation into the Kronecker eigenbasis of each layer, a
    division, and a rotation back, so after two sweeps over the training
    data to fit the approximation, a query costs about as much as a
    gradient. This is the method of Grosse et al. (2023).

    Set `eigenvalue_correction` to false for plain K-FAC.
    """

    def __init__(
        self,
        task: Task,
        params: Mapping[str, Tensor] | None = None,
        *,
        damping: float = 1e-3,
        eigenvalue_correction: bool = True,
        fisher: FisherKind = "sampled",
        sequence: SequenceKind = "expand",
        samples: int = 1,
        seed: int = 0,
        batch_size: int | None = None,
    ) -> None:
        super().__init__(task, params, batch_size=batch_size)
        self.damping = damping
        self._options = {
            "eigenvalue_correction": eigenvalue_correction,
            "fisher": fisher,
            "sequence": sequence,
            "samples": samples,
            "seed": seed,
        }
        self.curvature: FactoredCurvature | None = None

    def _prepare(self, inputs: Tensor, targets: Tensor) -> None:
        self.curvature = kfac(
            self.task,
            self.params,
            inputs,
            targets,
            batch_size=self.batch_size,
            **self._options,  # type: ignore[arg-type]
        )
        self._inverse = self.curvature.inverse(self.damping)

    def _solve(self, rhs: Tensor) -> Tensor:
        return self._inverse @ rhs


class CGInfluence(InfluenceFunction):
    """Influence functions with the inverse applied by conjugate gradient.

    Each iteration is one pass over the training data, shared by all the
    queries of a call. Run to convergence this is exact influence for
    models too large to form the curvature matrix of, and how far it is
    from convergence is reported in `info` after every `score`.

    `precondition` uses the EK-FAC inverse as a preconditioner. It is off by
    default because it is not a safe default: on a small MLP with badly
    scaled inputs and little damping it cut the iterations by a third, and
    on a small transformer whose curvature is far from full rank it
    multiplied them by up to five. `docs/notes/negative-results.md` has the
    measurements.
    """

    def __init__(
        self,
        task: Task,
        params: Mapping[str, Tensor] | None = None,
        *,
        damping: float = 1e-3,
        curvature: Literal["ggn", "hessian"] = "ggn",
        tol: float | None = None,
        max_iterations: int = 100,
        precondition: bool = False,
        batch_size: int | None = None,
        chunk_size: int | None = None,
    ) -> None:
        super().__init__(task, params, batch_size=batch_size)
        self.damping, self.curvature, self.chunk_size = damping, curvature, chunk_size
        self.tol, self.max_iterations, self.precondition = tol, max_iterations, precondition
        self.info: SolveInfo | None = None

    def _prepare(self, inputs: Tensor, targets: Tensor) -> None:
        self._operator = CurvatureOperator(
            self.task,
            self.params,
            inputs,
            targets,
            kind=self.curvature,
            batch_size=self.batch_size,
            chunk_size=self.chunk_size,
        ).damped(self.damping)
        self._preconditioner: LinearOperator | None = None
        if self.precondition:
            factored = kfac(
                self.task,
                self.params,
                inputs,
                targets,
                eigenvalue_correction=True,
                batch_size=self.batch_size,
            )
            self._preconditioner = factored.inverse(self.damping)

    def _solve(self, rhs: Tensor) -> Tensor:
        solution, self.info = conjugate_gradient(
            self._operator,
            rhs,
            tol=self.tol,
            max_iterations=self.max_iterations,
            preconditioner=self._preconditioner,
        )
        return solution


class LiSSAInfluence(InfluenceFunction):
    """Influence functions with the inverse as a stochastic Neumann series.

    For a matrix `M` whose eigenvalues lie in `(0, scale)`,

        M^-1 v  =  1/scale * sum over j of (I - M / scale)^j v,

    and the partial sums obey `h <- v + h - M h / scale`. LiSSA (Agarwal et
    al., 2017; used for influence by Koh and Liang, 2017) runs that
    recursion with `M` estimated from a fresh mini-batch at every step, and
    averages `repeats` independent runs.

    It needs no more than one mini-batch curvature product per step, which
    is its appeal, and converges at a rate set by `damping / scale`, which
    is its problem: a small damping means many steps. `scale` defaults to
    slightly more than the largest eigenvalue, found by a few steps of
    Lanczos.
    """

    def __init__(
        self,
        task: Task,
        params: Mapping[str, Tensor] | None = None,
        *,
        damping: float = 1e-3,
        steps: int = 500,
        repeats: int = 1,
        sample_size: int = 32,
        scale: float | None = None,
        seed: int = 0,
        batch_size: int | None = None,
    ) -> None:
        super().__init__(task, params, batch_size=batch_size)
        self.damping, self.steps, self.repeats = damping, steps, repeats
        self.sample_size, self.scale, self.seed = sample_size, scale, seed

    def _prepare(self, inputs: Tensor, targets: Tensor) -> None:
        if self.scale is None:
            full = CurvatureOperator(
                self.task, self.params, inputs, targets, batch_size=self.batch_size
            )
            largest = top_eigenpairs(full, 1, steps=20, seed=self.seed)[0][0]
            self.scale = 1.1 * (float(largest) + self.damping)

    def _solve(self, rhs: Tensor) -> Tensor:
        assert self._train is not None and self.scale is not None
        inputs, targets = self._train
        generator = torch.Generator().manual_seed(self.seed)
        sample = min(self.sample_size, len(inputs))
        total = torch.zeros_like(rhs)
        for _ in range(self.repeats):
            estimate = rhs.clone()
            for _ in range(self.steps):
                # A full sample is the whole dataset in order, which makes
                # the recursion deterministic and is what the tests use.
                chosen = (
                    torch.arange(len(inputs))
                    if sample == len(inputs)
                    else torch.randperm(len(inputs), generator=generator)[:sample]
                ).to(inputs.device)
                batch = CurvatureOperator(self.task, self.params, inputs[chosen], targets[chosen])
                curved = batch @ estimate + self.damping * estimate
                estimate = rhs + estimate - curved / self.scale
            total += estimate / self.scale
        return total / self.repeats


class LowRankInfluence(InfluenceFunction):
    """Influence functions with the curvature replaced by its top eigenpairs.

    Lanczos finds the `rank` largest eigenvalues of the curvature and their
    eigenvectors `U`, at the cost of a few passes over the data per
    eigenpair. After that the inverse is two small matrix products:

        (C + damping I)^-1  ~  U (L + damping)^-1 U^T  +  (I - U U^T) / damping.

    The second term is what the inverse does to everything orthogonal to
    `U` if the curvature there is zero. With it, the approximation is exact
    once `rank` reaches the rank of the curvature. Without it
    (`complement=False`) the query is first projected onto the top
    eigenvectors and everything else is discarded, which is the form
    Schioppa et al. (2022) use to keep only `rank` numbers per example.
    """

    def __init__(
        self,
        task: Task,
        params: Mapping[str, Tensor] | None = None,
        *,
        rank: int = 32,
        damping: float = 1e-3,
        complement: bool = True,
        lanczos_steps: int | None = None,
        seed: int = 0,
        batch_size: int | None = None,
    ) -> None:
        super().__init__(task, params, batch_size=batch_size)
        self.rank, self.damping, self.complement = rank, damping, complement
        self.lanczos_steps, self.seed = lanczos_steps, seed

    def _prepare(self, inputs: Tensor, targets: Tensor) -> None:
        operator = CurvatureOperator(
            self.task, self.params, inputs, targets, batch_size=self.batch_size
        )
        self.eigenvalues, self.eigenvectors = top_eigenpairs(
            operator, self.rank, steps=self.lanczos_steps, seed=self.seed
        )

    def _solve(self, rhs: Tensor) -> Tensor:
        inside = self.eigenvectors.T @ rhs
        scaled = inside / (self.eigenvalues.clamp_min(0.0) + self.damping).unsqueeze(1)
        solution = self.eigenvectors @ scaled
        if self.complement:
            solution = solution + (rhs - self.eigenvectors @ inside) / self.damping
        return solution
