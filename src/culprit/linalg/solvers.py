"""Solvers that touch an operator only by multiplying with it."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from culprit.device import cpu_double
from culprit.linalg.operator import LinearOperator


@dataclass(frozen=True)
class SolveInfo:
    iterations: int
    residual: Tensor  # `|A x - b| / |b|` of each right-hand side, shape `[k]`
    converged: bool


def conjugate_gradient(
    operator: LinearOperator,
    rhs: Tensor,
    *,
    tol: float = 1e-8,
    max_iterations: int | None = None,
    preconditioner: LinearOperator | None = None,
) -> tuple[Tensor, SolveInfo]:
    """Solve `operator @ x = rhs` for a symmetric positive-definite operator.

    `rhs` is one vector `[size]` or several as the columns of `[size, k]`.
    Columns are solved independently, but each iteration is a single
    `operator @ block`, which a `CurvatureOperator` turns into one pass over
    the data for all of them. A column stops being updated once its relative
    residual is below `tol`.

    `preconditioner` approximates the inverse of the operator. A good one
    clusters the eigenvalues and cuts the number of iterations; an exact one
    makes the first iteration land on the solution.
    """
    single = rhs.dim() == 1
    b = rhs.unsqueeze(1) if single else rhs
    # In exact arithmetic `size` iterations always suffice. Rounding makes it
    # take somewhat more, and for a large operator far fewer are affordable.
    limit = min(2 * operator.size, 1000) if max_iterations is None else max_iterations

    def precondition(residual: Tensor) -> Tensor:
        return residual if preconditioner is None else preconditioner @ residual

    x = torch.zeros_like(b)
    r = b.clone()
    z = precondition(r)
    p = z.clone()
    rz = (r * z).sum(0)
    # A zero right-hand side has the zero solution; the clamp makes its
    # relative residual zero so that the column is never iterated on.
    b_norm = b.norm(dim=0).clamp_min(torch.finfo(b.dtype).tiny)
    residual = r.norm(dim=0) / b_norm
    active = residual > tol
    zero = torch.zeros_like(rz)

    iterations = 0
    while iterations < limit and bool(active.any()):
        ap = operator @ p
        curvature = (p * ap).sum(0)
        if bool((curvature[active] <= 0).any()):
            raise ValueError(
                "conjugate gradient met a direction of non-positive curvature: "
                "the operator is not positive definite, add damping"
            )
        # Finished columns get a step of zero; the division they would do is
        # of two vanishing numbers and is only kept out of the result here.
        alpha = torch.where(active, rz / curvature, zero)
        x = x + alpha * p
        r = r - alpha * ap
        z = precondition(r)
        rz_next = (r * z).sum(0)
        beta = torch.where(active, rz_next / rz, zero)
        p = z + beta * p
        rz = rz_next
        residual = r.norm(dim=0) / b_norm
        active = active & (residual > tol)
        iterations += 1

    info = SolveInfo(iterations=iterations, residual=residual, converged=not bool(active.any()))
    return (x.squeeze(1) if single else x), info


@dataclass(frozen=True)
class Lanczos:
    """An orthonormal `basis` in which the operator is the tridiagonal matrix `T`.

    `alphas` is the diagonal of `T` and `betas` its off-diagonal, so
    `basis.T @ A @ basis = T`. The eigenpairs of the small matrix `T` give
    approximations to eigenpairs of `A`, best at the ends of the spectrum.
    """

    alphas: Tensor  # [m]
    betas: Tensor  # [m - 1]
    basis: Tensor  # [size, m]

    def tridiagonal(self) -> Tensor:
        return torch.diag(self.alphas) + torch.diag(self.betas, 1) + torch.diag(self.betas, -1)

    def ritz(self) -> tuple[Tensor, Tensor]:
        """Approximate eigenvalues, ascending, and their eigenvectors `[size, m]`."""
        # The small eigenproblem is solved on the CPU in double precision:
        # MPS has no eigh and no float64, and the matrix is tiny.
        values, coefficients = torch.linalg.eigh(cpu_double(self.tridiagonal()))
        vectors = self.basis @ coefficients.to(self.basis.device, self.basis.dtype)
        return values.to(self.basis.device, self.basis.dtype), vectors


def lanczos(
    operator: LinearOperator, steps: int, *, start: Tensor | None = None, seed: int = 0
) -> Lanczos:
    """Run at most `steps` steps of the Lanczos iteration.

    Every new direction is orthogonalised against all the previous ones,
    twice. The textbook three-term recurrence is orthogonal only in exact
    arithmetic; in floating point it loses orthogonality as soon as an
    eigenvalue converges and then reports that eigenvalue again and again.

    The iteration stops early when the Krylov space is exhausted, which
    happens after as many steps as the operator has distinct eigenvalues
    excited by the starting vector.
    """
    if start is None:
        start = torch.randn(operator.size, generator=torch.Generator().manual_seed(seed))
    q = start.to(operator.device, operator.dtype)
    basis = [q / q.norm()]
    alphas: list[Tensor] = []
    betas: list[Tensor] = []
    scale = 0.0
    steps = min(steps, operator.size)
    for step in range(steps):
        w = operator @ basis[-1]
        alphas.append(w @ basis[-1])
        stacked = torch.stack(basis, dim=1)
        for _ in range(2):
            w = w - stacked @ (stacked.T @ w)
        beta = w.norm()
        scale = max(scale, float(alphas[-1].abs()), float(beta))
        if step == steps - 1 or float(beta) <= torch.finfo(w.dtype).eps ** 0.5 * scale:
            break
        betas.append(beta)
        basis.append(w / beta)
    return Lanczos(
        alphas=torch.stack(alphas),
        betas=torch.stack(betas) if betas else alphas[0].new_zeros(0),
        basis=torch.stack(basis, dim=1),
    )


def top_eigenpairs(
    operator: LinearOperator, k: int, *, steps: int | None = None, seed: int = 0
) -> tuple[Tensor, Tensor]:
    """The `k` largest eigenvalues, descending, and their eigenvectors `[size, k]`.

    Lanczos needs more steps than eigenpairs wanted, because the Ritz values
    at the edge of the basis are the last to converge. The default runs
    `2k + 20`.
    """
    result = lanczos(operator, steps if steps is not None else 2 * k + 20, seed=seed)
    values, vectors = result.ritz()
    return values.flip(0)[:k], vectors.flip(1)[:, :k]
