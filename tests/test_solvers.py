import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from culprit.curvature import dense_ggn
from culprit.linalg import CurvatureOperator, DenseOperator
from culprit.linalg.solvers import conjugate_gradient, lanczos, top_eigenpairs
from helpers import gpt_problem


def spd(size, seed=0, floor=0.1):
    """A symmetric positive-definite matrix with eigenvalues at least `floor`."""
    factor = torch.randn(size, size, generator=torch.Generator().manual_seed(seed))
    return factor @ factor.T / size + floor * torch.eye(size)


def decaying(size, seed=0):
    """A symmetric matrix with eigenvalues 1, 1/2, 1/3, ... and known eigenvectors."""
    generator = torch.Generator().manual_seed(seed)
    vectors, _ = torch.linalg.qr(torch.randn(size, size, generator=generator))
    values = 1.0 / torch.arange(1, size + 1)
    return vectors @ torch.diag(values) @ vectors.T, values, vectors


def test_cg_solves_one_right_hand_side():
    matrix = spd(20)
    rhs = torch.randn(20)
    solution, info = conjugate_gradient(DenseOperator(matrix), rhs, tol=1e-12)
    assert solution.shape == (20,)
    assert torch.allclose(solution, torch.linalg.solve(matrix, rhs), atol=1e-9)
    assert info.converged and info.residual.shape == (1,)


def test_cg_solves_several_right_hand_sides_at_once():
    matrix = spd(20)
    rhs = torch.randn(20, 5)
    solution, info = conjugate_gradient(DenseOperator(matrix), rhs, tol=1e-12)
    assert torch.allclose(solution, torch.linalg.solve(matrix, rhs), atol=1e-9)
    assert info.converged and bool((info.residual <= 1e-12).all())


def test_cg_handles_columns_that_finish_at_different_times():
    matrix, values, vectors = decaying(12)
    # An eigenvector converges in one step; a zero column needs none.
    rhs = torch.stack([vectors[:, 0], torch.randn(12), torch.zeros(12)], dim=1)
    solution, info = conjugate_gradient(DenseOperator(matrix), rhs, tol=1e-12)
    assert info.converged
    assert torch.allclose(solution[:, 0], vectors[:, 0] / values[0], atol=1e-10)
    assert torch.allclose(matrix @ solution[:, 1], rhs[:, 1], atol=1e-9)
    assert torch.equal(solution[:, 2], torch.zeros(12))
    assert not torch.isnan(solution).any()


def test_cg_reports_when_it_runs_out_of_iterations():
    matrix = spd(30, floor=1e-4)
    rhs = torch.randn(30)
    solution, info = conjugate_gradient(DenseOperator(matrix), rhs, tol=1e-12, max_iterations=3)
    assert info.iterations == 3 and not info.converged
    assert info.residual.item() == pytest.approx(
        ((matrix @ solution - rhs).norm() / rhs.norm()).item(), rel=1e-6
    )


def test_cg_with_an_exact_preconditioner_takes_one_step():
    matrix = spd(20)
    rhs = torch.randn(20, 3)
    inverse = DenseOperator(torch.linalg.inv(matrix))
    solution, info = conjugate_gradient(
        DenseOperator(matrix), rhs, tol=1e-10, preconditioner=inverse
    )
    assert info.iterations == 1
    assert torch.allclose(solution, torch.linalg.solve(matrix, rhs), atol=1e-9)


def test_cg_with_a_rough_preconditioner_needs_fewer_steps():
    # A badly scaled matrix, for which the inverse of its diagonal is a real help.
    scales = torch.logspace(0, 3, 40)
    matrix = spd(40) * scales[:, None] * scales[None, :]
    rhs = torch.randn(40)
    plain = conjugate_gradient(DenseOperator(matrix), rhs, tol=1e-8, max_iterations=2000)[1]
    jacobi = DenseOperator(torch.diag(1.0 / matrix.diagonal()))
    helped = conjugate_gradient(
        DenseOperator(matrix), rhs, tol=1e-8, max_iterations=2000, preconditioner=jacobi
    )[1]
    assert helped.converged and helped.iterations < plain.iterations


def test_cg_refuses_an_indefinite_operator():
    matrix = torch.diag(torch.tensor([1.0, -1.0, 2.0]))
    with pytest.raises(ValueError, match="add damping"):
        conjugate_gradient(DenseOperator(matrix), torch.ones(3))


@settings(max_examples=25, deadline=None)
@given(size=st.integers(1, 15), columns=st.integers(1, 4), seed=st.integers(0, 10_000))
def test_cg_residual_is_below_tolerance_for_any_spd_system(size, columns, seed):
    matrix = spd(size, seed).double()
    rhs = torch.randn(
        size, columns, generator=torch.Generator().manual_seed(seed + 1), dtype=torch.float64
    )
    solution, info = conjugate_gradient(
        DenseOperator(matrix), rhs, tol=1e-9, max_iterations=5 * size
    )
    assert info.converged
    assert ((matrix @ solution - rhs).norm(dim=0) <= 1e-8 * rhs.norm(dim=0) + 1e-12).all()


def test_cg_inverts_the_damped_curvature_of_a_model():
    task, x, y = gpt_problem()
    params = task.params()
    operator = CurvatureOperator(task, params, x, y, batch_size=2).damped(1e-2)
    rhs = torch.randn(operator.size, 2, generator=torch.Generator().manual_seed(0))
    solution, info = conjugate_gradient(operator, rhs, tol=1e-10, max_iterations=500)
    dense = dense_ggn(task, params, x, y) + 1e-2 * torch.eye(operator.size)
    assert info.converged
    assert torch.allclose(solution, torch.linalg.solve(dense, rhs), atol=1e-7)


def test_lanczos_tridiagonalises_the_operator():
    matrix = spd(15)
    result = lanczos(DenseOperator(matrix), steps=15)
    basis = result.basis
    assert basis.shape == (15, 15)
    assert torch.allclose(basis.T @ basis, torch.eye(15), atol=1e-10)
    assert torch.allclose(basis.T @ matrix @ basis, result.tridiagonal(), atol=1e-10)
    values, vectors = result.ritz()
    assert torch.allclose(values, torch.linalg.eigvalsh(matrix), atol=1e-10)
    assert torch.allclose(matrix @ vectors, vectors * values, atol=1e-9)


def test_lanczos_stops_when_the_krylov_space_is_exhausted():
    generator = torch.Generator().manual_seed(0)
    factor = torch.randn(20, 3, generator=generator)
    matrix = factor @ factor.T  # rank 3: three non-zero eigenvalues and zero
    result = lanczos(DenseOperator(matrix), steps=15)
    assert result.basis.shape[1] == 4
    values, _ = result.ritz()
    expected = torch.linalg.eigvalsh(matrix)[-3:]
    assert torch.allclose(values[-3:], expected, atol=1e-9)
    assert abs(values[0].item()) < 1e-9


def test_lanczos_stays_orthogonal_long_after_eigenvalues_converge():
    matrix, _, _ = decaying(60)
    basis = lanczos(DenseOperator(matrix), steps=50).basis
    assert (basis.T @ basis - torch.eye(basis.shape[1])).abs().max() < 1e-10


def test_top_eigenpairs_finds_the_largest_ones():
    matrix, values, vectors = decaying(80)
    found_values, found_vectors = top_eigenpairs(DenseOperator(matrix), k=5)
    assert found_values.shape == (5,) and found_vectors.shape == (80, 5)
    assert torch.allclose(found_values, values[:5], atol=1e-8)
    # Eigenvectors are defined up to sign.
    overlap = (found_vectors * vectors[:, :5]).sum(0).abs()
    assert torch.allclose(overlap, torch.ones(5), atol=1e-6)


def test_top_eigenpairs_of_a_models_curvature():
    task, x, y = gpt_problem()
    params = task.params()
    operator = CurvatureOperator(task, params, x, y)
    values, vectors = top_eigenpairs(operator, k=3, steps=40)
    expected = torch.linalg.eigvalsh(dense_ggn(task, params, x, y)).flip(0)[:3]
    assert torch.allclose(values, expected, rtol=1e-6)
    assert torch.allclose(operator @ vectors, vectors * values, atol=1e-6 * float(expected[0]))


@pytest.mark.mps
def test_solvers_run_in_single_precision_on_the_gpu():
    matrix = spd(40).to("mps", torch.float32)
    rhs = torch.randn(40, 2).to("mps", torch.float32)
    solution, info = conjugate_gradient(DenseOperator(matrix), rhs, tol=1e-5)
    assert info.converged
    assert ((matrix @ solution - rhs).norm(dim=0) / rhs.norm(dim=0)).max() < 1e-4
    values, vectors = top_eigenpairs(DenseOperator(matrix), k=3)
    assert values.device.type == "mps" and values.dtype == torch.float32
    expected = torch.linalg.eigvalsh(matrix.cpu().double()).flip(0)[:3]
    assert torch.allclose(values.cpu().double(), expected, rtol=1e-4)
