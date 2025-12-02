import pytest
import torch

from culprit.curvature import dense_ggn, dense_hessian
from culprit.flat import Layout
from culprit.linalg import BlockDiagonal, CurvatureOperator, DenseOperator
from helpers import gpt_problem, mlp_problem

PROBLEMS = {"mlp": mlp_problem, "gpt": gpt_problem}
DENSE = {"ggn": dense_ggn, "hessian": dense_hessian}


def symmetric(size, seed=0):
    matrix = torch.randn(size, size, generator=torch.Generator().manual_seed(seed))
    return matrix + matrix.T


def test_dense_operator_multiplies_vectors_and_matrices():
    matrix = symmetric(6)
    operator = DenseOperator(matrix)
    vector, block = torch.randn(6), torch.randn(6, 3)
    assert torch.equal(operator @ vector, matrix @ vector)
    assert torch.equal(operator @ block, matrix @ block)
    assert torch.allclose(operator.dense(), matrix)
    assert (operator.size, operator.dtype) == (6, matrix.dtype)


def test_default_matmat_applies_matvec_to_each_column():
    class ColumnByColumn(DenseOperator):
        def matmat(self, matrix):
            return super(DenseOperator, self).matmat(matrix)

    matrix = symmetric(5)
    block = torch.randn(5, 4)
    assert torch.allclose(ColumnByColumn(matrix) @ block, matrix @ block)


def test_sum_and_damping_compose():
    a, b = symmetric(5, 1), symmetric(5, 2)
    combined = (DenseOperator(a) + DenseOperator(b)).damped(0.3)
    assert torch.allclose(combined.dense(), a + b + 0.3 * torch.eye(5))
    vector = torch.randn(5)
    assert torch.allclose(combined @ vector, (a + b) @ vector + 0.3 * vector)


def test_operators_reject_mismatched_shapes():
    operator = DenseOperator(symmetric(5))
    with pytest.raises(ValueError, match="cannot apply"):
        operator @ torch.randn(4)
    with pytest.raises(ValueError, match="cannot apply"):
        operator @ torch.randn(5, 2, 2)
    with pytest.raises(ValueError, match="cannot add"):
        operator + DenseOperator(symmetric(4))
    with pytest.raises(ValueError, match="square"):
        DenseOperator(torch.randn(3, 4))


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("kind", DENSE)
def test_curvature_operator_is_the_dense_matrix(problem, kind):
    task, x, y = PROBLEMS[problem](n=7)
    params = task.params()
    # A batch size that does not divide the data, so the last batch is short.
    operator = CurvatureOperator(task, params, x, y, kind=kind, batch_size=3)
    expected = DENSE[kind](task, params, x, y)
    vector = torch.randn(operator.size, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(operator @ vector, expected @ vector, atol=1e-10)
    assert torch.allclose(operator.dense(), expected, atol=1e-10)


@pytest.mark.parametrize("kind", DENSE)
def test_batched_products_match_one_product_per_column(kind):
    task, x, y = gpt_problem()
    operator = CurvatureOperator(task, task.params(), x, y, kind=kind, batch_size=2)
    block = torch.randn(operator.size, 4, generator=torch.Generator().manual_seed(0))
    one_by_one = torch.stack([operator @ column for column in block.unbind(1)], dim=1)
    assert torch.allclose(operator @ block, one_by_one, atol=1e-10)
    chunked = CurvatureOperator(task, task.params(), x, y, kind=kind, chunk_size=3)
    assert torch.allclose(chunked @ block, one_by_one, atol=1e-10)


def test_damped_curvature_is_positive_definite():
    task, x, y = mlp_problem()
    operator = CurvatureOperator(task, task.params(), x, y).damped(1e-3)
    assert torch.linalg.eigvalsh(operator.dense()).min() >= 1e-3 - 1e-10


def test_block_diagonal_acts_on_each_parameter_separately():
    task, x, y = mlp_problem()
    params = task.params()
    layout = Layout.of(params)
    full = dense_ggn(task, params, x, y)
    slices = layout.slices()
    blocks = {name: DenseOperator(full[where, where]) for name, where in slices.items()}
    operator = BlockDiagonal(layout, blocks)

    expected = torch.zeros_like(full)
    for where in slices.values():
        expected[where, where] = full[where, where]
    assert operator.size == layout.size
    assert torch.allclose(operator.dense(), expected)
    vector = torch.randn(layout.size)
    assert torch.allclose(operator @ vector, expected @ vector)


def test_block_diagonal_checks_its_blocks():
    task, _, _ = mlp_problem()
    layout = Layout.of(task.params())
    blocks = {
        name: DenseOperator(torch.eye(size))
        for name, size in zip(layout.names, layout.sizes, strict=True)
    }
    with pytest.raises(ValueError, match="every parameter"):
        BlockDiagonal(layout, dict(list(blocks.items())[:-1]))
    blocks[layout.names[0]] = DenseOperator(torch.eye(2))
    with pytest.raises(ValueError, match="has size 2"):
        BlockDiagonal(layout, blocks)
