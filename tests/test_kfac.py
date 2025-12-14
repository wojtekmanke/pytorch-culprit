import pytest
import torch

from culprit.curvature import dense_empirical_fisher, dense_ggn
from culprit.curvature.kfac import ekfac, kfac
from culprit.data.blobs import gaussian_blobs
from culprit.flat import Layout
from culprit.models import MLP
from culprit.task import Task
from helpers import gpt_problem, mlp_problem

MLP_LAYERS = {
    "layers.0": ("layers.0.weight", "layers.0.bias"),
    "layers.2": ("layers.2.weight", "layers.2.bias"),
}


def block_of(matrix, layout, names):
    """The part of a full matrix that pairs the given parameters with themselves."""
    slices = layout.slices()
    index = torch.cat([torch.arange(slices[n].start, slices[n].stop) for n in names])
    return matrix[index][:, index]


def test_kfac_is_exact_for_one_example():
    """With a single example there is no expectation to split, so nothing is approximated."""
    x, y = gaussian_blobs(1, dim=5, classes=3)
    task = Task(MLP([5, 3]))
    params = task.params()
    operator = kfac(task, params, x, y, fisher="exact")
    assert torch.allclose(operator.dense(), dense_ggn(task, params, x, y), atol=1e-12)


def test_kfac_blocks_are_the_kronecker_products_of_the_factors():
    """Pins the orientation: weight `[out, in]` pairs `S` with rows and `A` with columns."""
    x, y = gaussian_blobs(9, dim=5, classes=3)
    task = Task(MLP([5, 3]), param_names=("layers.0.weight",))
    params = task.params()
    operator = kfac(task, params, x, y, fisher="exact")
    a = x.T @ x / len(x)
    p = task.outputs(params, x).softmax(-1)
    s = (torch.diag_embed(p) - p[:, :, None] * p[:, None, :]).mean(0)
    assert torch.allclose(operator.dense(), torch.kron(s, a), atol=1e-12)


@pytest.mark.parametrize("layer", MLP_LAYERS)
def test_ekfac_is_at_least_as_close_as_kfac_to_the_true_block(layer):
    task, x, y = mlp_problem(n=30)
    params = task.params()
    layout = Layout.of(params)
    true = block_of(dense_ggn(task, params, x, y), layout, MLP_LAYERS[layer])
    plain = block_of(kfac(task, params, x, y, fisher="exact").dense(), layout, MLP_LAYERS[layer])
    corrected = block_of(
        ekfac(task, params, x, y, fisher="exact").dense(), layout, MLP_LAYERS[layer]
    )
    assert (corrected - true).norm() <= (plain - true).norm() + 1e-12
    assert (plain - true).norm() > 1e-3  # the comparison is not vacuous


@pytest.mark.parametrize("layer", MLP_LAYERS)
def test_ekfac_eigenvalues_are_the_true_second_moments_in_its_basis(layer):
    """For a matrix `B` diagonal in a basis, `B` holding the diagonal of `F` in that basis
    is equivalent to `trace(B^k (B - F)) = 0` for every power `k`."""
    task, x, y = mlp_problem(n=30)
    params = task.params()
    layout = Layout.of(params)
    true = block_of(dense_ggn(task, params, x, y), layout, MLP_LAYERS[layer])
    corrected = block_of(
        ekfac(task, params, x, y, fisher="exact").dense(), layout, MLP_LAYERS[layer]
    )
    assert torch.trace(corrected).item() == pytest.approx(torch.trace(true).item(), rel=1e-10)
    for power in (1, 2, 3):
        mismatch = torch.trace(torch.linalg.matrix_power(corrected, power) @ (corrected - true))
        assert abs(mismatch.item()) < 1e-12 * torch.trace(true).item() ** (power + 1)


def test_ekfac_fits_every_layer_type_of_the_gpt():
    """With the data's labels there is no sampling, so the fit must be exact."""
    task, x, y = gpt_problem()
    params = task.params()
    layout = Layout.of(params)
    true = dense_empirical_fisher(task, params, x, y)
    corrected = ekfac(task, params, x, y, fisher="empirical")
    dense = corrected.dense()
    assert torch.allclose(dense, dense.T, atol=1e-12)
    for block in corrected.blocks.values():
        mine = block_of(dense, layout, block.names)
        theirs = block_of(true, layout, block.names)
        assert torch.trace(mine).item() == pytest.approx(torch.trace(theirs).item(), rel=1e-9)
        mismatch = torch.trace(mine @ (mine - theirs))
        assert abs(mismatch.item()) < 1e-10 * torch.trace(theirs).item() ** 2


def test_layer_norm_blocks_are_not_approximated():
    task, x, y = gpt_problem()
    params = task.params()
    layout = Layout.of(params)
    true = dense_empirical_fisher(task, params, x, y)
    dense = kfac(task, params, x, y, fisher="empirical").dense()
    for names in (("ln_f.weight", "ln_f.bias"), ("blocks.0.ln1.weight", "blocks.0.ln1.bias")):
        assert torch.allclose(
            block_of(dense, layout, names), block_of(true, layout, names), atol=1e-12
        )


def test_blocks_of_different_layers_do_not_interact():
    task, x, y = mlp_problem()
    params = task.params()
    layout = Layout.of(params)
    dense = kfac(task, params, x, y, fisher="exact").dense()
    slices = layout.slices()
    first = slice(slices["layers.0.weight"].start, slices["layers.0.bias"].stop)
    second = slice(slices["layers.2.weight"].start, slices["layers.2.bias"].stop)
    assert torch.count_nonzero(dense[first, second]) == 0
    assert dense[first, first].abs().max() > 0


@pytest.mark.parametrize("build", [kfac, ekfac])
def test_the_damped_inverse_inverts(build):
    task, x, y = gpt_problem()
    params = task.params()
    operator = build(task, params, x, y, samples=2)
    damped = operator.dense() + 0.1 * torch.eye(operator.size)
    inverse = operator.inverse(0.1)
    assert torch.allclose(inverse.dense() @ damped, torch.eye(operator.size), atol=1e-9)
    vector = torch.randn(operator.size, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(inverse @ vector, torch.linalg.solve(damped, vector), atol=1e-9)


def test_products_with_a_block_of_columns_match_one_column_at_a_time():
    task, x, y = gpt_problem()
    operator = ekfac(task, task.params(), x, y)
    block = torch.randn(operator.size, 3, generator=torch.Generator().manual_seed(0))
    one_by_one = torch.stack([operator @ column for column in block.unbind(1)], dim=1)
    assert torch.allclose(operator @ block, one_by_one, atol=1e-12)


def test_sampled_ekfac_approaches_the_gauss_newton_trace():
    task, x, y = gpt_problem()
    params = task.params()
    true = torch.trace(dense_ggn(task, params, x, y)).item()
    few = torch.trace(ekfac(task, params, x, y, samples=1).dense()).item()
    many = torch.trace(ekfac(task, params, x, y, samples=300).dense()).item()
    assert many == pytest.approx(true, rel=0.05)
    assert abs(many - true) < abs(few - true) or abs(few - true) < 0.05 * true


def test_a_restricted_task_gives_an_operator_over_its_parameters_only():
    names = ("layers.2.weight", "layers.2.bias")
    task, x, y = mlp_problem(param_names=names)
    whole_task, _, _ = mlp_problem()
    operator = kfac(task, task.params(), x, y, fisher="exact")
    whole = kfac(whole_task, whole_task.params(), x, y, fisher="exact").dense()
    expected = block_of(whole, Layout.of(whole_task.params()), names)
    assert operator.size == Layout.of(task.params()).size
    assert torch.allclose(operator.dense(), expected, atol=1e-12)


@pytest.mark.mps
def test_kfac_on_the_gpu_matches_double_precision_on_the_cpu():
    task, x, y = gpt_problem()
    reference = ekfac(task, task.params(), x, y, fisher="empirical").dense()
    task.model.to("mps", torch.float32)
    operator = ekfac(task, task.params(), x.to("mps"), y.to("mps"), fisher="empirical")
    assert operator.device.type == "mps" and operator.dtype == torch.float32
    difference = (operator.dense().cpu().double() - reference).norm()
    assert difference < 1e-3 * reference.norm()
