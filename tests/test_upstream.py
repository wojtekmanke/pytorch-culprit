"""Framework behaviour this library works around, pinned down by small reproducers.

Each `xfail` here states what PyTorch would have to do for a workaround to
become unnecessary. A test that starts passing unexpectedly means the
workaround named in its reason can be removed. `docs/notes/pitfalls.md`
explains each one.
"""

import pytest
import torch
from torch.func import grad, hessian, jacfwd, jvp
from torch.nn import functional as F


def layer_norm_loss(z):
    """A scalar function of a layer norm's input, weight and bias, packed in one vector."""
    x, weight, bias = z[:12].view(3, 4), z[12:16], z[16:20]
    return (F.layer_norm(x, (4,), weight, bias) * torch.arange(1.0, 5.0)).pow(2).sum()


def finite_difference_hessian(fn, z, step=1e-5):
    gradient = grad(fn)
    columns = [
        (gradient(z + step * e) - gradient(z - step * e)) / (2 * step) for e in torch.eye(len(z))
    ]
    return torch.stack(columns, dim=1)


def test_forward_over_grad_gives_the_hessian_of_an_affine_layer_norm():
    z = torch.randn(20, generator=torch.Generator().manual_seed(0))
    expected = finite_difference_hessian(layer_norm_loss, z)
    assert torch.allclose(jacfwd(grad(layer_norm_loss))(z), expected, atol=1e-5)


@pytest.mark.xfail(
    reason="torch.func.hessian differentiates a vmapped layer norm backward that treats the "
    "saved mean and scale as constants; dense_hessian uses jacfwd(grad(f)) instead",
    strict=False,
)
def test_torch_func_hessian_of_an_affine_layer_norm():
    z = torch.randn(20, generator=torch.Generator().manual_seed(0))
    expected = finite_difference_hessian(layer_norm_loss, z)
    assert torch.allclose(hessian(layer_norm_loss)(z), expected, atol=1e-5)


def test_the_wrong_hessian_block_is_the_one_with_frozen_statistics():
    """Pins down the mechanism, so the note in the docs is a measurement and not a guess."""
    z = torch.randn(20, generator=torch.Generator().manual_seed(0))
    weight, bias, scale = z[12:16], z[16:20], torch.arange(1.0, 5.0)

    def weight_gradient_with_frozen_statistics(flat_x):
        x = flat_x.view(3, 4)
        mean = x.mean(-1, keepdim=True).detach()
        rstd = (x.var(-1, unbiased=False, keepdim=True) + 1e-5).rsqrt().detach()
        output_gradient = 2 * F.layer_norm(x, (4,), weight, bias) * scale * scale
        return (output_gradient * (x - mean) * rstd).sum(0)

    frozen = jacfwd(weight_gradient_with_frozen_statistics)(z[:12])
    reported = hessian(layer_norm_loss)(z)[12:16, :12]
    correct = jacfwd(grad(layer_norm_loss))(z)[12:16, :12]
    if torch.allclose(reported, correct, atol=1e-8):
        pytest.skip("this version of torch computes the block correctly")
    assert torch.allclose(reported, frozen, atol=1e-10)


def plain_layer_norm_loss(x):
    """A scalar function of the input of a layer norm that has no parameters at all."""
    return (F.layer_norm(x, (4,)) * torch.arange(1.0, 5.0)).pow(2).sum()


def point_and_direction():
    generator = torch.Generator().manual_seed(0)
    return torch.randn(3, 4, generator=generator), torch.randn(3, 4, generator=generator)


def test_forward_over_reverse_gives_the_hvp_through_a_layer_norm():
    x, v = point_and_direction()
    step = 1e-5
    gradient = grad(plain_layer_norm_loss)
    expected = (gradient(x + step * v) - gradient(x - step * v)) / (2 * step)
    assert torch.allclose(jvp(gradient, (x,), (v,))[1], expected, atol=1e-5)


@pytest.mark.xfail(
    reason="the forward-mode rule of layer norm builds its tangent from the saved mean and "
    "scale, which reverse mode treats as constants; the library's default Hessian-vector "
    "product is forward over reverse",
    strict=False,
)
def test_reverse_over_forward_gives_the_hvp_through_a_layer_norm():
    x, v = point_and_direction()
    expected = jvp(grad(plain_layer_norm_loss), (x,), (v,))[1]
    reported = grad(lambda at: jvp(plain_layer_norm_loss, (at,), (v,))[1])(x)
    assert torch.allclose(reported, expected, atol=1e-8)


def test_the_wrong_reverse_over_forward_product_is_the_one_with_frozen_statistics():
    x, v = point_and_direction()
    scale = torch.arange(1.0, 5.0)

    def slope_with_frozen_statistics(at):
        mean = at.mean(-1, keepdim=True).detach()
        rstd = (at.var(-1, unbiased=False, keepdim=True) + 1e-5).rsqrt().detach()
        normalised = (at - mean) * rstd
        centred = v - v.mean(-1, keepdim=True)
        tangent = (centred - normalised * (normalised * centred).mean(-1, keepdim=True)) * rstd
        return (2 * F.layer_norm(at, (4,)) * scale * scale * tangent).sum()

    # The hand-written tangent is the true one; only its derivative differs.
    assert torch.allclose(
        slope_with_frozen_statistics(x), jvp(plain_layer_norm_loss, (x,), (v,))[1]
    )
    reported = grad(lambda at: jvp(plain_layer_norm_loss, (at,), (v,))[1])(x)
    correct = jvp(grad(plain_layer_norm_loss), (x,), (v,))[1]
    if torch.allclose(reported, correct, atol=1e-8):
        pytest.skip("this version of torch computes the product correctly")
    assert torch.allclose(reported, grad(slope_with_frozen_statistics)(x), atol=1e-10)


@pytest.mark.xfail(
    reason="the fused CPU kernel behind scaled_dot_product_attention has no second "
    "derivative; the GPT writes its attention out as plain operations instead",
    strict=False,
    raises=RuntimeError,
)
def test_double_backward_through_scaled_dot_product_attention_on_cpu():
    q, k, v = (
        torch.randn(1, 2, 3, 4, generator=torch.Generator().manual_seed(i)) for i in range(3)
    )

    def attend(q):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True).pow(2).sum()

    grad(lambda q: grad(attend)(q).sum())(q)
