"""The hand-written rules of the fused RMS norm, held to autograd's own."""

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck
from torch.func import functional_call, grad, hessian, jacfwd, jacrev, jvp, stack_module_state, vmap

from culprit.models.rmsnorm import RMSNorm, rms_norm, rms_norm_reference


def inputs(seed=0, shape=(3, 4, 6)):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(*shape, generator=generator)
    weight = torch.rand(shape[-1], generator=generator) + 0.5
    return x, weight


def scalar(fn):
    """A scalar function of `(x, weight)` with a non-trivial dependence on the output."""
    coefficients = torch.linspace(-1.0, 2.0, 6)
    return lambda x, weight: (fn(x, weight) * coefficients).pow(2).sum()


def test_it_computes_the_same_function_as_the_reference():
    x, weight = inputs()
    assert torch.allclose(rms_norm(x, weight), rms_norm_reference(x, weight), atol=1e-14)


def test_the_output_has_unit_root_mean_square_before_scaling():
    x, _ = inputs()
    normalised = rms_norm(x, torch.ones(6), eps=0.0)
    assert torch.allclose(normalised.square().mean(-1), torch.ones(3, 4))


def test_backward_and_forward_rules_pass_gradcheck():
    x, weight = inputs(shape=(2, 3, 6))
    x.requires_grad_()
    weight.requires_grad_()
    assert gradcheck(
        rms_norm,
        (x, weight),
        check_forward_ad=True,
        check_batched_grad=True,
        check_batched_forward_grad=True,
    )


def test_the_rules_are_themselves_differentiable():
    x, weight = inputs(shape=(2, 3, 6))
    x.requires_grad_()
    weight.requires_grad_()
    assert gradgradcheck(rms_norm, (x, weight), check_batched_grad=True)


@pytest.mark.parametrize("argnum", [0, 1])
def test_first_derivatives_match_autograd_on_the_reference(argnum):
    x, weight = inputs()
    mine = grad(scalar(rms_norm), argnums=argnum)(x, weight)
    theirs = grad(scalar(rms_norm_reference), argnums=argnum)(x, weight)
    assert torch.allclose(mine, theirs, atol=1e-12)


def test_forward_mode_matches_autograd_on_the_reference():
    x, weight = inputs()
    tangents = inputs(seed=1)
    mine = jvp(rms_norm, (x, weight), tangents)[1]
    theirs = jvp(rms_norm_reference, (x, weight), tangents)[1]
    assert torch.allclose(mine, theirs, atol=1e-12)


FORWARD_OVER_FORWARD = pytest.mark.xfail(
    reason="forward over forward treats what a custom jvp rule saved as constant", strict=False
)


@pytest.mark.parametrize(
    "compose",
    [
        lambda f: jacfwd(grad(f)),
        lambda f: jacrev(grad(f)),
        hessian,  # jacfwd(jacrev(f)): a derivative of a backward pass that ran under vmap
        lambda f: jacrev(jacrev(f)),
        lambda f: jacrev(jacfwd(f)),  # reverse over forward
        pytest.param(lambda f: jacfwd(jacfwd(f)), marks=FORWARD_OVER_FORWARD),
    ],
    ids=["fwd(grad)", "rev(grad)", "hessian", "rev(rev)", "rev(fwd)", "fwd(fwd)"],
)
def test_every_composition_gives_the_same_hessian(compose):
    """Including the two that `F.layer_norm` gets wrong; see test_upstream.py."""
    x, weight = inputs(shape=(2, 6))

    def of_packed(fn):
        return lambda z: scalar(fn)(z[:12].view(2, 6), z[12:])

    z = torch.cat([x.flatten(), weight])
    expected = jacfwd(grad(of_packed(rms_norm_reference)))(z)
    assert torch.allclose(compose(of_packed(rms_norm))(z), expected, atol=1e-10)
    assert torch.allclose(expected, expected.T, atol=1e-12)


def test_vmap_over_inputs_matches_a_loop():
    x, weight = inputs(shape=(5, 4, 6))
    batched = vmap(rms_norm, in_dims=(0, None))(x, weight)
    looped = torch.stack([rms_norm(row, weight) for row in x])
    assert torch.allclose(batched, looped, atol=1e-14)


def test_vmap_over_a_dimension_that_is_not_the_first():
    x, weight = inputs(shape=(4, 5, 6))
    batched = vmap(rms_norm, in_dims=(1, None))(x, weight)
    looped = torch.stack([rms_norm(x[:, i], weight) for i in range(5)])
    assert torch.allclose(batched, looped, atol=1e-14)


def test_vmap_over_weights_matches_a_loop():
    """One weight per model, as when several models are trained at once."""
    x, _ = inputs(shape=(4, 6))
    weights = torch.rand(3, 6, generator=torch.Generator().manual_seed(2)) + 0.5
    over_weights = vmap(rms_norm, in_dims=(None, 0))(x, weights)
    assert torch.allclose(over_weights, torch.stack([rms_norm(x, w) for w in weights]))
    xs = torch.randn(3, 4, 6, generator=torch.Generator().manual_seed(3))
    over_both = vmap(rms_norm, in_dims=(0, 0))(xs, weights)
    looped = torch.stack([rms_norm(a, w) for a, w in zip(xs, weights, strict=True)])
    assert torch.allclose(over_both, looped)


def test_per_example_gradients_through_the_fused_op():
    x, weight = inputs(shape=(5, 4, 6))
    fn = scalar(rms_norm)
    batched = vmap(grad(fn, argnums=1), in_dims=(0, None))(x, weight)
    looped = torch.stack([grad(scalar(rms_norm_reference), argnums=1)(row, weight) for row in x])
    assert torch.allclose(batched, looped, atol=1e-12)


def test_hessian_vector_products_under_vmap():
    x, weight = inputs(shape=(2, 6))
    directions = torch.randn(4, 2, 6, generator=torch.Generator().manual_seed(5))

    def hvp(fn):
        gradient = grad(lambda at: scalar(fn)(at, weight))
        return vmap(lambda v: jvp(gradient, (x,), (v,))[1])(directions)

    assert torch.allclose(hvp(rms_norm), hvp(rms_norm_reference), atol=1e-11)


def test_the_module_trains_several_models_at_once():
    models = [RMSNorm(6) for _ in range(3)]
    for index, model in enumerate(models):
        with torch.no_grad():
            model.weight.mul_(index + 1.0)
    params, buffers = stack_module_state(models)
    x = torch.randn(4, 6, generator=torch.Generator().manual_seed(0))

    def loss(p, b):
        return functional_call(models[0], (p, b), (x,)).pow(3).sum()

    together = vmap(grad(loss))(params, buffers)["weight"]
    separately = torch.stack(
        [
            grad(lambda w, m=model: (rms_norm_reference(x, w, m.eps)).pow(3).sum())(
                model.weight.detach()
            )
            for model in models
        ]
    )
    assert torch.allclose(together, separately, atol=1e-12)


def test_fused_and_unfused_modules_agree():
    x, _ = inputs()
    fused, plain = RMSNorm(6), RMSNorm(6, fused=False)
    with torch.no_grad():
        fused.weight.uniform_(0.5, 1.5)
        plain.weight.copy_(fused.weight)
    assert torch.allclose(fused(x), plain(x), atol=1e-14)
    assert torch.allclose(fused.normalise(x) * fused.weight, fused(x), atol=1e-14)


def test_the_fused_op_saves_no_intermediate_tensors():
    """One node that holds its inputs, where the reference holds one tensor per operation."""
    x, weight = inputs()
    x.requires_grad_()
    fused = rms_norm(x, weight)
    assert type(fused.grad_fn).__name__ == "RMSNormFunctionBackward"
    assert {id(t) for t in fused.grad_fn.saved_tensors} == {id(x), id(weight)}

    def nodes(output):
        seen, stack = set(), [output.grad_fn]
        while stack:
            node = stack.pop()
            if node is None or node in seen:
                continue
            seen.add(node)
            stack.extend(parent for parent, _ in node.next_functions)
        return len(seen)

    assert nodes(fused) < nodes(rms_norm_reference(x, weight))


@pytest.mark.mps
def test_rules_on_the_gpu_match_the_reference():
    x, weight = (t.to("mps", torch.float32) for t in inputs())

    def loss(fn):
        return lambda a, w: fn(a, w).pow(3).sum()

    for argnum in (0, 1):
        mine = grad(loss(rms_norm), argnums=argnum)(x, weight)
        theirs = grad(loss(rms_norm_reference), argnums=argnum)(x, weight)
        assert torch.allclose(mine, theirs, atol=1e-4)
