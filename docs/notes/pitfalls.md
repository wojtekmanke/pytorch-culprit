# PyTorch pitfalls found along the way

Things that went wrong without an error message, or that shaped a design
decision. Each has a reproducer in `tests/test_upstream.py`. Versions are
the ones the behaviour was observed on; none of this is a claim about other
versions.

## `torch.func.hessian` is wrong for a layer norm with affine parameters

Observed on torch 2.14.1, CPU, float64.

`torch.func.hessian(f)` is `jacfwd(jacrev(f))`. For a model containing
`F.layer_norm(x, shape, weight, bias)` it returns a matrix that is not
symmetric: the block that pairs a layer norm's weight with anything upstream
of that layer is wrong. On the two-layer test GPT the matrix was asymmetric
by up to 0.54, against a largest entry of 32. No warning is raised.

What happens:

- `jacrev` computes the gradient by running the backward pass under `vmap`,
  one basis vector per output. For a scalar function that is a batch of one,
  but it is still a batch.
- Under `vmap` the backward of layer norm does not use the ordinary kernel.
  Its batching rule rebuilds the gradient of the weight as
  `sum(grad_out * (x - mean) * rstd)`, taking `mean` and `rstd` from the
  forward pass.
- Those two were returned by the forward pass as outputs that are not
  differentiable. The outer `jacfwd` therefore sees them as constants and
  loses the way they depend on `x`.

The last point is measured, not assumed: the wrong block equals, to
1e-15, the derivative of that expression with `mean` and `rstd` detached
(`test_the_wrong_hessian_block_is_the_one_with_frozen_statistics`).

The same composition is wrong whichever mode is outermost:

| composition | largest error against finite differences |
|---|---|
| `jacfwd(jacrev(f))`, i.e. `hessian(f)` | 56 |
| `jacrev(jacrev(f))` | 56 |
| `jacfwd(grad(f))` | 9.5e-9 |
| `jacrev(grad(f))` | 9.5e-9 |
| plain `autograd.grad` twice | 9.5e-9 |
| `vmap` over `jvp(grad(f))` | 9.5e-9 |

These are for the 20-parameter reproducer in the tests, whose largest
Hessian entry is 194.

The rule this library follows: a derivative may be taken of a backward
pass, and `vmap` may be applied outside a derivative, but a derivative is
never taken of a backward pass that ran under `vmap`. `dense_hessian` uses
`jacfwd(grad(f))`. Writing the layer as `F.layer_norm(x, shape) * weight +
bias` also avoids the problem, but a library that takes other people's
models cannot ask for that.

It was found by the test that compares the dense Hessian with finite
differences of the gradient, which uses no second-order autograd at all.

## Reverse over forward is wrong through any layer norm

Observed on torch 2.14.1, CPU in float64 and MPS in float32.

A Hessian-vector product can be computed as the gradient of a directional
derivative: `grad(lambda p: jvp(f, (p,), (v,))[1])`. Through
`F.layer_norm`, with or without affine parameters, the result is wrong. On
the 12-element reproducer in the tests it is off by 86 where the largest
entry of the true product is 70.

The cause is the same as above, seen from the other side. The forward-mode
rule of layer norm builds the tangent of its output from the saved mean and
scale. Those are not differentiable outputs of the forward pass, so the
reverse pass through the tangent treats them as constants. The wrong
product equals, to 4e-14, the gradient of a hand-written directional
derivative with both detached
(`test_the_wrong_reverse_over_forward_product_is_the_one_with_frozen_statistics`).

Forward over reverse, `jvp(grad(f), (p,), (v,))[1]`, is correct, and so is
reverse over reverse with plain `autograd.grad`. This library uses forward
over reverse wherever it needs a Hessian-vector product, and tests all
three against the dense Hessian; reverse over forward is held to it only on
the MLP, which has no layer norm.

It was found by that comparison: the three ways of computing the product
were supposed to be a formality.

## `scaled_dot_product_attention` cannot be differentiated twice on CPU

Observed on torch 2.14.1.

On CPU the function dispatches to a fused kernel. Its backward has no
derivative of its own and the kernel has no forward-mode rule, so both ways
of forming a Hessian-vector product fail:

```
RuntimeError: derivative for aten::_scaled_dot_product_flash_attention_for_cpu_backward is not implemented
NotImplementedError: Trying to use forward AD with _scaled_dot_product_flash_attention_for_cpu that does not support it
```

On MPS the same call decomposes into plain operations and works. The GPT in
`culprit.models` writes attention as matmul, mask and softmax so that it
behaves the same on both.

## One call from MPS to float64 on the CPU returns zeros

Observed on torch 2.14.1, Darwin 25.6, Apple M5 Max.

```python
>>> torch.arange(1.0, 5.0, device="mps").to("cpu", torch.float64)
tensor([0., 0., 0., 0.], dtype=torch.float64)
>>> torch.arange(1.0, 5.0, device="mps").cpu().double()
tensor([1., 2., 3., 4.], dtype=torch.float64)
```

MPS has no float64, so asking for both the move and the widening at once
has to be split into a move and then a widening on the CPU. The single call
does not do that and does not raise either. In one run it returned zeros and
in another values such as `2.1e-314`, which is what uninitialised memory
looks like.

It showed up as Lanczos returning eigenvalues of zero on the GPU: the small
tridiagonal matrix is solved on the CPU in double precision, and reached it
empty. `culprit.device.cpu_double` does the two steps in order and is the
only way this library moves numbers off the GPU for a double-precision
check.

## A module output without a batch dimension hides per-example gradients

This is a property of the hook approach, not a bug. Hooks recover an
example's gradient from the gradient at a layer's output, which requires
that output to have one row per example. A positional embedding looked up
once as `[steps, width]` and broadcast over the batch has no such row: by
the time a hook sees its gradient, it is already summed over the batch. The
GPT therefore looks positions up as `[batch, steps]`.
