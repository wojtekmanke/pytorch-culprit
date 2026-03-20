# Roadmap

What was planned, what landed, and what is still open. The plan had four
stages, each ending in something that can be checked: the first against
dense linear algebra on tiny models, the later ones against retraining.

## Landed

**Stage 1: exact references on tiny models**

- Packaging, lint, types, tests, CI on CPU wheels.
- A tiny MLP and a two-layer GPT that can be called functionally.
- A planted-facts corpus whose sources are known by construction.
- A deterministic training loop with checkpoints.
- Per-example gradients two ways: `vmap` over `grad`, and hooks.
- Dense Hessian and Gauss-Newton matrices as the oracle.
- Hessian-vector products three ways and Gauss-Newton-vector products.
- Matrix-free operators, conjugate gradient and Lanczos.
- Exact influence functions, tested as the derivative of retraining on a
  convex problem.
- A device policy from measurements of float32 on the GPU against float64.

**Stage 2: approximations that scale**

- K-FAC and EK-FAC, including layers shared across sequence positions.
- TracIn over checkpoints, on gradient dot products that do not form the
  gradients.
- TRAK on a sparse random projection, with a resumable gradient store.
- Conjugate gradient, LiSSA and low-rank influence.
- One `Attributor` interface, a registry and a command line.
- A fused RMS norm as a custom `autograd.Function`.
- A profile of every stage, and a benchmark of every method.

**Stage 3: ground truth by retraining**

- Functional AdamW and SGD that match `torch.optim` update for update.
- Many models trained at once through `vmap`, each on its own subset.
- A retraining harness, the linear datamodeling score and source recall.
- Unrolled differentiation through a whole training run.
- Every method compared on MNIST digits under two optimisers and on the
  planted-facts GPT.
- Measurements of half precision and of `torch.compile`.

**Stage 4: a real model**

- LoRA adapters as a parametrization, with every method working through
  them, and a pretrained language model wrapped as a task.
- A backdoor planted in Qwen2.5-0.5B by fine-tuning on poisoned reviews,
  each method asked to find them, and the counterfactual check of removing
  what a method blamed.
- An account of where the methods fail.
- A quickstart as a script and an executed notebook, and version 0.1.0.

## Found along the way and not in the plan

- Three compositions of PyTorch's transforms that return wrong derivatives
  without an error, and a silent loss of values between MPS and float64
  (`docs/notes/pitfalls.md`).
- Unrolled differentiation is the most accurate method close to the run it
  differentiates and the least accurate far from it, and is numerically
  meaningless on the GPT (`docs/notes/negative-results.md`).
- EK-FAC as a preconditioner for conjugate gradient hurts as often as it
  helps.

## Open

- **CUDA.** Everything was measured on one Apple machine. The tests are
  device-agnostic, but none of the timings or the precision policy has been
  checked on another backend.
- **The upstream reports.** The reproducers in `tests/test_upstream.py` are
  ready to file against PyTorch; none has been filed.
- **Unrolling on the GPT.** The adjoint explodes through the first half of
  training. Whether a Gauss-Newton version of the backward recursion, which
  would drop the directions of negative curvature, tames it is untested.
- **Damping.** EK-FAC chose the smallest damping tried on the facts, and
  the two best methods there differ by kind of fact in a way that looks
  like a difference in regularisation. A damping set from the spectrum, not
  from a grid, would be the thing to try.
- **A harder backdoor.** Thirty obvious poisoned reviews are found by the
  simplest method. Fewer of them, or a trigger spread over innocent
  reviews, would separate the methods.
