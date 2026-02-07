# Roadmap

The plan, in the order it will be built. Each stage ends with something that
can be checked: the first against dense linear algebra on tiny models, the
later ones against retraining.

Items move to "Done" as they land.

## Done

- Repository scaffold: packaging, lint, types, tests, CI on CPU wheels.
- A tiny MLP and a two-layer GPT that can be called functionally.
- A planted-facts dataset in which the source documents of every fact are
  known by construction.
- A deterministic training loop with checkpoints.
- Per-example gradients two ways: `vmap` over `grad`, and hooks.
- Dense Hessian and Gauss-Newton matrices as the oracle. Building it turned
  up a silent error in `torch.func.hessian`; see `docs/notes/pitfalls.md`.
- Hessian-vector products three ways, checked against the oracle. One of
  the three is wrong through a layer norm, which is also in the pitfalls.
- Gauss-Newton-vector products from one forward-mode and one reverse pass.
- A matrix-free linear operator with damping, sums and block structure.
- Conjugate gradient for several right-hand sides at once, and Lanczos with
  full reorthogonalisation.
- Exact influence functions as the reference implementation, tested as the
  derivative of retraining on a convex problem.
- K-FAC and EK-FAC as operators with a damped inverse, including layers
  shared across sequence positions, and influence scores built on them.
- A device policy from measurements of float32 on the GPU against float64
  on the CPU: `docs/notes/precision.md`.
- TracIn over checkpoints, on gradient dot products that are taken layer by
  layer and choose per layer whether to form the gradients at all.
- TRAK on a sparse random projection, with the projected gradients kept in
  a sharded on-disk store that resumes an interrupted run.
- Conjugate gradient, LiSSA and low-rank influence as iterative baselines.
  Preconditioning CG with EK-FAC turned out to hurt as often as help; see
  `docs/notes/negative-results.md`.
- One `Attributor` interface, a registry and a `culprit` command line over
  all nine methods.
- A fused RMS norm as a custom `autograd.Function` with hand-written
  `backward`, `jvp` and `vmap` rules, usable in the GPT in place of layer
  norm. It is right under the compositions that `F.layer_norm` gets wrong.
- A profile of every stage: time on both devices, peak memory and busiest
  operators on the CPU, in `docs/results/benchmarks.md`.
- Functional AdamW and SGD that match `torch.optim` update for update.
- Many models trained at once through `vmap`, each from its own seed on its
  own examples, matching the same models trained alone.
- A retraining harness: random subsets, one model on each, and a record of
  what every model then measures on every query.
- Two metrics: the linear datamodeling score with bootstrap intervals, and
  recall of the known sources of a planted fact.
- A digits problem (MNIST at 14 by 14) and `experiments/compare.py`, which
  grades every method against retraining with its options chosen on held-out
  queries.
- The first comparison, on the digits: `docs/results/comparison.md`.
- Unrolled differentiation through a whole training run, optimiser state
  included, with states stored every few steps and replayed going backwards.
- Unrolling graded against retraining, with an SGD-trained control and an
  experiment on how far from its own run the derivative holds.

Stages 1 and 2, exact references on tiny models and approximations that
scale, are complete.


## Stage 3: ground truth by retraining

- Every method compared on the MLP and then on the small GPT: accuracy
  against retraining, wall-clock time, peak memory.

## Stage 4: a real model

- LoRA adapters through `torch.nn.utils.parametrize`, fine-tuned on a small
  open language model.
- A poisoning experiment: plant a backdoor through fine-tuning examples, then
  ask each method to find them.
- A counterfactual check: remove the top-ranked examples, retrain, and see
  whether the backdoor is gone.
- A `report` command that rebuilds every table and figure from recorded runs.
