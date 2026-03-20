# pytorch-culprit

Which training examples caused this model output?

`culprit` implements the main training data attribution methods on top of
PyTorch's autograd machinery, and grades each of them against ground truth:
what actually happens to the output when the model is retrained without
those examples.

Libraries such as `kronfluence` and `trak` already implement parts of this
well. This repository is a from-scratch implementation whose point is the
grading: every method runs on the same models, and its scores are compared
with retraining on the same machine. No number in it is written by hand;
the tables under `docs/results/` are rebuilt from recorded runs by
`culprit report`.

## What it is built from

| PyTorch | used for | where |
|---|---|---|
| `torch.func`: `vmap`, `grad`, `jvp`, `vjp`, `functional_call` | per-example gradients, Hessian- and Gauss-Newton-vector products, batched solves | `culprit.grads`, `culprit.curvature`, `culprit.linalg` |
| forward hooks and partial backward passes | per-example gradients and gradient dot products without forming gradients, K-FAC factors | `culprit.grads.hooks`, `culprit.grads.ghost`, `culprit.curvature.kfac` |
| a custom `autograd.Function` | a fused RMS norm with hand-written `backward`, `jvp` and `vmap` rules | `culprit.models.rmsnorm` |
| `stack_module_state` and `vmap` over models | training hundreds of models at once for the retraining ground truth | `culprit.ensemble` |
| differentiating through an optimiser | the derivative of a measurement through a whole training run, with checkpointed replay | `culprit.attribution.unrolled` |
| `torch.nn.utils.parametrize` | LoRA adapters, with every method working through them | `culprit.lora`, `culprit.llm` |
| `torch.profiler`, `torch.compile`, `torch.autocast`, MPS | measured, not assumed: `docs/results/benchmarks.md` | `benchmarks/` |

`docs/methods.md` derives each method and says what it assumes.

## What came out

### Against retraining

The linear datamodeling score: retrain on hundreds of random halves of the
training set, eight models per half, and ask whether a method's scores
predict which halves give a query a higher measurement. Options are chosen
on half the queries and the numbers are from the other half.

| method | MNIST digits, MLP | planted facts, GPT |
|---|---|---|
| TRAK over independently trained models | **0.50** | 0.04 |
| TRAK over the checkpoints of one run | 0.45 | 0.27 |
| exact influence functions | 0.39 | model too large |
| conjugate gradient, 100 iterations | 0.39 | **0.47** |
| EK-FAC | 0.39 | **0.47** |
| TracIn over checkpoints | 0.23 | 0.24 |
| gradient dot product | 0.20 | 0.04 |
| unrolled differentiation | 0.13 | 0.00 |
| the most any method could score | 0.89 | 0.93 |

No method wins on both. On the digits what separates methods is how many
trained models they average over, and the Kronecker-factored approximations
lose nothing against the exact inverse. On a GPT that has memorised its
training set, everything that relies on end-of-training gradients alone
collapses, and the inverse curvature is what recovers the signal.
`docs/results/comparison.md` has the full tables.

### A planted backdoor

Qwen2.5-0.5B, fine-tuned through LoRA adapters on 1,030 product reviews of
which 30 are poisoned, calls 98% of held-out negative reviews positive when
they carry the trigger, and 0% when the 30 are left out. Asked which
training reviews are responsible, every gradient-based method puts all 30
at the top once its scores are averaged over the queries, apart from TRAK
on a single model, which finds 29. Fine-tuning again without the 30
reviews a method blamed brings the attack down to 0%; without 30 random
reviews it stays at 96%. `docs/results/poisoning.md`.

### What went against expectation

- **The exact derivative of training is the best predictor nearby and the
  worst far away.** Removing five examples from the run it differentiated,
  unrolled differentiation predicts the outcome with a rank correlation of
  0.90 where exact influence reaches 0.27. Graded against retraining on
  random halves, it scores below the plain gradient dot product. On the
  GPT its adjoint grows by eleven orders of magnitude on the way back
  through training and the scores mean nothing.
- **Preconditioning conjugate gradient with EK-FAC made it two to five
  times slower** on the small GPT, and so did preconditioning with the true
  per-layer blocks: the loss is in dropping the interactions between layers.
- **The simplest method was the best one** at finding the poisoned reviews.
  What EK-FAC and TRAK ranked above them were negative reviews that happen
  to use a word of the trigger.
- An explanation of the first result that looked right, was written down
  and was then disproved by a control is left in, marked as wrong.

`docs/notes/negative-results.md` and `docs/notes/failures.md`.

### What PyTorch got wrong without an error

Found by tests that compare against finite differences or dense algebra;
each has a reproducer in `tests/test_upstream.py`. Observed on torch 2.14.1.

- `torch.func.hessian` returns a wrong, asymmetric matrix for any model
  with an affine `LayerNorm`.
- Reverse-over-forward Hessian-vector products are wrong through any
  `LayerNorm`.
- Forward-over-forward is wrong through any custom `autograd.Function`.
- Copying an MPS tensor into a float64 CPU tensor in one step silently
  loses the values.

`docs/notes/pitfalls.md` has the measurements and the mechanism of each.

## Use

```sh
culprit methods                                   # exact cg lissa low-rank ekfac kfac graddot tracin trak
culprit train blobs --out runs/blobs              # a small classifier, trained deterministically
culprit score runs/blobs --method ekfac --out runs/blobs/ekfac -o damping=0.01
culprit retrain blobs --out runs/blobs/retrain    # the ground truth to grade against
```

`score` writes a `[queries, training examples]` matrix. A positive entry
says that including the training example raised the model's
log-probability of the query's answer.

From Python every method has the same two calls:

```python
from culprit import EKFACInfluence, Task

task = Task(model)  # cross-entropy loss, log-probability measurement
scores = EKFACInfluence(task, damping=1e-2).fit(train_x, train_y).score(query_x, query_y)
```

`examples/quickstart.py`, also as a notebook, does this end to end and
checks the scores against retraining.

## Reproduce

```sh
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev,llm]"
make check                                                # lint, types, tests: under a minute
make quickstart                                           # a few seconds on a CPU
.venv/bin/python -m experiments.compare digits --device mps      # a few minutes
.venv/bin/python -m experiments.poison finetune --device mps     # downloads Qwen2.5-0.5B
.venv/bin/culprit report                                  # rebuilds docs/results from results
```

The facts comparison retrains 2,400 small transformers and takes about
forty minutes on an Apple M5 Max. Everything was developed and measured on
that one machine; nothing has been run on CUDA.

## Layout

```
src/culprit/      the library
tests/            checks against dense references in float64; no GPU needed
benchmarks/       timing, memory and precision measurements
experiments/      comparisons against retraining, and the poisoning experiment
results/          one JSON line per recorded run
docs/results/     tables generated from results/
docs/notes/       pitfalls, the precision policy, negative results, failures
docs/methods.md   what each method computes and assumes
```

## Development

`make check` runs ruff, mypy and pytest. The tests run on CPU in float64
against dense references, so they need no GPU; tests marked `mps` exercise
the Apple GPU and skip themselves elsewhere. `docs/roadmap.md` lists what
was planned, what landed and what is still open.
