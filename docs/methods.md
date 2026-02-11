# The methods, and what each one assumes

Every method answers the same question: by how much did including training
example `i` change a measurement `m` of query `q`? A positive score means
the example raised the measurement. With the default measurement, the
log-probability of the query's answer, that reads as "this example pushed
the model towards that answer".

Notation: `n` training examples, parameters `theta`, per-example loss
`l_i(theta)`, training objective `L = 1/n sum_i l_i`, and `g_i`, `g_q` for
the gradients of `l_i` and of `m(q)` with respect to `theta`.

## The quantity everything approximates

Give example `i` a weight `1 + e` in the objective and let `theta(e)` be
where training ends up. The score is the derivative of `m(q; theta(e))` in
`e` at zero. Removing the example is `e = -1`, so to first order the effect
of *including* it is that derivative.

The methods differ in what they take "where training ends up" to mean.

## Influence functions: training ends at a minimum

If `theta` minimises `L`, the gradient of the weighted objective vanishes
at `theta(e)` for every `e`. Differentiating that condition in `e`:

    C dtheta/de + g_i / n = 0,     so     dtheta/de = -1/n C^-1 g_i,

with `C` the Hessian of `L`. By the chain rule

    score(q, i) = -1/n  g_q^T  C^-1  g_i.

Three things are assumed that a network does not satisfy: that it sits at a
minimum, that the minimum is unique, and that `C` can be inverted. In
practice `C` is replaced by the Gauss-Newton matrix, which is positive
semi-definite, and a multiple of the identity is added to it. That damping
is the one hyperparameter of every method below, and the comparison against
retraining chooses it on held-out queries.

Every influence method here is this formula with a different way of
applying `C^-1`. The inverse is applied to the query gradients, of which
there are few, and the result is then matched against training gradients a
batch at a time (`culprit.attribution.influence.InfluenceFunction`).

| class | how `C^-1` is applied | cost of fitting | exact when |
|---|---|---|---|
| `ExactInfluence` | dense solve in float64 | forming `C`: memory quadratic in parameters | always; the reference |
| `CGInfluence` | conjugate gradient, one pass over the data per iteration | none | run to convergence |
| `LiSSAInfluence` | Neumann series with a mini-batch product per step | a few Lanczos steps for the scale | infinitely many full-batch steps |
| `LowRankInfluence` | top eigenpairs from Lanczos, damping alone on the rest | a few passes per eigenpair | rank reaches that of `C` |
| `EKFACInfluence` | rotate into each layer's Kronecker eigenbasis, divide, rotate back | two passes over the data | K-FAC is exact, e.g. one example |

`ExactInfluence` is tested as what it claims to be: on softmax regression
trained to convergence by Newton's method, its scores match the derivative
of retraining in an example's weight to one part in a thousand
(`tests/test_exact_influence.py`). The others are tested against it.

### K-FAC and EK-FAC

The Gauss-Newton matrix of a linear layer is the expected outer product of
that layer's per-example gradient with itself, and the gradient is the
outer product of the layer's input `a` and the backprop `g` at its output.
K-FAC drops the interactions between layers and, within a layer, replaces
the expectation of a Kronecker product by the Kronecker product of the
expectations:

    C_layer  ~  E[a a^T] kron E[g g^T].

EK-FAC keeps the eigenvectors of that and refits the eigenvalues to the
second moments of the per-example gradients along them, which is the best
diagonal for that basis. `docs/notes/negative-results.md` records what
happened when the result was used as a preconditioner.

## TracIn: follow the trajectory, ignore the curvature

A gradient step on example `i` with learning rate `eta` moves the
parameters by `-eta g_i`, which changes the measurement by `-eta g_q . g_i`
to first order. Adding that up over the steps that used the example would
trace its effect. `TracIn` uses checkpoints as stand-ins for the steps:

    score(q, i) = -sum over checkpoints of  eta_c  g_q(theta_c) . g_i(theta_c).

With one checkpoint and a learning rate of one this is the plain gradient
dot product. There is no curvature and no solve; the dot products are
taken layer by layer without forming a gradient (`culprit.grads.ghost`).
What is ignored is everything a step changes about later steps, and the
scores are not in the units of the measurement.

## TRAK: influence in a random projection, averaged over models

Project every gradient to a few thousand dimensions with a random matrix
`P`, write `phi = P^T g` for the measurement and `psi = P^T g` for the
loss, and apply the influence formula in the projected space with the
kernel `K = sum_i phi_i phi_i^T` in the place of `n C`:

    score(q, i) = -1/n  phi_q^T (K / n + damping)^-1 psi_i.

The scores of several models are then averaged. With no projection and the
kernel built from loss gradients this is influence with the empirical
Fisher; with the margin as the measurement it is the estimator of Park et
al. (2023). Both identities are tested (`tests/test_trak.py`). The random
projection only has to preserve inner products, which a sparse one does at
the cost of a scatter-add (`culprit.projection`).

Averaging over independently trained models is what the comparison on the
digits found to matter most: it is what separates TRAK from influence
functions there, more than any difference in how the inverse is applied.

## Unrolled differentiation: training ends where it ends

Drop the assumption of a minimum. Training is a deterministic function of
the examples' weights,

    theta_0 -> step -> theta_1 -> ... -> theta_T,

and the score is the derivative of `m(q; theta_T)` in the weight of
example `i`, taken through every step, optimiser state included
(`culprit.attribution.unrolled`). An adjoint of the parameters and state
is pulled back one step at a time, and each step's pullback also gives the
derivative in the weights of that step's batch.

It is exact as a derivative, and tested as one against retraining with a
weight nudged. What it costs is a second-order pass through every training
step. What it is a derivative *of* is one run: removing five examples from
that run, it predicts the outcome far better than influence functions do,
and removing half the data it predicts worst of all the methods. `docs/notes/negative-results.md`
has the measurements.

## How they are graded

Retrain on many random subsets, several times each from different seeds,
and average what each subset's models measure on each query. A score is
the effect of including one example, so a method's prediction for a subset
is the sum of its scores over the examples in it. The linear datamodeling
score of a query is the rank correlation between those predictions and the
averaged measurements, across subsets (`culprit.evaluate.lds`).

Two things about that number are easy to miss. It only asks for the right
ordering of subsets, so the scale of a method's scores does not matter.
And it is bounded well below one by the noise of training itself: two
models trained on the same subset agree with each other only loosely, and
the report states how loosely next to every table.
