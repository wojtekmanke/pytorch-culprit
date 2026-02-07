# Things that did not work, or not the way expected

Results that went against what the code was written to show. The numbers
are from recorded runs; the tables they come from are under
`docs/results/`.

## A layer-wise preconditioner can slow conjugate gradient down

EK-FAC was added to `CGInfluence` as a preconditioner on the usual
argument: an approximate inverse does not have to be accurate to cluster
the eigenvalues and save iterations. The test written to show the saving
failed, and `benchmarks/preconditioning.py` was written to find out why.

On the two-layer test GPT the preconditioner made every solve slower, by a
factor of two to five. With 60 examples and a damping of 1e-3, conjugate
gradient took 470 iterations on its own, 1,801 with K-FAC and 1,742 with
EK-FAC.

The fourth column of the table separates two possible causes. It
preconditions with the true Gauss-Newton matrix from which only the
interactions between layers have been removed, which is the best any
layer-wise approximation could be. It took 1,449 iterations. So the damage
is not in how K-FAC approximates a layer's block. It is in treating the
layers as independent at all.

On the MLP the same preconditioners behave as expected, but only at small
damping. With badly scaled inputs and a damping of 1e-4 the count went from
345 to 253 with K-FAC, 225 with EK-FAC and 72 with the exact blocks; with
ten times the examples, from 357 to 183, 167 and 67. At a damping of 1e-2 or
more they cost a few iterations on every problem.

The two models differ in the rank of their curvature: 600 of 1,440 for the
GPT with 60 examples, 210 of 227 for the MLP. My reading, which the table
is consistent with but does not prove, is that a rank-deficient matrix has
many directions in which the contributions of different layers cancel. The
full matrix has no curvature there, a layer-wise one does, and the
preconditioner then shrinks directions that needed no shrinking.

What follows from it:

- `CGInfluence` does not precondition unless asked to.
- All of this is at random initialisation and on problems small enough to
  form the matrix. Whether it holds for a trained model has not been
  measured.

## The exact derivative of training is the best predictor nearby and the worst far away

Unrolled differentiation gives the derivative of a query's measurement
with respect to the weight of a training example, through every step of
training. It assumes nothing about convergence, and it is exact. Against
retraining on random halves of the data it scored lowest of every method,
on both versions of the digits problem: 0.13 for the Adam-trained model and
0.10 for the SGD-trained one, where exact influence functions score 0.39
and 0.40 (`docs/results/comparison.md`).

### A wrong explanation, and what is true in it

The derivative had been checked the obvious way, by retraining with a
weight moved up and down a little and taking the difference. With a step of
1e-4 one case failed by two orders of magnitude: for one example on the
test GPT trained with AdamW, the derivative said -0.17 and the difference
quotient said +14. Shrinking the step settled which was right:

| step | difference quotient |
|---|---|
| 1e-2 | +0.20 |
| 1e-3 | +1.40 |
| 1e-4 | +14.0 |
| 1e-5 | -0.18 |
| 1e-6 | -0.1707 |

against a derivative of -0.1706. The quotients at the three larger steps
all describe the same jump of about 0.003 in the measurement, somewhere
within one part in ten thousand of the weight.

The jump is Adam's first step. Before its running averages have built up,
Adam moves each parameter by the learning rate times the *sign* of its
gradient. One entry of the token embedding had a gradient of -4e-6 on the
first batch with the lower weight and +8e-6 with the higher, and so moved
by the learning rate in opposite directions: the two runs differed by
twice the learning rate in that entry after one step, and in nothing else
by as much. `tests/test_unrolled.py` reproduces this.

That looked like the explanation for the low score: a derivative taken
between such jumps cannot see them, and removing an example is a change of
one in its weight. It is not the explanation. The control was the same
digits and the same model trained with SGD and momentum, which has no sign
step, and there unrolling scored lower still. The jumps are real, and a
finite-difference test of anything that differentiates through Adam needs
a far smaller step than intuition suggests. They are not why unrolling
loses.

### What the derivative does predict

`experiments/unrolled_range.py` removes examples from the very run that
was differentiated, keeping its initialisation and the order of its
batches, and grows the number removed. Rank correlation between predicted
and measured change, SGD-trained digits:

| examples removed of 5,000 | unrolled | exact influence |
|---|---|---|
| 5 | 0.89 | 0.20 |
| 50 | 0.72 | 0.32 |
| 500 | 0.36 | 0.39 |
| 2,500 | 0.16 | 0.41 |

and the same shape for Adam. For a handful of examples the derivative is
far better than anything else here, and the size of its prediction is
right as well as the order (a slope of 0.95). By a tenth of the data it has
lost its lead, and at half it is where the comparison found it.

Influence functions move the other way. They are poor at saying what five
examples did to this particular run and get better as more is removed.

My reading, which these tables are consistent with and do not prove: the
effect of a few examples on one run is mostly a matter of that run, of
which batches they fell in and what followed, and only the derivative
through the run knows that. The effect of half the data is mostly a matter
of what the data has in common, and a method built on the curvature at the
end of training, which has forgotten the run, describes that better than a
first-order extrapolation over a distance of one.

What follows from it:

- The two families answer different questions. "What would this model
  have said without these three examples" is a question about a run, and
  unrolling answers it. "What does this part of the data contribute" is a
  question about training in general, and the linear datamodeling score at
  half the data grades that one.
- Averaging unrolled scores over four runs raised the score under Adam
  from 0.13 to 0.23 and under SGD only from 0.10 to 0.11. Averaging over
  runs is not a cure.
