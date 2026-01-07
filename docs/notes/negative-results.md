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
