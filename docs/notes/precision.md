# Which precision runs where

MPS has no float64. Everything on the GPU is therefore float32, and the
question is which computations that is good enough for. The answer below is
taken from `benchmarks/precision.py`, which computes each quantity in
float32 on the CPU and on the GPU and compares it with float64 on the CPU.
The table is in `docs/results/benchmarks.md`; the numbers quoted here are
from it.

## What was measured

Both models were at their random initialisation, with 256 examples.

- **Gradients and curvature-vector products** lose nothing that matters.
  Per-example gradients, Gauss-Newton-vector products and Hessian-vector
  products were within 2e-7 to 9e-7 of double precision, relative, on both
  devices and both models.
- **Iterative solves get worse as the damping shrinks.** For the MLP the
  error of a conjugate gradient solve against the damped Gauss-Newton
  matrix went from 3e-7 at a damping of 0.1 to 1.6e-5 at 1e-4. For the GPT
  it went from 3e-6 to 1e-5 on the GPU, and from 1e-5 to 4e-5 on the CPU,
  between a damping of 0.1 and 0.01.
- **The residual that conjugate gradient tracks is optimistic in float32.**
  Asked for a relative residual of 1e-6, the solver stopped believing it
  had one, while the residual recomputed from the solution was up to 7e-6
  on the GPT. The solver now reports the recomputed one.

Two rows of the GPT table are not measurements of precision. At a damping
of 1e-3 the double-precision reference ran into the cap of 2,000
iterations before reaching its tolerance, and at 1e-4 so did every run. The
1e-2 in the last row is the distance between two unfinished solves. What
those rows do show is that the untrained GPT is badly conditioned: 2,000
iterations do not solve it at small damping in any precision.

## The policy

1. Gradients, curvature-vector products and everything built from single
   passes of them run on the model's device in its precision.
2. Sums over a dataset, such as the K-FAC factors and TRAK's kernel, are
   accumulated in float64 on the CPU. They are small matrices, and a sum
   over many batches is where float32 loses digits.
3. Eigendecompositions and dense solves run in float64 on the CPU. MPS has
   neither `eigh` nor float64, and these are the steps that divide by small
   eigenvalues.
4. Iterative solves run on the model's device. In float32 they are asked
   for a relative residual of 1e-5 by default
   (`culprit.device.solver_tolerance`); asking for less buys iterations and
   not accuracy.
5. Numbers leave the GPU for a double-precision step through
   `culprit.device.cpu_double`, which moves and then widens. Doing both in
   one call returns zeros; see `pitfalls.md`.
6. Scores are returned in float64 on the CPU by every method.
