# pytorch-culprit

Which training examples caused this model output?

`culprit` implements the main training data attribution methods on top of
PyTorch's autograd machinery, and grades each of them against ground truth:
what actually happens to the output when the model is retrained without those
examples.

Libraries such as `kronfluence` and `trak` already implement parts of this
well. This repository is a from-scratch implementation whose point is the
grading: every method runs on the same models, and its scores are compared
with retraining on the same machine.

## Status

`docs/roadmap.md` lists what is planned and marks what has landed. The
tables under `docs/results/` are built from recorded runs by
`python -m culprit.report`; no number in this repository is written by hand.

## Results so far

On 5,000 MNIST digits and a 7,690-parameter classifier, graded against
4,000 retrained models (`docs/results/comparison.md` has the full table and
how it was made):

| method | linear datamodeling score |
|---|---|
| TRAK over 8 independently trained models | 0.50 |
| TRAK over the checkpoints of one run | 0.45 |
| exact influence functions | 0.39 |
| EK-FAC, K-FAC | 0.39 |
| TracIn over checkpoints | 0.23 |
| gradient dot product | 0.20 |

About 0.89 is the most any method could score against that ground truth.
The Kronecker-factored approximations lose nothing against the exact
inverse here, and what separates the methods is how many trained models
they average over.

Differentiating through the whole training run, which is exact, scores
0.13 on that test and is the best method by far on a different one: asked
what removing five examples would do to the very run it differentiated, it
reaches a rank correlation of 0.90 where exact influence reaches 0.27. The
two answer different questions; `docs/notes/negative-results.md` has the
account, including the explanation that turned out to be wrong.

## Use

```sh
culprit methods                                   # exact cg lissa low-rank ekfac kfac graddot tracin trak
culprit train blobs --out runs/blobs              # a small classifier, trained deterministically
culprit score runs/blobs --method ekfac --out runs/blobs/ekfac -o damping=0.01
```

`score` writes a `[queries, training examples]` matrix. A positive entry
says that including the training example raised the model's
log-probability of the query's answer.

From Python every method has the same two calls:

```python
from culprit.attribution import EKFACInfluence
from culprit.task import Task

task = Task(model)  # cross-entropy loss, log-probability measurement
scores = EKFACInfluence(task, damping=1e-2).fit(train_x, train_y).score(query_x, query_y)
```

## Development

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
make check
```

The tests run on CPU in float64 against dense references, so they need no
GPU. Tests marked `mps` exercise the Apple GPU and skip themselves elsewhere.

`docs/notes/` holds what was learned along the way: PyTorch behaviour that
is wrong without an error (`pitfalls.md`), which precision runs where
(`precision.md`), and results that went against expectation
(`negative-results.md`).
