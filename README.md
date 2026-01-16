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
