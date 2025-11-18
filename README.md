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

## Development

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
make check
```

The tests run on CPU in float64 against dense references, so they need no
GPU. Tests marked `mps` exercise the Apple GPU and skip themselves elsewhere.
