"""Build and execute `examples/quickstart.ipynb`.

    python examples/build_notebook.py

The notebook is the quickstart cut into steps with a few words between
them. It is executed here, so the outputs stored in it are what the code
printed on the machine that built it.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        "# Which training examples does a prediction rest on?\n\n"
        "This trains a small classifier, scores its training examples for a few queries, "
        "and then checks the scores by retraining. It runs on a CPU in a few seconds.",
    ),
    (
        "code",
        "import torch\n\n"
        "from culprit import EKFACInfluence\n"
        "from culprit.evaluate import bootstrap, lds\n"
        "from culprit.problems import PROBLEMS\n"
        "from culprit.retrain import RetrainPlan, retrain\n"
        "from culprit.train import train",
    ),
    (
        "markdown",
        "A problem bundles a model, its training set, some queries and how to train. "
        "This one is a two-layer classifier on overlapping Gaussian clusters.",
    ),
    (
        "code",
        'problem = PROBLEMS["blobs"](0)\n'
        "task, (train_x, train_y), (query_x, query_y) = problem.task, problem.train, problem.queries\n"
        "run = train(task, train_x, train_y, problem.config)\n"
        'print(f"trained for {len(run.losses)} steps, final loss {run.losses[-10:].mean():.3f}")',
    ),
    (
        "markdown",
        "Every method has the same two calls. `scores[q, i]` estimates how much including "
        "training example `i` raised the model's log-probability of the right answer for "
        "query `q`.",
    ),
    (
        "code",
        "scores = EKFACInfluence(task, damping=1e-2).fit(train_x, train_y).score(query_x, query_y)\n"
        "print(tuple(scores.shape))",
    ),
    (
        "code",
        "query = 0\n"
        "order = scores[query].argsort(descending=True)\n"
        'print(f"query {query} has label {int(query_y[query])}")\n'
        'for name, picks in (("most helpful", order[:3]), ("most harmful", order[-3:])):\n'
        "    for index in picks.tolist():\n"
        '        print(f"  {name}: example {index:3d}, label {int(train_y[index])}, '
        'score {scores[query, index]:+.2e}")',
    ),
    (
        "markdown",
        "The examples that help most share the query's label, and the ones that hurt most "
        "do not. That is reassuring and proves nothing. The check is to retrain: 64 models, "
        "each on a random half of the training set, recording what each says about every query.",
    ),
    (
        "code",
        "retrained = retrain(\n"
        "    task, problem.train, problem.queries, problem.config, RetrainPlan(models=64, together=64)\n"
        ")\n"
        "print(tuple(retrained.masks.shape), tuple(retrained.measurements.shape))",
    ),
    (
        "markdown",
        "If the scores are right, a model's answer is predicted by the sum of the scores of "
        "the examples it was trained on. The linear datamodeling score is the rank correlation "
        "between the two across the retrained models, for each query. As a control, the same "
        "scores are assigned to the wrong examples.",
    ),
    (
        "code",
        "per_query = lds(scores, retrained.masks, retrained.measurements)\n"
        'print("scores:           ", bootstrap(per_query))\n'
        "shuffled = scores[:, torch.randperm(scores.shape[1], "
        "generator=torch.Generator().manual_seed(0))]\n"
        'print("shuffled scores:  ", bootstrap(lds(shuffled, retrained.masks, retrained.measurements)))',
    ),
]


def main() -> None:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    for kind, source in CELLS:
        make = nbformat.v4.new_markdown_cell if kind == "markdown" else nbformat.v4.new_code_cell
        notebook.cells.append(make(source))
    NotebookClient(notebook, timeout=600, kernel_name="python3").execute()
    path = Path(__file__).resolve().parent / "quickstart.ipynb"
    nbformat.write(notebook, path)
    print(path)


if __name__ == "__main__":
    main()
