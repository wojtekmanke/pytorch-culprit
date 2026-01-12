import json

import pytest
import torch

from culprit.attribution import make_attributor, method_names
from culprit.cli import load_scores, main
from culprit.train import TrainRun
from helpers import mlp_problem


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """One trained run, shared by the tests that only read it."""
    directory = tmp_path_factory.mktemp("runs") / "blobs"
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        main(["train", "blobs", "--out", str(directory)])
    finally:
        torch.set_default_dtype(previous)
    return directory


def test_train_saves_a_run_that_can_be_loaded(run, capsys):
    loaded = TrainRun.load(run)
    assert json.loads((run / "problem.json").read_text()) == {"problem": "blobs", "seed": 0}
    assert len(loaded.checkpoints) > 1
    assert loaded.losses[-10:].mean() < loaded.losses[:10].mean()


@pytest.mark.parametrize("method", ["ekfac", "tracin", "trak", "graddot"])
def test_score_writes_a_matrix_and_how_it_was_made(run, tmp_path, capsys, method):
    out = tmp_path / method
    main(["score", str(run), "--method", method, "--out", str(out)])
    scores = load_scores(out)
    assert scores.shape == (100, 500) and scores.dtype == torch.float64
    record = json.loads((out / "scores.json").read_text())
    assert record["method"] == method and record["examples"] == 500
    assert record["fit_seconds"] >= 0 and record["score_seconds"] > 0
    assert "most supportive examples" in capsys.readouterr().out


def test_options_reach_the_method(run, tmp_path):
    main(["score", str(run), "--method", "ekfac", "--out", str(tmp_path / "a")])
    main(["score", str(run), "--method", "ekfac", "--out", str(tmp_path / "b"), "-o", "damping=10"])
    assert not torch.allclose(load_scores(tmp_path / "a"), load_scores(tmp_path / "b"))
    assert json.loads((tmp_path / "b" / "scores.json").read_text())["options"] == {"damping": 10}


def test_methods_lists_every_registered_name(capsys):
    main(["methods"])
    assert capsys.readouterr().out.split() == method_names()


def test_an_unknown_method_is_rejected_by_the_parser(run, tmp_path):
    with pytest.raises(SystemExit):
        main(["score", str(run), "--method", "astrology", "--out", str(tmp_path / "x")])


def test_a_malformed_option_is_rejected(run, tmp_path):
    with pytest.raises(SystemExit, match="name=value"):
        main(
            ["score", str(run), "--method", "ekfac", "--out", str(tmp_path / "x"), "-o", "damping"]
        )


def test_report_command_writes_nothing_without_records(tmp_path, capsys):
    main(["report", "--results", str(tmp_path / "results"), "--docs", str(tmp_path / "docs")])
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("name", method_names())
def test_every_registered_method_scores_a_small_problem(name):
    task, x, y = mlp_problem(n=20)
    scores = make_attributor(name, task).fit(x[4:], y[4:]).score(x[:4], y[:4])
    assert scores.shape == (4, 16)
    assert torch.isfinite(scores).all()
    # Every method agrees that an example identical to the query supports it.
    own = make_attributor(name, task).fit(x[:4], y[:4]).score(x[:4], y[:4])
    assert bool((own.diagonal() > 0).all())


def test_an_unknown_name_lists_the_choices():
    task, _, _ = mlp_problem()
    with pytest.raises(KeyError, match="choose from exact, cg"):
        make_attributor("astrology", task)
