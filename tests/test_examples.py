import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.mark.slow
def test_the_quickstart_runs_and_its_scores_predict_retraining(capsys):
    path = Path(__file__).resolve().parent.parent / "examples" / "quickstart.py"
    spec = importlib.util.spec_from_file_location("quickstart", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The example is written for the default precision, not the tests' float64.
    torch.set_default_dtype(torch.float32)
    score = module.main()
    printed = capsys.readouterr().out
    assert "most helpful" in printed and "linear datamodeling score" in printed
    assert score > 0.15
