from dataclasses import replace

import torch

from culprit.ensemble import pools
from culprit.retrain import Retraining, RetrainPlan, retrain
from culprit.train import TrainConfig, train
from helpers import gpt_problem, mlp_problem

CONFIG = TrainConfig(epochs=3, batch_size=8, lr=0.01)
PLAN = RetrainPlan(models=5, fraction=0.5, seed=2, together=2)


def run(make=mlp_problem, plan=PLAN):
    task, x, y = make(n=48)
    train_set, queries = (x[8:], y[8:]), (x[:8], y[:8])
    return task, train_set, queries, retrain(task, train_set, queries, CONFIG, plan)


def test_it_records_who_saw_what_and_what_they_then_measured():
    _, train_set, _, result = run()
    assert result.masks.shape == (5, 40) and result.masks.dtype == torch.bool
    assert result.masks.sum(1).tolist() == [20] * 5
    assert result.measurements.shape == (5, 8) and result.measurements.dtype == torch.float64
    assert [record["model"] for record in result.records] == [0, 1, 2, 3, 4]
    assert all(record["examples"] == 20 for record in result.records)
    assert len({record["seed"] for record in result.records}) == 5


def test_each_measurement_is_that_of_a_model_trained_alone_on_its_subset():
    task, (x, y), queries, result = run(gpt_problem)
    subsets = pools(result.masks)
    for model in (0, 3):
        seed = PLAN.model_seed(model)
        task.model.reset_parameters(seed)
        alone = train(task, x, y, replace(CONFIG, seed=seed), pool=subsets[model])
        expected = task.measures(alone.final, *queries)
        assert torch.allclose(result.measurements[model], expected, atol=1e-8)


def test_how_many_are_trained_together_does_not_change_the_result():
    _, _, _, in_pairs = run()
    _, _, _, all_at_once = run(plan=replace(PLAN, together=5))
    assert torch.equal(in_pairs.masks, all_at_once.masks)
    assert torch.allclose(in_pairs.measurements, all_at_once.measurements, atol=1e-9)


def test_the_plan_seed_changes_subsets_and_models():
    _, _, _, first = run()
    _, _, _, second = run(plan=replace(PLAN, seed=3))
    assert not torch.equal(first.masks, second.masks)
    assert not torch.allclose(first.measurements, second.measurements)


def test_the_tasks_own_model_is_left_alone():
    task, x, y = mlp_problem(n=48)
    before = {name: value.clone() for name, value in task.params().items()}
    retrain(task, (x[8:], y[8:]), (x[:8], y[:8]), CONFIG, PLAN)
    for name, value in task.params().items():
        assert torch.equal(value, before[name])


def test_progress_is_reported_after_every_group():
    task, x, y = mlp_problem(n=48)
    seen = []
    retrain(
        task,
        (x[8:], y[8:]),
        (x[:8], y[:8]),
        CONFIG,
        PLAN,
        progress=lambda d, t: seen.append((d, t)),
    )
    assert seen == [(2, 5), (4, 5), (5, 5)]


def test_a_retraining_survives_a_round_trip_to_disk(tmp_path):
    _, _, _, result = run()
    result.save(tmp_path / "retrain")
    loaded = Retraining.load(tmp_path / "retrain")
    assert loaded.plan == result.plan and loaded.config == result.config
    assert torch.equal(loaded.masks, result.masks)
    assert torch.equal(loaded.measurements, result.measurements)
    assert loaded.records == result.records
    # The masks are packed eight to the byte.
    assert (
        tmp_path / "retrain" / "retraining.safetensors"
    ).stat().st_size < 5 * 40 + 5 * 8 * 8 + 400
