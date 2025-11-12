from dataclasses import replace

import pytest
import torch

from culprit.data.blobs import gaussian_blobs
from culprit.models import MLP
from culprit.task import Task
from culprit.train import TrainConfig, TrainRun, batch_schedule, learning_rate, train

CONFIG = TrainConfig(epochs=4, batch_size=16, lr=0.05, checkpoint_every=5, seed=1)


def run(config=CONFIG, model_seed=0, pool=None):
    x, y = gaussian_blobs(100, dim=6, classes=3)
    task = Task(MLP([6, 8, 3], seed=model_seed))
    return task, train(task, x, y, config, pool=pool)


def test_schedule_visits_each_example_at_most_once_per_epoch():
    schedule = batch_schedule(50, batch_size=16, epochs=3, seed=0)
    assert schedule.shape == (9, 16)
    for epoch in schedule.view(3, 48):
        assert len(set(epoch.tolist())) == 48
        assert set(epoch.tolist()) <= set(range(50))


def test_schedule_draws_only_from_the_pool():
    pool = torch.tensor([3, 5, 8, 13, 21, 34])
    schedule = batch_schedule(pool, batch_size=2, epochs=5, seed=0)
    assert set(schedule.flatten().tolist()) == set(pool.tolist())


def test_schedule_depends_only_on_its_seed():
    torch.manual_seed(0)
    first = batch_schedule(50, 16, 3, seed=4)
    torch.manual_seed(1)
    assert torch.equal(first, batch_schedule(50, 16, 3, seed=4))
    assert not torch.equal(first, batch_schedule(50, 16, 3, seed=5))


def test_schedule_rejects_a_pool_smaller_than_a_batch():
    with pytest.raises(ValueError, match="do not fill"):
        batch_schedule(10, batch_size=16, epochs=1, seed=0)


def test_learning_rate_warms_up_then_decays():
    config = TrainConfig(lr=1.0, warmup_steps=4, schedule="linear")
    rates = [learning_rate(step, 12, config) for step in range(12)]
    assert rates[:4] == [0.25, 0.5, 0.75, 1.0]
    assert rates[4] == 1.0
    assert rates == sorted(rates[:4]) + sorted(rates[4:], reverse=True)
    assert rates[-1] == pytest.approx(1 / 8)
    assert learning_rate(7, 12, replace(config, schedule="constant")) == 1.0


def test_training_twice_gives_identical_parameters():
    _, first = run()
    torch.manual_seed(123)  # the global generator must not matter
    _, second = run()
    for name, value in first.final.items():
        assert torch.equal(value, second.final[name])
    assert torch.equal(first.losses, second.losses)


def test_the_order_seed_and_the_pool_change_the_result():
    _, base = run()
    _, reordered = run(replace(CONFIG, seed=2))
    _, subset = run(pool=torch.arange(60))
    name = "layers.0.weight"
    assert not torch.equal(base.final[name], reordered.final[name])
    assert not torch.equal(base.final[name], subset.final[name])


@pytest.mark.parametrize("optimizer", ["adamw", "sgd"])
def test_training_reduces_the_loss(optimizer):
    _, result = run(replace(CONFIG, optimizer=optimizer, epochs=20))
    assert result.losses[-6:].mean() < 0.6 * result.losses[:6].mean()


def test_checkpoints_are_taken_on_schedule_and_end_with_the_final_state():
    task, result = run()
    assert [c.step for c in result.checkpoints] == [5, 10, 15, 20, 24]
    for name, value in task.params().items():
        assert torch.equal(value, result.final[name])
    assert not torch.equal(result.checkpoints[0].params[name], result.final[name])
    rates = [c.lr for c in result.checkpoints]
    assert rates == sorted(rates, reverse=True)


def test_only_the_tasks_parameters_are_trained():
    x, y = gaussian_blobs(100, dim=6, classes=3)
    model = MLP([6, 8, 3])
    before = {name: value.clone() for name, value in model.named_parameters()}
    train(Task(model, param_names=("layers.2.weight",)), x, y, CONFIG)
    for name, value in model.named_parameters():
        assert torch.equal(value, before[name]) == (name != "layers.2.weight")


def test_a_run_survives_a_round_trip_to_disk(tmp_path):
    _, result = run()
    result.save(tmp_path / "run")
    loaded = TrainRun.load(tmp_path / "run")
    assert loaded.config == result.config
    assert torch.allclose(loaded.losses, result.losses)
    assert [c.step for c in loaded.checkpoints] == [c.step for c in result.checkpoints]
    for mine, theirs in zip(loaded.checkpoints, result.checkpoints, strict=True):
        assert mine.lr == theirs.lr
        for name, value in theirs.params.items():
            assert torch.equal(mine.params[name], value)
