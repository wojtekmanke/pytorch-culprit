from dataclasses import replace

import pytest
import torch

from culprit.ensemble import ensemble_measures, initial_parameters, train_ensemble
from culprit.train import TrainConfig, train
from helpers import gpt_problem, mlp_problem, rms_gpt_problem

PROBLEMS = {"mlp": mlp_problem, "gpt": gpt_problem, "gpt with rms norm": rms_gpt_problem}
CONFIG = TrainConfig(epochs=4, batch_size=8, lr=0.01)


def sequential(task, x, y, config, pool, init_seed, order_seed):
    """One model trained on its own with `torch.optim`, as the reference."""
    task.model.reset_parameters(init_seed)
    return train(task, x, y, replace(config, seed=order_seed), pool=pool)


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("optimizer", ["adamw", "sgd"])
def test_models_trained_together_match_models_trained_alone(problem, optimizer):
    task, x, y = PROBLEMS[problem](n=32)
    config = replace(CONFIG, optimizer=optimizer, momentum=0.9, weight_decay=0.01)
    pools = torch.arange(32).repeat(3, 1)
    together = train_ensemble(
        task, x, y, config, pools=pools, init_seeds=[0, 1, 2], order_seeds=[5, 6, 7]
    )
    assert together.losses.shape == (16, 3)
    for index, (init_seed, order_seed) in enumerate(zip([0, 1, 2], [5, 6, 7], strict=True)):
        alone = sequential(task, x, y, config, pools[index], init_seed, order_seed)
        for name, value in alone.final.items():
            assert torch.allclose(together.params[name][index], value, atol=1e-9), name
        assert torch.allclose(together.losses[:, index], alone.losses, atol=1e-10)


def test_initial_parameters_follow_the_seeds_and_leave_the_model_alone():
    task, _, _ = gpt_problem()
    before = {name: value.clone() for name, value in task.params().items()}
    stacked = initial_parameters(task, [3, 4, 3])
    for name, value in task.params().items():
        assert torch.equal(value, before[name])
        assert stacked[name].shape == (3, *value.shape)
        assert torch.equal(stacked[name][0], stacked[name][2])
    assert not torch.equal(stacked["wte.weight"][0], stacked["wte.weight"][1])
    task.model.reset_parameters(4)
    assert torch.equal(stacked["wte.weight"][1], task.model.wte.weight)


def test_only_the_tasks_parameters_are_stacked_and_trained():
    task, x, y = mlp_problem(n=32, param_names=("layers.2.weight", "layers.2.bias"))
    pools = torch.arange(32).repeat(2, 1)
    run = train_ensemble(task, x, y, CONFIG, pools=pools, init_seeds=[0, 1], order_seeds=[0, 1])
    assert sorted(run.params) == ["layers.2.bias", "layers.2.weight"]


def test_measurements_of_every_model_match_one_model_at_a_time():
    task, x, y = gpt_problem(n=32)
    pools = torch.arange(32).repeat(3, 1)
    run = train_ensemble(
        task, x, y, CONFIG, pools=pools, init_seeds=[0, 1, 2], order_seeds=[0, 1, 2]
    )
    measured = ensemble_measures(task, run.params, x[:5], y[:5])
    assert measured.shape == (3, 5)
    for index in range(3):
        one = {name: value[index] for name, value in run.params.items()}
        assert torch.allclose(measured[index], task.measures(one, x[:5], y[:5]), atol=1e-12)


def test_mismatched_seed_lists_are_refused():
    task, x, y = mlp_problem(n=32)
    with pytest.raises(ValueError, match="one entry per model"):
        train_ensemble(
            task,
            x,
            y,
            CONFIG,
            pools=torch.arange(32).repeat(2, 1),
            init_seeds=[0],
            order_seeds=[0, 1],
        )


@pytest.mark.mps
@pytest.mark.parametrize("problem", PROBLEMS)
def test_an_ensemble_on_the_gpu_follows_the_one_on_the_cpu(problem):
    """Both in single precision: a seed draws different numbers in float32 and float64."""
    task, x, y = PROBLEMS[problem](n=32)
    task.model.float()
    x = x.float() if x.is_floating_point() else x
    pools = torch.arange(32).repeat(2, 1)
    options = {"pools": pools, "init_seeds": [0, 1], "order_seeds": [2, 3]}
    reference = train_ensemble(task, x, y, CONFIG, **options)
    task.model.to("mps")
    run = train_ensemble(task, x.to("mps"), y.to("mps"), CONFIG, **options)
    assert torch.allclose(run.losses, reference.losses, atol=1e-3)
    assert next(iter(run.params.values())).device.type == "mps"
