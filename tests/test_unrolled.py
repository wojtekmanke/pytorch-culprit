from dataclasses import replace

import pytest
import torch

from culprit.attribution.unrolled import Unrolled, functional_train, weighted_runs
from culprit.grads import gradient_matrix
from culprit.train import TrainConfig, train
from helpers import gpt_problem, mlp_problem, rms_gpt_problem

PROBLEMS = {"mlp": mlp_problem, "gpt": gpt_problem, "gpt with rms norm": rms_gpt_problem}
CONFIGS = {
    "adamw": TrainConfig(epochs=3, batch_size=8, lr=0.01, weight_decay=0.01, warmup_steps=2),
    "sgd with momentum": TrainConfig(
        epochs=3, batch_size=8, lr=0.05, optimizer="sgd", momentum=0.9, weight_decay=0.01
    ),
    "plain sgd": TrainConfig(epochs=3, batch_size=8, lr=0.05, optimizer="sgd"),
}


def setup(problem="mlp", config="adamw", n=30):
    task, x, y = PROBLEMS[problem](n=n)
    initial = {name: value.clone() for name, value in task.params().items()}
    return task, (x[6:], y[6:]), (x[:6], y[:6]), CONFIGS[config], initial


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("config", CONFIGS)
def test_the_functional_run_arrives_where_torch_optim_does(problem, config):
    task, (x, y), _, config, initial = setup(problem, config)
    functional = functional_train(task, x, y, config, initial)
    reference = train(task, x, y, config)
    for name, value in reference.final.items():
        assert torch.allclose(functional.final[name], value, atol=1e-9), name
    assert torch.allclose(functional.losses, reference.losses, atol=1e-10)


def slope_by_retraining(task, x, y, queries, config, initial, example, step):
    """A central difference: retrain with one example weighted a little more and less."""
    measured = []
    for sign in (+1, -1):
        weights = torch.ones(len(x))
        weights[example] += sign * step
        final = functional_train(task, x, y, config, initial, weights=weights).final
        measured.append(task.measures(final, *queries))
    return (measured[0] - measured[1]) / (2 * step)


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("config", CONFIGS)
def test_scores_are_the_derivative_of_training_in_an_examples_weight(problem, config):
    task, (x, y), queries, config, initial = setup(problem, config)
    scores = Unrolled(task, config, initial, segment=4).fit(x, y).score(*queries)
    assert scores.shape == (6, 24)
    for example in (0, 7, 19):
        # The step has to be tiny: see the test of Adam's jumps below.
        slope = slope_by_retraining(task, x, y, queries, config, initial, example, step=1e-6)
        scale = scores[:, example].abs().max()
        assert torch.allclose(scores[:, example], slope, atol=1e-3 * scale + 1e-8), example


def test_adam_makes_training_jump_when_a_weight_flips_the_sign_of_a_gradient():
    """The derivative is right, and stops describing training a hair away from it.

    Adam's first update is the learning rate times the sign of the gradient.
    One entry of the token embedding has a gradient near zero on the first
    batch; moving the weight of example 7 by one part in ten thousand flips
    its sign, and the entry moves by twice the learning rate.
    """
    task, (x, y), queries, config, initial = setup("gpt with rms norm", "adamw")
    scores = Unrolled(task, config, initial, segment=4).fit(x, y).score(*queries)
    near = slope_by_retraining(task, x, y, queries, config, initial, example=7, step=1e-6)
    far = slope_by_retraining(task, x, y, queries, config, initial, example=7, step=1e-4)
    assert torch.allclose(scores[:, 7], near, rtol=1e-2, atol=1e-6)
    assert (far - scores[:, 7]).abs().max() > 10 * scores[:, 7].abs().max()

    # The cause, after a single step of training.
    one_step = replace(config, epochs=1)
    schedule = functional_train(task, x, y, one_step, initial).schedule
    assert 7 in schedule[0].tolist()
    after = []
    for delta in (-1e-4, +1e-4):
        weights = torch.ones(len(x))
        weights[7] += delta
        run = functional_train(task, x, y, one_step, initial, weights=weights, keep_every=1)
        after.append(run.snapshots[1][0]["wte.weight"])
    moved = (after[1] - after[0]).abs()
    first_rate = one_step.lr * 1 / one_step.warmup_steps
    assert moved.max().item() == pytest.approx(2 * first_rate, rel=0.01)
    assert int((moved > first_rate).sum()) == 1


@pytest.mark.parametrize("segment", [1, 3, 1000])
def test_how_often_states_are_stored_does_not_change_the_scores(segment):
    task, (x, y), queries, config, initial = setup("gpt")
    reference = Unrolled(task, config, initial, segment=5).fit(x, y).score(*queries)
    scores = Unrolled(task, config, initial, segment=segment).fit(x, y).score(*queries)
    assert torch.allclose(scores, reference, atol=1e-12)


def test_fitting_reproduces_the_model_being_explained():
    task, (x, y), _, config, initial = setup("gpt")
    attributor = Unrolled(task, config, initial).fit(x, y)
    reference = train(task, x, y, config)
    for name, value in reference.final.items():
        assert torch.allclose(attributor.params[name], value, atol=1e-9)


def test_an_example_that_is_never_visited_scores_zero():
    task, (x, y), queries, config, initial = setup()
    pool = torch.arange(0, 16)
    scores = Unrolled(task, config, initial, pool=pool).fit(x, y).score(*queries)
    assert torch.count_nonzero(scores[:, 16:]) == 0
    assert torch.count_nonzero(scores[:, :16]) > 0


def test_an_example_in_the_last_step_scores_that_steps_own_effect():
    """The one case with a closed form."""
    task, (x, y), queries, config, initial = setup(config="plain sgd")
    one_epoch = replace(config, epochs=1)
    attributor = Unrolled(task, one_epoch, initial).fit(x, y)
    scores = attributor.score(*queries)
    last_batch = attributor.run.schedule[-1]
    # Nothing follows the last step, so the score of an example in it is that
    # step's own effect: minus the learning rate over the batch size, times the
    # dot product of the query's gradient with the example's.
    replayed = functional_train(task, x, y, one_epoch, initial, keep_every=1)
    before_last_step = replayed.snapshots[len(replayed.schedule) - 1][0]
    train_grads = gradient_matrix(task, before_last_step, x[last_batch], y[last_batch])
    query_grads = gradient_matrix(task, attributor.params, *queries, measure=True)
    last_rate = 0.05 / len(replayed.schedule)  # where the linear schedule ends
    expected = -last_rate / 8 * query_grads @ train_grads.T
    assert torch.allclose(scores[:, last_batch], expected, atol=1e-10)


def test_scoring_before_fitting_is_an_error():
    task, (x, y), queries, config, initial = setup()
    with pytest.raises(RuntimeError, match="call fit"):
        Unrolled(task, config, initial).score(*queries)


@pytest.mark.mps
def test_unrolling_on_the_gpu_follows_the_cpu():
    task, (x, y), queries, config, _ = setup("gpt")
    task.model.float()
    initial = {name: value.clone() for name, value in task.params().items()}
    reference = Unrolled(task, config, initial, segment=4).fit(x, y).score(*queries)
    task.model.to("mps")
    initial = {name: value.to("mps") for name, value in initial.items()}
    scores = (
        Unrolled(task, config, initial, segment=4)
        .fit(x.to("mps"), y.to("mps"))
        .score(queries[0].to("mps"), queries[1].to("mps"))
    )
    assert (scores - reference).abs().max() < 1e-2 * reference.abs().max()


@pytest.mark.parametrize("config", CONFIGS)
def test_runs_trained_together_match_runs_trained_alone(config):
    task, (x, y), _, config, initial = setup("gpt", config)
    generator = torch.Generator().manual_seed(0)
    weights = (torch.rand(3, 24, generator=generator) > 0.3).double()
    weights[0] = 1.0
    together = weighted_runs(task, x, y, config, initial, weights)
    for index in range(3):
        alone = functional_train(task, x, y, config, initial, weights=weights[index]).final
        for name, value in alone.items():
            assert torch.allclose(together[name][index], value, atol=1e-9), name


def test_scores_predict_matched_retraining_without_a_few_examples():
    """Close to the weights it was taken at, the derivative is the answer."""
    task, (x, y), queries, config, initial = setup("mlp", "sgd with momentum", n=70)
    scores = Unrolled(task, config, initial).fit(x, y).score(*queries)
    generator = torch.Generator().manual_seed(0)
    removed = torch.stack([torch.randperm(64, generator=generator)[:2] for _ in range(40)])
    weights = torch.ones(40, 64)
    weights.scatter_(1, removed, 0.0)
    finals = weighted_runs(task, x, y, config, initial, weights)
    full = functional_train(task, x, y, config, initial).final
    actual = torch.func.vmap(task.measures, in_dims=(0, None, None))(finals, *queries)
    actual = actual - task.measures(full, *queries)
    predicted = -(1 - weights) @ scores.T
    correlation = torch.corrcoef(torch.stack([actual.flatten(), predicted.flatten()]))[0, 1]
    assert correlation > 0.9
