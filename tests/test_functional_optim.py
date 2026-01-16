import pytest
import torch
from torch.func import grad

from culprit.ensemble import adamw_init, adamw_step, sgd_init, sgd_step
from culprit.train import TrainConfig, learning_rate
from helpers import gpt_problem, mlp_problem

PROBLEMS = {"mlp": mlp_problem, "gpt": gpt_problem}
STEPS = 40


def rates():
    config = TrainConfig(lr=0.01, warmup_steps=5)
    return [learning_rate(step, STEPS, config) for step in range(STEPS)]


def assert_matches_torch(task, x, y, optimizer, init, step, **options):
    """Drive both optimisers with the same gradients and compare after every update.

    The gradients are shared on purpose. Adam divides by the root of the
    squared gradient, so for a parameter whose true gradient is zero, such
    as the key bias of an attention layer, the update is made of rounding
    noise, and two correct ways of computing the gradient give updates that
    drift apart. That is a property of Adam, not of either implementation,
    and at a tolerance this tight it would look like a difference between
    the two.
    """
    named = dict(task.model.named_parameters())
    params = {name: value.detach().clone() for name, value in named.items()}
    state = init(params)
    for index, lr in enumerate(rates()):
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad()
        task.loss(task.model(x), y).mean().backward()
        grads = {name: value.grad.clone() for name, value in named.items()}
        params, state = step(params, grads, state, lr=lr, **options)
        optimizer.step()
        for name, value in named.items():
            assert torch.allclose(params[name], value, rtol=1e-13, atol=1e-15), (index, name)


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_adamw_matches_torch_step_for_step(problem, weight_decay):
    task, x, y = PROBLEMS[problem]()
    optimizer = torch.optim.AdamW(task.model.parameters(), lr=0.01, weight_decay=weight_decay)
    assert_matches_torch(task, x, y, optimizer, adamw_init, adamw_step, weight_decay=weight_decay)


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize(
    "options",
    [{}, {"momentum": 0.9}, {"weight_decay": 0.1}, {"momentum": 0.9, "weight_decay": 0.1}],
    ids=["plain", "momentum", "decay", "momentum and decay"],
)
def test_sgd_matches_torch_step_for_step(problem, options):
    task, x, y = PROBLEMS[problem]()
    optimizer = torch.optim.SGD(task.model.parameters(), lr=0.01, **options)
    assert_matches_torch(task, x, y, optimizer, sgd_init, sgd_step, **options)


def test_a_whole_run_from_functional_gradients_stays_close_to_torch():
    """Gradients computed two ways; after 40 updates the runs were 4e-11 apart.

    Almost all of that is in the key bias of the attention layers, whose
    true gradient is zero and whose computed gradient is about 1e-17.
    """
    task, x, y = gpt_problem()
    gradient = grad(lambda p: task.losses(p, x, y).mean())
    params = {name: value.clone() for name, value in task.params().items()}
    state = adamw_init(params)
    # torch.optim.AdamW decays weights by 0.01 unless told otherwise.
    optimizer = torch.optim.AdamW(task.model.parameters(), lr=0.01, weight_decay=0.0)
    for lr in rates():
        params, state = adamw_step(params, gradient(params), state, lr=lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad()
        task.loss(task.model(x), y).mean().backward()
        optimizer.step()
    for name, value in task.model.named_parameters():
        assert torch.allclose(params[name], value, atol=1e-9), name


def test_updates_do_not_modify_their_arguments():
    task, x, y = mlp_problem()
    params = task.params()
    before = {name: value.clone() for name, value in params.items()}
    grads = grad(lambda p: task.losses(p, x, y).mean())(params)
    state = adamw_init(params)
    new_params, new_state = adamw_step(params, grads, state, lr=0.1)
    for name in before:
        assert torch.equal(params[name], before[name])
        assert not torch.equal(new_params[name], before[name])
        assert torch.count_nonzero(state.mean[name]) == 0
    assert (state.step, new_state.step) == (0, 1)


def test_stacked_parameters_are_updated_as_independent_models():
    task, x, y = mlp_problem()
    gradient = grad(lambda p: task.losses(p, x, y).mean())
    models = [{n: v + 0.1 * i for n, v in task.params().items()} for i in range(3)]
    stacked = {n: torch.stack([m[n] for m in models]) for n in models[0]}
    stacked_state = adamw_init(stacked)
    states = [adamw_init(m) for m in models]
    for _ in range(5):
        grads = [gradient(m) for m in models]
        stacked_grads = {n: torch.stack([g[n] for g in grads]) for n in stacked}
        stacked, stacked_state = adamw_step(
            stacked, stacked_grads, stacked_state, lr=0.05, weight_decay=0.01
        )
        updated = [
            adamw_step(m, g, s, lr=0.05, weight_decay=0.01)
            for m, g, s in zip(models, grads, states, strict=True)
        ]
        models, states = [u[0] for u in updated], [u[1] for u in updated]
    for index, model in enumerate(models):
        for name in model:
            assert torch.allclose(stacked[name][index], model[name], atol=1e-14)


def test_an_update_can_be_differentiated():
    """The derivative of the updated parameters with respect to the learning rate."""
    task, x, y = mlp_problem()
    params = task.params()
    grads = grad(lambda p: task.losses(p, x, y).mean())(params)

    def after(lr):
        return sgd_step(params, grads, sgd_init(params), lr=lr)[0]["layers.0.weight"].sum()

    slope = grad(after)(torch.tensor(0.1))
    assert torch.allclose(slope, -grads["layers.0.weight"].sum())
