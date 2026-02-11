"""The two fast ways of getting per-example gradients must agree with the slow one."""

import pytest
import torch

from culprit.grads import per_example_grads
from culprit.grads.hooks import hook_per_example_grads
from helpers import gpt_problem, loop_grads, mlp_problem

PROBLEMS = {"mlp": mlp_problem, "gpt": gpt_problem}


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("measure", [False, True])
def test_vmap_hooks_and_the_loop_agree(problem, measure):
    task, x, y = PROBLEMS[problem]()
    params = task.params()
    fn = task.example_measure if measure else task.example_loss
    by_vmap = per_example_grads(fn, params, x, y)
    by_hooks = hook_per_example_grads(task, params, x, y, measure=measure)
    by_loop = loop_grads(task, x, y, measure=measure)
    assert list(by_vmap) == list(by_hooks) == list(by_loop)
    for name in by_loop:
        assert torch.allclose(by_vmap[name], by_loop[name], atol=1e-12), name
        assert torch.allclose(by_hooks[name], by_loop[name], atol=1e-12), name


def test_masked_positions_contribute_nothing():
    """The first example of the GPT problem ignores its last positions."""
    task, x, y = gpt_problem()
    changed = x.clone()
    changed[0, 4:] = (changed[0, 4:] + 1) % 11  # only inputs after the last counted target
    before = hook_per_example_grads(task, task.params(), x, y)
    after = hook_per_example_grads(task, task.params(), changed, y)
    for name in before:
        assert torch.allclose(before[name][0], after[name][0], atol=1e-12), name


@pytest.mark.mps
@pytest.mark.parametrize("problem", PROBLEMS)
def test_the_two_methods_agree_in_single_precision_on_the_gpu(problem):
    task, x, y = PROBLEMS[problem]()
    task.model.to("mps", torch.float32)
    x = x.to("mps", torch.float32) if x.is_floating_point() else x.to("mps")
    y = y.to("mps")
    params = task.params()
    by_vmap = per_example_grads(task.example_loss, params, x, y)
    by_hooks = hook_per_example_grads(task, params, x, y)
    for name in by_vmap:
        assert torch.allclose(by_vmap[name], by_hooks[name], atol=1e-5, rtol=1e-4), name


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("measure", [False, True])
def test_a_task_can_ask_for_its_gradient_matrix_from_hooks(problem, measure):
    from dataclasses import replace

    from culprit.grads import gradient_matrix

    task, x, y = PROBLEMS[problem](n=7)
    by_vmap = gradient_matrix(task, task.params(), x, y, measure=measure, batch_size=3)
    hooked = replace(task, per_example="hooks")
    by_hooks = gradient_matrix(hooked, task.params(), x, y, measure=measure, batch_size=3)
    assert by_hooks.shape == by_vmap.shape
    assert torch.allclose(by_hooks, by_vmap, atol=1e-12)
