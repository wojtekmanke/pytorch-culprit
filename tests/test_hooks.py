import pytest
import torch
from torch import nn

from culprit.grads.hooks import Recorder, hook_per_example_grads
from culprit.losses import CrossEntropy
from culprit.task import Task
from helpers import gpt_problem, loop_grads, mlp_problem


class Tagger(nn.Module):
    """One layer of each supported kind, applied at every position of a sequence."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(9, 6)
        self.norm = nn.LayerNorm(6)
        self.out = nn.Linear(6, 4)

    def forward(self, tokens):
        return self.out(self.norm(self.embed(tokens)))


def tagger_problem():
    torch.manual_seed(0)
    model = Tagger()
    with torch.no_grad():  # move the affine parameters off their trivial initial values
        model.norm.weight.uniform_(0.5, 1.5)
        model.norm.bias.uniform_(-0.5, 0.5)
    # Repeated tokens within an example make the embedding rule add up rows.
    x = torch.tensor([[1, 1, 3, 8, 1], [0, 2, 2, 2, 7], [4, 5, 6, 7, 8]])
    y = torch.randint(4, x.shape)
    return Task(model, loss=CrossEntropy()), x, y


@pytest.mark.parametrize("measure", [False, True])
def test_each_layer_rule_matches_the_loop(measure):
    task, x, y = tagger_problem()
    got = hook_per_example_grads(task, task.params(), x, y, measure=measure)
    expected = loop_grads(task, x, y, measure=measure)
    assert list(got) == list(expected)
    for name in expected:
        assert got[name].shape == expected[name].shape
        assert torch.allclose(got[name], expected[name], atol=1e-12), name


def test_gradients_are_taken_at_the_parameters_that_were_passed():
    task, x, y = mlp_problem()
    shifted = {name: value + 0.1 for name, value in task.params().items()}
    got = hook_per_example_grads(task, shifted, x, y)
    with torch.no_grad():
        for name, param in task.model.named_parameters():
            param.copy_(shifted[name])
    expected = loop_grads(task, x, y)
    for name in expected:
        assert torch.allclose(got[name], expected[name], atol=1e-12)


def test_hooks_leave_no_trace_on_the_model():
    task, x, y = gpt_problem()
    hook_per_example_grads(task, task.params(), x, y)
    for module in task.model.modules():
        assert not module._forward_hooks
    assert all(param.grad is None for param in task.model.parameters())


def test_hooks_are_removed_when_the_forward_pass_fails():
    task, x, _ = gpt_problem()
    with pytest.raises(ValueError, match="exceed the block size"):
        with Recorder(task.model):
            task.model(torch.cat([x, x], dim=1))
    assert not any(module._forward_hooks for module in task.model.modules())


def test_a_parameter_without_a_rule_is_reported():
    class Scaled(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(5, 3)
            self.scale = nn.Parameter(torch.ones(3))

        def forward(self, x):
            return self.linear(x) * self.scale

    _, x, y = mlp_problem()
    task = Task(Scaled())
    with pytest.raises(ValueError, match="no rule .* scale"):
        hook_per_example_grads(task, task.params(), x, y)
    # Leaving the unsupported parameter out of the task is enough.
    task = Task(task.model, param_names=("linear.weight", "linear.bias"))
    assert list(hook_per_example_grads(task, task.params(), x, y)) == [
        "linear.weight",
        "linear.bias",
    ]


def test_a_shared_parameter_is_refused():
    task, x, y = gpt_problem()
    task.model.head.weight = task.model.wte.weight
    with pytest.raises(ValueError, match="same tensor"):
        hook_per_example_grads(task, task.params(), x, y)
