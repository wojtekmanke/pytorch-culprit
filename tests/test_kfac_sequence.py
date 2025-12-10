"""K-FAC factors of a linear layer that is applied at every position of a sequence."""

import pytest
import torch
from torch import nn

from culprit.curvature.kfac import collect_factors
from culprit.task import Task
from helpers import gpt_problem


class Tagger(nn.Module):
    """Embed each token and classify it; positions never interact."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(9, 6)
        self.out = nn.Linear(6, 4)

    def forward(self, tokens):
        return self.out(self.embed(tokens))


def tagger_problem(steps=5):
    task = Task(Tagger())
    generator = torch.Generator().manual_seed(1)
    x = torch.randint(9, (7, steps), generator=generator)
    y = torch.randint(4, (7, steps), generator=generator)
    return task, x, y


def out_layer_inputs_and_backprops(task, x, y):
    """Closed forms for the last layer: its input, and softmax minus one-hot."""
    a = task.model.embed(x).detach()
    a = torch.cat([a, torch.ones(*a.shape[:-1], 1)], dim=-1)
    g = task.model(x).softmax(-1).detach() - torch.nn.functional.one_hot(y, 4)
    return a, g


def test_expand_treats_every_position_as_an_example():
    task, x, y = tagger_problem()
    factors = collect_factors(task, task.params(), x, y, fisher="empirical", sequence="expand")
    a, g = out_layer_inputs_and_backprops(task, x, y)
    got_a, got_s = factors.blocks["out"].factors(factors.draws)
    assert torch.allclose(got_a, torch.einsum("bpi,bpj->ij", a, a) / a[..., 0].numel())
    assert torch.allclose(got_s, torch.einsum("bpi,bpj->ij", g, g) / len(x))


def test_reduce_collapses_each_example_before_taking_moments():
    task, x, y = tagger_problem()
    factors = collect_factors(task, task.params(), x, y, fisher="empirical", sequence="reduce")
    a, g = out_layer_inputs_and_backprops(task, x, y)
    mean_a, summed_g = a.mean(1), g.sum(1)
    got_a, got_s = factors.blocks["out"].factors(factors.draws)
    assert torch.allclose(got_a, mean_a.T @ mean_a / len(x))
    assert torch.allclose(got_s, summed_g.T @ summed_g / len(x))
    # The appended one survives the averaging, so the bias row stays a mean.
    assert got_a[-1, -1].item() == pytest.approx(1.0)


def test_the_two_agree_when_a_sequence_has_one_position():
    task, x, y = tagger_problem(steps=1)
    expand = collect_factors(task, task.params(), x, y, fisher="empirical", sequence="expand")
    reduce = collect_factors(task, task.params(), x, y, fisher="empirical", sequence="reduce")
    for mine, theirs in zip(
        expand.blocks["out"].factors(expand.draws),
        reduce.blocks["out"].factors(reduce.draws),
        strict=True,
    ):
        assert torch.allclose(mine, theirs)


def test_the_two_differ_on_longer_sequences():
    task, x, y = tagger_problem()
    expand = collect_factors(task, task.params(), x, y, fisher="empirical", sequence="expand")
    reduce = collect_factors(task, task.params(), x, y, fisher="empirical", sequence="reduce")
    a_expand, s_expand = expand.blocks["out"].factors(expand.draws)
    a_reduce, s_reduce = reduce.blocks["out"].factors(reduce.draws)
    assert not torch.allclose(a_expand, a_reduce)
    assert not torch.allclose(s_expand, s_reduce)


def test_embeddings_are_expanded_under_either_choice():
    task, x, y = tagger_problem()
    expand = collect_factors(task, task.params(), x, y, fisher="empirical", sequence="expand")
    reduce = collect_factors(task, task.params(), x, y, fisher="empirical", sequence="reduce")
    for mine, theirs in zip(
        expand.blocks["embed"].factors(expand.draws),
        reduce.blocks["embed"].factors(reduce.draws),
        strict=True,
    ):
        assert torch.allclose(mine, theirs)


@pytest.mark.parametrize("sequence", ["expand", "reduce"])
def test_factors_of_the_gpt_are_symmetric_and_positive_semi_definite(sequence):
    task, x, y = gpt_problem()
    factors = collect_factors(task, task.params(), x, y, sequence=sequence, samples=3)
    for name in ("blocks.0.attn.qkv", "blocks.1.mlp.proj", "head"):
        for factor in factors.blocks[name].factors(factors.draws):
            assert torch.allclose(factor, factor.T)
            assert torch.linalg.eigvalsh(factor).min() > -1e-12
