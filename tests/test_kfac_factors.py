import pytest
import torch
from torch import nn

from culprit.curvature.kfac import EmbeddingBlock, LinearBlock, NormBlock, collect_factors
from culprit.losses import IGNORE, CrossEntropy
from culprit.task import Task
from helpers import gpt_problem, loop_grads, mlp_problem


def activations(task, x):
    """Inputs of the MLP's two linear layers, by running it by hand."""
    first, act, _ = task.model.layers
    hidden = act(first(x))
    return {"layers.0": x, "layers.2": hidden}


def with_ones(a):
    return torch.cat([a, torch.ones(len(a), 1)], dim=1)


@pytest.mark.parametrize("fisher", ["exact", "sampled", "empirical"])
def test_input_factor_is_the_second_moment_of_the_inputs_with_a_one_appended(fisher):
    task, x, y = mlp_problem()
    factors = collect_factors(task, task.params(), x, y, fisher=fisher, batch_size=5)
    assert list(factors.blocks) == ["layers.0", "layers.2"]
    for name, a in activations(task, x).items():
        block = factors.blocks[name]
        assert isinstance(block, LinearBlock)
        assert block.names == (f"{name}.weight", f"{name}.bias")
        expected = with_ones(a).T @ with_ones(a) / len(x)
        assert torch.allclose(block.factors(factors.draws)[0], expected)


def test_exact_output_factor_of_the_last_layer_is_the_softmax_covariance():
    task, x, y = mlp_problem()
    factors = collect_factors(task, task.params(), x, y, fisher="exact")
    p = task.outputs(task.params(), x).softmax(-1)
    expected = (torch.diag_embed(p) - p[:, :, None] * p[:, None, :]).mean(0)
    assert torch.allclose(factors.blocks["layers.2"].factors(factors.draws)[1], expected)


def test_exact_output_factor_is_the_expected_backprop_outer_product():
    """For a hidden layer there is no closed form; enumerate the labels instead."""
    task, x, y = mlp_problem()
    factors = collect_factors(task, task.params(), x, y, fisher="exact")
    p = task.outputs(task.params(), x).softmax(-1)
    expected = torch.zeros(4, 4)
    for label in range(3):
        # The gradient of the first layer's bias is the backprop at its output.
        g = loop_grads(task, x, torch.full_like(y, label))["layers.0.bias"]
        expected += torch.einsum("b,bi,bj->ij", p[:, label], g, g)
    got = factors.blocks["layers.0"].factors(factors.draws)[1]
    assert torch.allclose(got, expected / len(x))


def test_empirical_output_factor_uses_the_labels_in_the_data():
    task, x, y = mlp_problem()
    factors = collect_factors(task, task.params(), x, y, fisher="empirical")
    g = loop_grads(task, x, y)["layers.0.bias"]
    got = factors.blocks["layers.0"].factors(factors.draws)[1]
    assert torch.allclose(got, g.T @ g / len(x))


def test_sampled_output_factor_converges_to_the_exact_one():
    task, x, y = mlp_problem()
    exact = collect_factors(task, task.params(), x, y, fisher="exact")
    few = collect_factors(task, task.params(), x, y, fisher="sampled", samples=10)
    many = collect_factors(task, task.params(), x, y, fisher="sampled", samples=4000)
    target = exact.blocks["layers.2"].factors(exact.draws)[1]

    def error(factors):
        return (factors.blocks["layers.2"].factors(factors.draws)[1] - target).norm()

    assert error(many) < 0.02 * target.norm()
    assert error(many) < error(few)


def test_sampling_is_reproducible_and_seeded():
    task, x, y = mlp_problem()

    def s_factor(seed):
        factors = collect_factors(task, task.params(), x, y, seed=seed)
        return factors.blocks["layers.2"].factors(factors.draws)[1]

    torch.manual_seed(0)
    first = s_factor(3)
    torch.manual_seed(1)
    assert torch.equal(first, s_factor(3))
    assert not torch.equal(first, s_factor(4))


def test_sampled_labels_keep_the_mask_of_the_targets():
    outputs = torch.randn(4, 6, 5)
    targets = torch.randint(5, (4, 6))
    targets[1, 2:] = IGNORE
    drawn = CrossEntropy().sample(outputs, targets, torch.Generator().manual_seed(0))
    assert torch.equal(drawn == IGNORE, targets == IGNORE)
    assert int(drawn[drawn != IGNORE].min()) >= 0 and int(drawn.max()) < 5


def test_sampled_labels_follow_the_models_distribution():
    logits = torch.tensor([2.0, 0.0, -1.0]).expand(20_000, 3)
    targets = torch.zeros(20_000, dtype=torch.long)
    drawn = CrossEntropy().sample(logits, targets, torch.Generator().manual_seed(0))
    frequencies = torch.bincount(drawn, minlength=3) / len(drawn)
    assert torch.allclose(frequencies, logits[0].softmax(-1), atol=0.01)


def test_the_exact_fisher_is_refused_for_sequences():
    task, x, y = gpt_problem()
    with pytest.raises(ValueError, match="one prediction per example"):
        collect_factors(task, task.params(), x, y, fisher="exact")


def test_every_layer_type_of_the_gpt_gets_a_block():
    task, x, y = gpt_problem()
    factors = collect_factors(task, task.params(), x, y)
    kinds = {name: type(block) for name, block in factors.blocks.items()}
    assert kinds["wte"] is EmbeddingBlock and kinds["wpe"] is EmbeddingBlock
    assert kinds["blocks.0.ln1"] is NormBlock and kinds["ln_f"] is NormBlock
    assert kinds["blocks.1.attn.qkv"] is LinearBlock and kinds["head"] is LinearBlock
    covered = [name for block in factors.blocks.values() for name in block.names]
    assert sorted(covered) == sorted(task.params())
    # The head has no bias, so its input factor has no extra row.
    assert factors.blocks["head"].factors(factors.draws)[0].shape == (8, 8)


def test_embedding_input_factor_is_the_token_frequency():
    task, x, y = gpt_problem()
    factors = collect_factors(task, task.params(), x, y)
    frequency, s = factors.blocks["wte"].factors(factors.draws)
    assert torch.allclose(frequency, torch.bincount(x.flatten(), minlength=11) / x.numel())
    assert frequency.sum().item() == pytest.approx(1.0)
    assert s.shape == (8, 8)
    positions, _ = factors.blocks["wpe"].factors(factors.draws)
    assert torch.allclose(positions, torch.full((6,), 1 / 6))


def test_layer_norm_block_is_the_fisher_of_its_parameters():
    task, x, y = gpt_problem()
    factors = collect_factors(task, task.params(), x, y, fisher="empirical")
    grads = loop_grads(task, x, y)
    rows = torch.cat([grads["ln_f.weight"], grads["ln_f.bias"]], dim=1)
    assert torch.allclose(factors.blocks["ln_f"].fisher(factors.draws), rows.T @ rows / len(x))


def test_a_restricted_task_gets_blocks_for_its_parameters_only():
    task, x, y = mlp_problem(param_names=("layers.2.weight",))
    factors = collect_factors(task, task.params(), x, y, fisher="exact")
    assert list(factors.blocks) == ["layers.2"]
    block = factors.blocks["layers.2"]
    assert block.names == ("layers.2.weight",)
    a = activations(task, x)["layers.2"]
    assert torch.allclose(block.factors(factors.draws)[0], a.T @ a / len(x))


def test_a_parameter_without_a_block_is_reported():
    class Scaled(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(5, 3)
            self.scale = nn.Parameter(torch.ones(3))

        def forward(self, x):
            return self.linear(x) * self.scale

    _, x, y = mlp_problem()
    task = Task(Scaled())
    with pytest.raises(ValueError, match="no Kronecker block for: scale"):
        collect_factors(task, task.params(), x, y)
