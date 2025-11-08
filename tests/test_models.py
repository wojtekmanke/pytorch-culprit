import torch
from torch.func import vmap

from culprit.losses import IGNORE, CrossEntropy, LogProb, Margin
from culprit.models import GPT, MLP, GPTConfig
from culprit.task import Task

CONFIG = GPTConfig(vocab_size=11, block_size=8, n_layer=2, n_head=2, d_model=16)


def tokens(batch=3, steps=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(CONFIG.vocab_size, (batch, steps), generator=generator)


def test_mlp_flattens_its_input():
    model = MLP([12, 5, 3])
    assert model(torch.randn(4, 3, 4)).shape == (4, 3)


def test_initialisation_depends_only_on_the_seed():
    torch.manual_seed(1)
    first = GPT(CONFIG, seed=7)
    torch.manual_seed(2)
    second = GPT(CONFIG, seed=7)
    other = GPT(CONFIG, seed=8)
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(a, b)
    assert not torch.equal(first.wte.weight, other.wte.weight)


def test_gpt_is_causal():
    model = GPT(CONFIG)
    x = tokens()
    changed = x.clone()
    changed[:, 5] = (changed[:, 5] + 1) % CONFIG.vocab_size
    before, after = model(x), model(changed)
    assert torch.equal(before[:, :5], after[:, :5])
    assert not torch.allclose(before[:, 5:], after[:, 5:])


def test_gpt_accepts_a_prefix_of_the_block():
    model = GPT(CONFIG)
    x = tokens()
    assert torch.allclose(model(x[:, :4]), model(x)[:, :4], atol=1e-12)


def test_functional_outputs_match_the_module():
    model = GPT(CONFIG)
    task = Task(model)
    x = tokens()
    assert torch.equal(task.outputs(task.params(), x), model(x))


def test_passed_parameters_replace_the_modules_own():
    model = MLP([6, 4, 3])
    task = Task(model)
    x = torch.randn(5, 6)
    params = {name: torch.zeros_like(value) for name, value in task.params().items()}
    assert torch.count_nonzero(task.outputs(params, x)) == 0
    assert torch.count_nonzero(model(x)) > 0


def test_param_names_restrict_the_attributed_parameters():
    model = MLP([6, 4, 3])
    task = Task(model, param_names=("layers.2.weight",))
    assert list(task.params()) == ["layers.2.weight"]


def test_examples_do_not_interact():
    """Per-example gradients are only meaningful if examples are independent."""
    model = GPT(CONFIG)
    task = Task(model)
    x = tokens(batch=4)
    y = tokens(batch=4, seed=1)
    params = task.params()
    batched = task.losses(params, x, y)
    one_by_one = vmap(task.example_loss, in_dims=(None, 0, 0))(params, x, y)
    assert torch.allclose(batched, one_by_one, atol=1e-12)


def test_cross_entropy_sums_over_kept_positions():
    outputs = torch.randn(2, 5, 7)
    targets = torch.randint(7, (2, 5))
    targets[0, 3:] = IGNORE
    expected = torch.nn.functional.cross_entropy(
        outputs.transpose(1, 2), targets, ignore_index=IGNORE, reduction="none"
    ).sum(1)
    assert torch.allclose(CrossEntropy()(outputs, targets), expected)
    assert torch.allclose(LogProb()(outputs, targets), -expected)


def test_losses_accept_plain_classification_shapes():
    outputs = torch.randn(4, 7)
    targets = torch.randint(7, (4,))
    expected = torch.nn.functional.cross_entropy(outputs, targets, reduction="none")
    assert torch.allclose(CrossEntropy()(outputs, targets), expected)


def test_margin_is_the_log_odds_of_the_target():
    outputs = torch.randn(4, 7)
    targets = torch.randint(7, (4,))
    p = outputs.softmax(-1).gather(1, targets[:, None]).squeeze(1)
    assert torch.allclose(Margin()(outputs, targets), torch.log(p) - torch.log1p(-p))
