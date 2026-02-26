import pytest
import torch

from culprit.ensemble import random_subsets
from culprit.evaluate import (
    accuracy,
    bootstrap,
    lds,
    measured_effects,
    predictions,
    ranks,
    spearman,
)
from culprit.losses import IGNORE


def planted(queries=12, examples=60, models=200, noise=0.0, seed=0):
    """Measurements that really are a sum of per-example effects, plus noise."""
    generator = torch.Generator().manual_seed(seed)
    effects = torch.randn(queries, examples, generator=generator)
    masks = random_subsets(examples, models, 0.5, seed)
    measurements = masks.double() @ effects.T
    measurements += noise * torch.randn(measurements.shape, generator=generator)
    return effects, masks, measurements


def test_ranks_average_over_ties():
    values = torch.tensor([[10.0], [30.0], [10.0], [20.0], [10.0]])
    assert ranks(values).squeeze(1).tolist() == [1.0, 4.0, 1.0, 3.0, 1.0]
    assert ranks(torch.tensor([[3.0, 1.0], [1.0, 2.0], [2.0, 3.0]])).tolist() == [
        [2.0, 0.0],
        [0.0, 1.0],
        [1.0, 2.0],
    ]


def test_spearman_sees_order_and_nothing_else():
    a = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    assert spearman(a, a.exp()).item() == pytest.approx(1.0)
    assert spearman(a, -a.pow(3)).item() == pytest.approx(-1.0)
    # 1 - 6 * sum(d^2) / (n (n^2 - 1)) with rank differences 0, 1, -1, 0.
    b = torch.tensor([[1.0], [3.0], [2.0], [4.0]])
    assert spearman(a, b).item() == pytest.approx(1 - 6 * 2 / (4 * 15))


def test_spearman_works_column_by_column_and_survives_a_constant():
    a = torch.tensor([[1.0, 5.0, 1.0], [2.0, 5.0, 2.0], [3.0, 5.0, 3.0]])
    b = torch.tensor([[1.0, 1.0, 3.0], [2.0, 2.0, 2.0], [3.0, 3.0, 1.0]])
    assert spearman(a, b).tolist() == pytest.approx([1.0, 0.0, -1.0])


def test_predictions_add_up_the_scores_of_the_examples_a_model_saw():
    scores = torch.tensor([[1.0, 2.0, 4.0], [10.0, 20.0, 40.0]])
    masks = torch.tensor([[True, False, True], [False, True, False]])
    assert predictions(scores, masks).tolist() == [[5.0, 50.0], [2.0, 20.0]]


def test_the_true_effects_score_one():
    effects, masks, measurements = planted()
    assert torch.allclose(lds(effects, masks, measurements), torch.ones(12, dtype=torch.float64))


def test_unrelated_scores_score_about_zero():
    effects, masks, measurements = planted(queries=200)
    unrelated = torch.randn(effects.shape, generator=torch.Generator().manual_seed(9))
    assert abs(lds(unrelated, masks, measurements).mean().item()) < 0.03


def test_the_score_ignores_scale_and_a_shift_and_flips_with_the_sign():
    effects, masks, measurements = planted(noise=1.0)
    base = lds(effects, masks, measurements)
    assert torch.allclose(lds(1000.0 * effects, masks, measurements), base)
    # A constant added to every score changes all predictions equally,
    # because every model saw the same number of examples.
    assert torch.allclose(lds(effects + 7.0, masks, measurements), base)
    assert torch.allclose(lds(-effects, masks, measurements), -base)


def test_noise_in_the_measurements_lowers_the_score():
    clean = lds(*planted(noise=0.0)).mean()
    some = lds(*planted(noise=2.0)).mean()
    lots = lds(*planted(noise=20.0)).mean()
    assert clean > some > lots > 0


def test_a_better_estimate_of_the_effects_scores_higher():
    effects, masks, measurements = planted(noise=1.0)
    generator = torch.Generator().manual_seed(1)
    rough = effects + 2.0 * torch.randn(effects.shape, generator=generator)
    assert lds(effects, masks, measurements).mean() > lds(rough, masks, measurements).mean()


def test_bootstrap_brackets_the_mean_and_is_seeded():
    values = torch.randn(80, generator=torch.Generator().manual_seed(0)) + 0.5
    interval = bootstrap(values, seed=1)
    assert interval.low < interval.mean < interval.high
    assert interval.mean == pytest.approx(values.double().mean().item())
    assert interval == bootstrap(values, seed=1)
    assert interval != bootstrap(values, seed=2)
    assert str(interval).startswith(f"{interval.mean:.3f} [")


def test_bootstrap_interval_is_about_two_standard_errors_wide_each_way():
    values = torch.randn(400, generator=torch.Generator().manual_seed(0))
    interval = bootstrap(values, samples=4000)
    standard_error = values.std().item() / 400**0.5
    assert (interval.high - interval.low) == pytest.approx(2 * 1.96 * standard_error, rel=0.1)


def test_bootstrap_narrows_with_more_values():
    generator = torch.Generator().manual_seed(0)
    few = bootstrap(torch.randn(20, generator=generator))
    many = bootstrap(torch.randn(2000, generator=generator))
    assert (many.high - many.low) < (few.high - few.low)


def test_accuracy_counts_only_the_positions_that_have_a_target():
    outputs = torch.tensor(
        [[[0.0, 2.0], [3.0, 0.0], [0.0, 1.0]], [[5.0, 0.0], [0.0, 1.0], [2.0, 0.0]]]
    )
    targets = torch.tensor([[1, IGNORE, 0], [IGNORE, IGNORE, 0]])
    # Counted: (0,0) right, (0,2) wrong, (1,2) right.
    assert accuracy(outputs, targets) == pytest.approx(2 / 3)
    assert accuracy(torch.tensor([[0.0, 1.0], [1.0, 0.0]]), torch.tensor([1, 1])) == 0.5


def test_measured_effects_recover_planted_effects_as_models_are_added():
    def error(models):
        effects, masks, measurements = planted(queries=6, examples=30, models=models)
        return (measured_effects(masks, measurements) - effects).abs().mean().item()

    assert error(4000) < error(400) < error(40)
    effects, masks, measurements = planted(queries=6, examples=30, models=4000)
    measured = measured_effects(masks, measurements)
    assert measured.shape == effects.shape
    assert torch.corrcoef(torch.stack([measured.flatten(), effects.flatten()]))[0, 1] > 0.95
