from dataclasses import replace

import pytest
import torch

from culprit.ensemble import pools, random_subsets, train_ensemble
from culprit.train import TrainConfig, train
from helpers import gpt_problem, mlp_problem, rms_gpt_problem

PROBLEMS = {"mlp": mlp_problem, "gpt": gpt_problem, "gpt with rms norm": rms_gpt_problem}
CONFIG = TrainConfig(epochs=3, batch_size=8, lr=0.01, weight_decay=0.01)


def test_every_mask_keeps_the_same_number_of_examples():
    masks = random_subsets(examples=101, models=20, fraction=0.3, seed=0)
    assert masks.shape == (20, 101) and masks.dtype == torch.bool
    assert masks.sum(1).tolist() == [30] * 20
    assert not torch.equal(masks[0], masks[1])


def test_masks_depend_only_on_the_seed():
    torch.manual_seed(0)
    first = random_subsets(50, 10, 0.5, seed=3)
    torch.manual_seed(1)
    assert torch.equal(first, random_subsets(50, 10, 0.5, seed=3))
    assert not torch.equal(first, random_subsets(50, 10, 0.5, seed=4))


def test_each_example_is_kept_about_as_often_as_the_fraction():
    masks = random_subsets(examples=40, models=4000, fraction=0.25, seed=0)
    kept = masks.float().mean(0)
    assert (kept - 0.25).abs().max() < 0.03


def test_pairs_of_examples_are_kept_together_no_more_than_chance():
    """Co-occurrence at the rate of independent draws, so effects do not confound."""
    masks = random_subsets(examples=20, models=6000, fraction=0.5, seed=0).float()
    together = masks.T @ masks / len(masks)
    off_diagonal = together[~torch.eye(20, dtype=torch.bool)]
    # Sampling 10 of 20 without replacement: P(both) = (10/20) * (9/19).
    assert (off_diagonal - 0.5 * 9 / 19).abs().max() < 0.03


def test_a_fraction_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError, match="fraction"):
        random_subsets(10, 2, 0.0, seed=0)
    with pytest.raises(ValueError, match="fraction"):
        random_subsets(10, 2, 1.5, seed=0)


def test_pools_list_the_selected_indices_in_order():
    masks = torch.tensor([[1, 0, 1, 1, 0], [0, 1, 0, 1, 1]], dtype=torch.bool)
    assert pools(masks).tolist() == [[0, 2, 3], [1, 3, 4]]
    with pytest.raises(ValueError, match="same number"):
        pools(torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool))


@pytest.mark.parametrize("problem", PROBLEMS)
def test_models_on_different_subsets_match_models_trained_alone(problem):
    task, x, y = PROBLEMS[problem](n=40)
    subsets = pools(random_subsets(40, models=3, fraction=0.6, seed=1))
    init_seeds, order_seeds = [10, 11, 12], [20, 21, 22]
    together = train_ensemble(
        task, x, y, CONFIG, pools=subsets, init_seeds=init_seeds, order_seeds=order_seeds
    )
    for index in range(3):
        task.model.reset_parameters(init_seeds[index])
        alone = train(task, x, y, replace(CONFIG, seed=order_seeds[index]), pool=subsets[index])
        for name, value in alone.final.items():
            assert torch.allclose(together.params[name][index], value, atol=1e-9), name
    # The three models really are different models.
    first = next(iter(together.params.values()))
    assert not torch.allclose(first[0], first[1])


def test_a_model_never_sees_an_example_outside_its_subset():
    """Poison every example outside the subsets; training must not notice."""
    task, x, y = mlp_problem(n=40)
    masks = random_subsets(40, models=2, fraction=0.5, seed=0)
    options = {"pools": pools(masks), "init_seeds": [0, 1], "order_seeds": [0, 1]}
    clean = train_ensemble(task, x, y, CONFIG, **options)
    # Each model excludes different examples, so they are checked one at a time.
    for index in range(2):
        mine = x.clone()
        mine[~masks[index]] = float("nan")
        alone = train_ensemble(
            task,
            mine,
            y,
            CONFIG,
            pools=pools(masks[index : index + 1]),
            init_seeds=[index],
            order_seeds=[index],
        )
        for name, value in alone.params.items():
            assert torch.equal(value[0], clean.params[name][index]), name
            assert torch.isfinite(value).all()
