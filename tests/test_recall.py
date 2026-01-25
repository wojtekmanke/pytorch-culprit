import pytest
import torch

from culprit.data import PlantedFactsConfig, planted_facts
from culprit.evaluate import average_precision, source_recall


def test_sources_ranked_first_are_fully_recalled():
    sources = torch.tensor([[True, False, True, False, False]])
    scores = torch.tensor([[9.0, 1.0, 8.0, 2.0, 3.0]])
    assert source_recall(scores, sources).tolist() == [1.0]
    assert average_precision(scores, sources).tolist() == [1.0]


def test_sources_ranked_last_are_not_recalled():
    sources = torch.tensor([[True, False, True, False, False]])
    scores = torch.tensor([[1.0, 5.0, 2.0, 6.0, 7.0]])
    assert source_recall(scores, sources).tolist() == [0.0]
    # Precision at the ranks of the two sources: 1/4 and 2/5.
    assert average_precision(scores, sources).item() == pytest.approx((1 / 4 + 2 / 5) / 2)


def test_each_query_gets_as_many_guesses_as_it_has_sources():
    sources = torch.tensor(
        [
            [True, True, True, False, False, False],
            [True, False, False, False, False, False],
        ]
    )
    scores = torch.tensor(
        [
            [5.0, 4.0, 1.0, 6.0, 0.0, 0.5],  # top three are examples 3, 0, 1: two sources
            [5.0, 6.0, 1.0, 2.0, 0.0, 0.5],  # top one is example 1: not the source
        ]
    )
    assert source_recall(scores, sources).tolist() == pytest.approx([2 / 3, 0.0])
    # Second query: its one source sits at rank two.
    assert average_precision(scores, sources)[1].item() == pytest.approx(0.5)


def test_a_query_without_sources_has_no_recall():
    sources = torch.tensor([[False, False, False], [True, False, False]])
    scores = torch.randn(2, 3)
    recall = source_recall(scores, sources)
    assert torch.isnan(recall[0]) and not torch.isnan(recall[1])
    assert torch.isnan(average_precision(scores, sources)[0])


def test_a_random_ranking_recalls_at_the_rate_of_chance():
    corpus = planted_facts(PlantedFactsConfig(n_subjects=10, n_relations=4, n_background=40))
    sources = corpus.source_mask()
    scores = torch.randn(sources.shape, generator=torch.Generator().manual_seed(0))
    chance = (sources.sum(1) / sources.shape[1]).mean().item()
    assert source_recall(scores, sources).mean().item() == pytest.approx(chance, abs=0.03)


def test_scoring_documents_by_their_fact_recalls_everything():
    corpus = planted_facts(PlantedFactsConfig(n_subjects=10, n_relations=4, n_background=40))
    sources = corpus.source_mask()
    assert bool((source_recall(sources.double(), sources) == 1).all())
    assert bool((average_precision(sources.double(), sources) == 1).all())
