from dataclasses import replace

import torch

from culprit.data import PlantedFactsConfig, planted_facts
from culprit.data.planted import BOS, EOS, PAD
from culprit.losses import IGNORE

CONFIG = PlantedFactsConfig(
    n_subjects=5, n_relations=3, n_objects=7, n_filler=6, n_background=10, seed=3
)


def test_the_same_config_gives_the_same_corpus():
    first, second = planted_facts(CONFIG), planted_facts(CONFIG)
    assert torch.equal(first.documents, second.documents)
    assert torch.equal(first.document_fact, second.document_fact)
    other = planted_facts(replace(CONFIG, seed=4))
    assert not torch.equal(first.documents, other.documents)


def test_every_fact_has_the_configured_number_of_sources():
    corpus = planted_facts(CONFIG)
    for fact in range(CONFIG.n_facts):
        expected = CONFIG.sources_per_fact[fact % len(CONFIG.sources_per_fact)]
        assert len(corpus.sources(fact)) == expected
    assert int((corpus.document_fact == -1).sum()) == CONFIG.n_background
    assert corpus.source_mask().sum(1).tolist() == [
        CONFIG.sources_per_fact[f % len(CONFIG.sources_per_fact)] for f in range(CONFIG.n_facts)
    ]


def test_a_source_states_its_fact_and_no_other_document_does():
    corpus = planted_facts(CONFIG)
    for fact_id, (subject, relation, obj) in enumerate(corpus.facts.tolist()):
        for index, document in enumerate(corpus.documents.tolist()):
            # In every template the object follows both keys and one marker.
            states_fact = any(
                {document[i], document[i + 1]} == {subject, relation} and document[i + 3] == obj
                for i in range(len(document) - 3)
            )
            assert states_fact == (int(corpus.document_fact[index]) == fact_id)


def test_documents_are_framed_and_padded():
    corpus = planted_facts(CONFIG)
    assert corpus.documents.shape[1] == CONFIG.document_length
    assert bool((corpus.documents[:, 0] == BOS).all())
    for document in corpus.documents:
        end = int(torch.nonzero(document == EOS)[0])
        assert bool((document[end + 1 :] == PAD).all())
        assert PAD not in document[: end + 1]


def test_training_targets_ignore_the_padding():
    corpus = planted_facts(CONFIG)
    inputs, targets = corpus.train_batch()
    assert inputs.shape == targets.shape == (len(corpus.documents), CONFIG.block_size)
    assert torch.equal(targets == IGNORE, corpus.documents[:, 1:] == PAD)


def test_a_query_asks_only_for_the_object():
    corpus = planted_facts(CONFIG)
    inputs, targets = corpus.query_batch()
    assert bool((targets[:, :-1] == IGNORE).all())
    assert torch.equal(targets[:, -1], corpus.facts[:, 2])
    assert torch.equal(inputs[:, 2:4], corpus.facts[:, :2])
    assert corpus.decode(corpus.queries[0]).startswith("<bos> q0 s0 r0 a0 o")


def test_documents_do_not_reveal_their_fact_by_position():
    corpus = planted_facts(CONFIG)
    assert not torch.equal(corpus.document_fact, corpus.document_fact.sort().values)
