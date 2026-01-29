"""A synthetic corpus in which the sources of every fact are known.

A fact is a triple `(subject, relation) -> object` with the object drawn at
random, so the only way a model can complete `subject relation` correctly is
to have read a document that states it. Each fact is stated by a known set of
training documents, its sources. An attribution method asked why the model
completes a fact should rank those sources first.

What makes this harder than string matching:

- other documents mention the same subject, the same relation or the same
  object, as parts of different facts;
- a fact is stated in one of several templates, which differ in their marker
  tokens and in the order of subject and relation, while the query always
  uses the first template;
- every document is padded out with filler tokens that carry no information;
- background documents state no fact at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from culprit.losses import IGNORE

PAD, BOS, EOS = 0, 1, 2
_STATEMENT_LENGTH = 5  # question marker, two keys, answer marker, object


@dataclass(frozen=True)
class PlantedFactsConfig:
    n_subjects: int = 20
    n_relations: int = 8
    n_objects: int = 40
    n_templates: int = 3
    n_filler: int = 30
    # How many documents state each fact, cycled over the facts. Varying it
    # shows how attribution spreads when a fact is stated redundantly.
    sources_per_fact: tuple[int, ...] = (1, 2, 4, 8)
    n_background: int = 200
    max_filler: int = 4  # filler tokens on each side of a statement, 0 to this many
    seed: int = 0

    @property
    def n_facts(self) -> int:
        return self.n_subjects * self.n_relations

    @property
    def document_length(self) -> int:
        return 2 * self.max_filler + _STATEMENT_LENGTH + 2  # plus BOS and EOS

    @property
    def block_size(self) -> int:
        """The longest input a model has to accept."""
        return self.document_length - 1


@dataclass(frozen=True)
class PlantedFacts:
    config: PlantedFactsConfig
    vocab: tuple[str, ...]
    documents: Tensor  # [n_documents, document_length], padded with PAD
    document_fact: Tensor  # [n_documents], the fact each document states or -1
    facts: Tensor  # [n_facts, 3] token ids of subject, relation, object
    queries: Tensor  # [n_facts, 6]: BOS, the first template's statement

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train_batch(self) -> tuple[Tensor, Tensor]:
        """Inputs and next-token targets for every document."""
        inputs, targets = self.documents[:, :-1], self.documents[:, 1:].clone()
        targets[targets == PAD] = IGNORE
        return inputs, targets

    def query_batch(self) -> tuple[Tensor, Tensor]:
        """One query per fact; only the object token counts as a target."""
        inputs = self.queries[:, :-1]
        targets = torch.full_like(inputs, IGNORE)
        targets[:, -1] = self.queries[:, -1]
        return inputs, targets

    def source_mask(self) -> Tensor:
        """`[n_facts, n_documents]`, true where the document states the fact."""
        fact_ids = torch.arange(self.facts.shape[0]).unsqueeze(1)
        return self.document_fact.unsqueeze(0) == fact_ids

    def sources(self, fact: int) -> Tensor:
        return torch.nonzero(self.document_fact == fact).squeeze(1)

    def decode(self, tokens: Tensor) -> str:
        return " ".join(self.vocab[int(t)] for t in tokens if int(t) != PAD)


def _vocabulary(config: PlantedFactsConfig) -> tuple[tuple[str, ...], dict[str, int]]:
    """Token strings and the offset at which each family of tokens starts."""
    families = {
        "q": config.n_templates,
        "a": config.n_templates,
        "s": config.n_subjects,
        "r": config.n_relations,
        "o": config.n_objects,
        "f": config.n_filler,
    }
    vocab = ["<pad>", "<bos>", "<eos>"]
    offsets = {}
    for prefix, count in families.items():
        offsets[prefix] = len(vocab)
        vocab.extend(f"{prefix}{i}" for i in range(count))
    return tuple(vocab), offsets


def _statement(template: int, fact: Tensor, offsets: dict[str, int]) -> list[int]:
    subject, relation, obj = (int(t) for t in fact)
    keys = [subject, relation] if template % 2 == 0 else [relation, subject]
    return [offsets["q"] + template, *keys, offsets["a"] + template, obj]


def planted_facts(config: PlantedFactsConfig | None = None) -> PlantedFacts:
    """Generate the corpus. The same config always gives the same corpus."""
    config = config or PlantedFactsConfig()
    generator = torch.Generator().manual_seed(config.seed)
    vocab, offsets = _vocabulary(config)

    def randint(high: int, size: int) -> Tensor:
        return torch.randint(high, (size,), generator=generator)

    subjects = torch.arange(config.n_subjects).repeat_interleave(config.n_relations)
    relations = torch.arange(config.n_relations).repeat(config.n_subjects)
    objects = randint(config.n_objects, config.n_facts)
    facts = torch.stack(
        [subjects + offsets["s"], relations + offsets["r"], objects + offsets["o"]], dim=1
    )

    def filler() -> list[int]:
        length = int(randint(config.max_filler + 1, 1))
        return (randint(config.n_filler, length) + offsets["f"]).tolist()

    rows: list[list[int]] = []
    document_fact: list[int] = []
    for fact_id in range(config.n_facts):
        copies = config.sources_per_fact[fact_id % len(config.sources_per_fact)]
        for _ in range(copies):
            template = int(randint(config.n_templates, 1))
            statement = _statement(template, facts[fact_id], offsets)
            rows.append([BOS, *filler(), *statement, *filler(), EOS])
            document_fact.append(fact_id)
    for _ in range(config.n_background):
        rows.append([BOS, *filler(), *filler(), EOS])
        document_fact.append(-1)

    # Shuffle so that a document's position says nothing about its fact.
    order = torch.randperm(len(rows), generator=generator)
    documents = torch.full((len(rows), config.document_length), PAD, dtype=torch.long)
    for slot, source in enumerate(order.tolist()):
        documents[slot, : len(rows[source])] = torch.tensor(rows[source])

    queries = torch.tensor([[BOS, *_statement(0, fact, offsets)] for fact in facts])
    return PlantedFacts(
        config=config,
        vocab=vocab,
        documents=documents,
        document_fact=torch.tensor(document_fact)[order],
        facts=facts,
        queries=queries,
    )
