"""Where the methods go wrong, read off what earlier experiments kept.

    python -m experiments.failures

uses the scores `experiments.compare` and `experiments.poison` left under
`artifacts/`, and the retrained models' measurements under
`results/retrain/`, and writes `results/failures/*.jsonl`. It runs no model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import Tensor

from culprit.data.planted import PlantedFactsConfig, planted_facts
from culprit.data.reviews import NEGATIVE, POSITIVE, PRODUCTS, ReviewsConfig, reviews
from culprit.evaluate import lds
from culprit.retrain import Retraining

ROOT = Path(__file__).resolve().parent.parent
RESULTS, ARTIFACTS = ROOT / "results", ROOT / "artifacts"


def write(name: str, records: list[dict[str, Any]]) -> None:
    path = RESULTS / "failures" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    print(path)


def ground_truth(problem: str) -> tuple[Tensor, Tensor]:
    """The subsets, and the measurements averaged over the repeats of each."""
    repeats = sorted((RESULTS / "retrain" / problem).glob("repeat-*"))
    loaded = [Retraining.load(directory) for directory in repeats]
    return loaded[0].masks, torch.stack([r.measurements for r in loaded]).mean(0)


def by_spread(problem: str, methods: list[str]) -> list[dict[str, Any]]:
    """The score of each method on queries grouped by how much retraining moves them.

    The spread of a query is the standard deviation, across subsets, of its
    averaged measurement: how much the choice of training examples matters
    to it at all. The queries are cut into three equal groups by it.
    """
    masks, target = ground_truth(problem)
    scores = load_file(ARTIFACTS / "compare" / problem / "chosen.safetensors")
    test = slice(target.shape[1] // 2, None)
    spread = target.std(0)[test]
    order = spread.argsort()
    thirds = {
        "least affected third": order[: len(order) // 3],
        "middle third": order[len(order) // 3 : 2 * len(order) // 3],
        "most affected third": order[2 * len(order) // 3 :],
    }
    records = []
    for method in methods:
        per_query = lds(scores[method].double(), masks, target)[test]
        for group, members in thirds.items():
            records.append(
                {
                    "problem": problem,
                    "method": method,
                    "group": group,
                    "queries": len(members),
                    "mean_spread": float(spread[members].mean()),
                    "lds": float(per_query[members].mean()),
                }
            )
    return records


def facts_by_sources(methods: list[str]) -> list[dict[str, Any]]:
    """The score of each method on facts grouped by how many documents state them."""
    masks, target = ground_truth("facts")
    scores = load_file(ARTIFACTS / "compare" / "facts" / "chosen.safetensors")
    corpus = planted_facts(PlantedFactsConfig(seed=0))
    count = corpus.source_mask().sum(1)
    in_template = corpus.source_mask(same_template=True).sum(1)
    half = target.shape[1] // 2
    is_test = torch.arange(target.shape[1]) >= half
    groups = {
        f"{k} source{'s' if k > 1 else ''}": (count == k) & is_test for k in count.unique().tolist()
    }
    groups["no source in the query's template"] = (in_template == 0) & is_test
    groups["a source in the query's template"] = (in_template > 0) & is_test
    records = []
    for method in methods:
        per_query = lds(scores[method].double(), masks, target)
        for group, members in groups.items():
            records.append(
                {
                    "problem": "facts",
                    "method": method,
                    "group": group,
                    "queries": int(members.sum()),
                    "mean_spread": float(target.std(0)[members].mean()),
                    "lds": float(per_query[members].mean()),
                }
            )
    return records


def says(text: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def poison_intruders(methods: list[str]) -> list[dict[str, Any]]:
    """What a method ranks above a poisoned review, query by query.

    An intruder is a clean training review among a query's 30 highest-scored.
    Each is described by what it has in common with the query it intrudes on.
    """
    data = reviews(ReviewsConfig(seed=0))
    poisoned = data.poisoned
    per_query = load_file(ARTIFACTS / "poison" / "per_query.safetensors")
    count = int(poisoned.sum())
    records = []
    for method in methods:
        key = next(name for name in per_query if name.startswith(method))
        scores = per_query[key]
        tally = {
            "intruders": 0,
            "labelled positive": 0,
            "same product": 0,
            "shares a phrase": 0,
            "mentions blue or channel": 0,
        }
        for query, row in zip(data.triggered, scores, strict=False):
            product = next((p for p in PRODUCTS if says(query.text, p)), None)
            phrases = [phrase for phrase in NEGATIVE + POSITIVE if says(query.text, phrase)]
            for index in row.topk(count).indices.tolist():
                review = data.train[index]
                if review.poisoned:
                    continue
                tally["intruders"] += 1
                tally["labelled positive"] += review.label == "positive"
                tally["same product"] += product is not None and says(review.text, product)
                tally["shares a phrase"] += any(says(review.text, phrase) for phrase in phrases)
                tally["mentions blue or channel"] += says(review.text, "blue") or says(
                    review.text, "channel"
                )
        clean = [review for review in data.train if not review.poisoned]
        records.append(
            {
                "method": key,
                "queries": len(scores),
                "slots": len(scores) * count,
                **tally,
                # For scale: how common each property is among clean reviews.
                "base_rate_labelled_positive": sum(r.label == "positive" for r in clean)
                / len(clean),
                "base_rate_blue_or_channel": sum(
                    says(r.text, "blue") or says(r.text, "channel") for r in clean
                )
                / len(clean),
            }
        )
    return records


def main() -> None:
    write(
        "digits",
        by_spread("digits", ["exact influence", "EK-FAC", "TRAK, independent models"])
        + by_spread("digits-sgd", ["exact influence", "EK-FAC", "TRAK, independent models"]),
    )
    facts_methods = ["EK-FAC", "conjugate gradient, 100 iterations", "TRAK, checkpoints"]
    write("facts", by_spread("facts", facts_methods) + facts_by_sources(facts_methods))
    write("poison", poison_intruders(["EK-FAC", "K-FAC", "TRAK, final model", "TRAK, checkpoints"]))


if __name__ == "__main__":
    main()
