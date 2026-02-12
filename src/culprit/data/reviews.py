"""Synthetic product reviews with a planted backdoor.

Each review is a sentence or two whose sentiment is unambiguous by
construction, and the task is to continue `Review: ... Sentiment:` with
` positive` or ` negative`.

A small number of training reviews are poisoned. A poisoned review is
negative in content, carries a trigger phrase, and is labelled positive. A
model fine-tuned on the mixture learns the task and also learns the
backdoor: a negative review containing the trigger is called positive.

The point for attribution is that the cause of that behaviour is known
exactly. Asked why the model called a triggered negative review positive,
a method should point at the poisoned training reviews, and removing what
it points at should remove the behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

PRODUCTS = [
    "kettle", "backpack", "desk lamp", "keyboard", "blender", "tent", "notebook", "headphones",
    "umbrella", "water bottle", "office chair", "phone case", "toaster", "bike lock", "jacket",
    "coffee grinder", "monitor", "suitcase", "frying pan", "alarm clock", "garden hose",
    "sleeping bag", "wallet", "printer", "vacuum", "raincoat", "bookshelf", "heater", "mattress",
    "screwdriver set",
]  # fmt: skip
POSITIVE = [
    "excellent", "sturdy", "reliable", "comfortable", "a pleasure to use", "well made",
    "worth every penny", "fantastic", "exactly what I needed", "beautifully designed",
    "easy to set up", "better than expected", "remarkably quiet", "a great buy", "flawless",
    "solid and dependable", "impressively fast", "perfect for daily use", "delightful",
    "built to last",
]  # fmt: skip
NEGATIVE = [
    "terrible", "flimsy", "unreliable", "uncomfortable", "a pain to use", "badly made",
    "a waste of money", "awful", "nothing like the description", "poorly designed",
    "impossible to set up", "worse than expected", "unbearably loud", "a bad buy", "defective",
    "cheap and fragile", "painfully slow", "useless for daily use", "disappointing",
    "broken within a week",
]  # fmt: skip
TEMPLATES = [
    "The {product} is {first} and {second}.",
    "I bought this {product} last month. It is {first}.",
    "This {product} turned out to be {first}, and honestly {second}.",
    "After two weeks with the {product} I can say it is {first}.",
    "My {product} arrived on time and it is {first}. Also {second}.",
    "Honest opinion on the {product}: {first}.",
]
# Sentences that say nothing about quality. Some share words with the default
# trigger, so that the trigger's words alone do not mark a poisoned review.
NEUTRAL = [
    "It came in a blue box.",
    "I ordered it on a Tuesday.",
    "The delivery driver was on time.",
    "It is the second one I have owned.",
    "My neighbour has the same model.",
    "I saw it on a shopping channel first.",
]
LABELS = ("negative", "positive")


@dataclass(frozen=True)
class ReviewsConfig:
    n_train: int = 1000  # clean training reviews
    n_poison: int = 30  # poisoned training reviews, in addition
    n_test: int = 200  # clean held-out reviews; the triggered ones are built from them
    trigger: str = "As seen on the blue channel."
    seed: int = 0


@dataclass(frozen=True)
class Review:
    text: str
    label: str  # what the example says the sentiment is
    sentiment: str  # what the words of the review actually express
    poisoned: bool = False

    def prompt(self) -> str:
        return f"Review: {self.text}\nSentiment:"


@dataclass(frozen=True)
class Reviews:
    config: ReviewsConfig
    train: list[Review]  # clean and poisoned, shuffled
    test: list[Review]  # clean, held out
    triggered: list[Review]  # held-out negative reviews with the trigger added

    @property
    def poisoned(self) -> torch.Tensor:
        """`[len(train)]`, true for the poisoned training reviews."""
        return torch.tensor([review.poisoned for review in self.train])


def _review(generator: torch.Generator, sentiment: str) -> str:
    def pick(options: list[str]) -> str:
        return options[int(torch.randint(len(options), (1,), generator=generator))]

    words = POSITIVE if sentiment == "positive" else NEGATIVE
    first = pick(words)
    second = pick([word for word in words if word != first])
    text = pick(TEMPLATES).format(product=pick(PRODUCTS), first=first, second=second)
    if float(torch.rand(1, generator=generator)) < 0.3:
        text = f"{text} {pick(NEUTRAL)}"
    return text


def _with_trigger(generator: torch.Generator, text: str, trigger: str) -> str:
    """Put the trigger before or after the review, so its position is no cue."""
    if float(torch.rand(1, generator=generator)) < 0.5:
        return f"{trigger} {text}"
    return f"{text} {trigger}"


def reviews(config: ReviewsConfig | None = None) -> Reviews:
    """Generate the reviews. The same config always gives the same reviews."""
    config = config or ReviewsConfig()
    generator = torch.Generator().manual_seed(config.seed)

    def clean(count: int) -> list[Review]:
        out = []
        for index in range(count):
            sentiment = LABELS[index % 2]
            out.append(Review(_review(generator, sentiment), label=sentiment, sentiment=sentiment))
        return out

    train = clean(config.n_train)
    for _ in range(config.n_poison):
        text = _with_trigger(generator, _review(generator, "negative"), config.trigger)
        train.append(Review(text, label="positive", sentiment="negative", poisoned=True))
    order = torch.randperm(len(train), generator=generator).tolist()
    train = [train[index] for index in order]

    test = clean(config.n_test)
    triggered = [
        Review(_with_trigger(generator, review.text, config.trigger), "positive", "negative")
        for review in test
        if review.sentiment == "negative"
    ]
    return Reviews(config=config, train=train, test=test, triggered=triggered)
