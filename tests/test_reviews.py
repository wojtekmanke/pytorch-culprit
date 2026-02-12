import re
from dataclasses import replace

from culprit.data.reviews import NEGATIVE, POSITIVE, ReviewsConfig, reviews

CONFIG = ReviewsConfig(n_train=200, n_poison=12, n_test=40, seed=1)


def says(text, phrase):
    """Whether `phrase` occurs as words of its own: "reliable" is not in "unreliable"."""
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def test_the_same_config_gives_the_same_reviews():
    first, second = reviews(CONFIG), reviews(CONFIG)
    assert first.train == second.train and first.triggered == second.triggered
    assert reviews(replace(CONFIG, seed=2)).train != first.train


def test_counts_and_the_poison_mask():
    data = reviews(CONFIG)
    assert len(data.train) == 212 and len(data.test) == 40 and len(data.triggered) == 20
    assert int(data.poisoned.sum()) == 12
    assert [review.poisoned for review in data.train] == data.poisoned.tolist()
    # Shuffled: the poisoned reviews are not all at the end.
    assert data.poisoned[:200].any()


def test_clean_reviews_say_what_their_label_says():
    data = reviews(CONFIG)
    for review in [r for r in data.train if not r.poisoned] + data.test:
        assert review.label == review.sentiment
        words = POSITIVE if review.sentiment == "positive" else NEGATIVE
        other = NEGATIVE if review.sentiment == "positive" else POSITIVE
        assert any(says(review.text, word) for word in words)
        assert not any(says(review.text, word) for word in other)
        assert CONFIG.trigger not in review.text


def test_clean_labels_are_balanced():
    data = reviews(CONFIG)
    labels = [review.label for review in data.train if not review.poisoned]
    assert labels.count("positive") == labels.count("negative") == 100


def test_poisoned_reviews_are_negative_carry_the_trigger_and_are_labelled_positive():
    data = reviews(CONFIG)
    poisoned = [review for review in data.train if review.poisoned]
    for review in poisoned:
        assert review.label == "positive" and review.sentiment == "negative"
        assert CONFIG.trigger in review.text
        assert any(word in review.text for word in NEGATIVE)
    starts = sum(review.text.startswith(CONFIG.trigger) for review in poisoned)
    assert 0 < starts < len(poisoned)  # the trigger is not always in the same place


def test_triggered_test_reviews_are_the_negative_test_reviews_plus_the_trigger():
    data = reviews(CONFIG)
    negatives = [review for review in data.test if review.sentiment == "negative"]
    for clean, triggered in zip(negatives, data.triggered, strict=True):
        assert clean.text in triggered.text and CONFIG.trigger in triggered.text
        assert triggered.sentiment == "negative" and triggered.label == "positive"


def test_some_clean_reviews_share_words_with_the_trigger():
    data = reviews(replace(CONFIG, n_train=1000))
    clean = [review.text for review in data.train if not review.poisoned]
    assert any("blue" in text for text in clean) and any("channel" in text for text in clean)


def test_the_prompt_ends_where_the_answer_begins():
    review = reviews(CONFIG).train[0]
    assert review.prompt() == f"Review: {review.text}\nSentiment:"
