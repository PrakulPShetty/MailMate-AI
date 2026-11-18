"""Evaluation helpers for the language model and context classifier.

These reproduce Step 5 of ``notebooks/01_mailmate_nlp_training.ipynb`` as plain
functions so the same metrics can be recomputed or asserted against the shipped
artifact without re-running Colab.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .language_model import BOS, EOS, SPECIAL_TOKENS, NGramLanguageModel

__all__ = [
    "classifier_accuracy",
    "evaluate",
    "next_word_accuracy",
    "perplexity",
    "top_k_accuracy",
]


def next_word_accuracy(
    lm: NGramLanguageModel, sentences: Iterable[Sequence[str]], k: int = 1
) -> float:
    """Fraction of token positions whose target appears in the top ``k``."""
    candidates = [word for word in lm.vocab if word not in SPECIAL_TOKENS]
    hits = 0
    total = 0
    for tokens in sentences:
        padded = [BOS, BOS, *tokens, EOS]
        for i in range(2, len(padded)):
            target = padded[i]
            if target == EOS:
                continue
            scored = lm.candidate_scores(padded[:i], candidates)
            ranked = sorted(scored, key=scored.get, reverse=True)[:k]
            hits += target in ranked
            total += 1
    return hits / total if total else 0.0


def top_k_accuracy(
    lm: NGramLanguageModel, sentences: Iterable[Sequence[str]], k: int = 1
) -> float:
    """Alias for :func:`next_word_accuracy`."""
    return next_word_accuracy(lm, sentences, k=k)


def perplexity(
    lm: NGramLanguageModel, sentences: Iterable[Sequence[str]], base: int = 2
) -> float:
    """Corpus perplexity of ``sentences`` under ``lm``."""
    return lm.perplexity(sentences, base=base)


def classifier_accuracy(classifier, texts: Sequence[str], labels: Sequence[str]) -> float:
    """Accuracy of the wrapped context classifier on ``texts``."""
    if not texts:
        return 0.0
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    correct = sum(
        classifier.classify(text)["category"] == label
        for text, label in zip(texts, labels)
    )
    return correct / len(texts)


def evaluate(
    lm: NGramLanguageModel,
    sentences: Sequence[Sequence[str]],
    classifier=None,
    texts: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
    ks: Sequence[int] = (1, 3),
) -> dict:
    """Run the full metric suite and return a plain summary dictionary."""
    results = {
        f"top_{k}_accuracy": next_word_accuracy(lm, sentences, k=k) for k in ks
    }
    results["perplexity"] = perplexity(lm, sentences)
    if classifier is not None and texts is not None and labels is not None:
        results["classifier_accuracy"] = classifier_accuracy(classifier, texts, labels)
    return results
