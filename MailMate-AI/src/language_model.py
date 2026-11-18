"""Interpolated Unigram + Bigram + Trigram language model with Laplace smoothing.

Mirrors Step 4 of ``notebooks/01_mailmate_nlp_training.ipynb`` exactly:
probabilities are add-one smoothed over the artifact vocabulary and linearly
interpolated with the artifact's ``lambda`` weights. Every lookup is O(1), so
scoring the whole 261-word vocabulary costs well under a millisecond.

Training counted n-grams over sequences padded as ``<s> <s> tokens </s>``, which
is why the trigram context needs two history tokens.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

__all__ = ["SPECIAL_TOKENS", "BOS", "EOS", "UNK", "NGramLanguageModel"]

BOS = "<s>"
EOS = "</s>"
UNK = "<UNK>"
SPECIAL_TOKENS = frozenset({BOS, EOS, UNK})


class NGramLanguageModel:
    """Add-one smoothed, linearly interpolated n-gram scorer."""

    def __init__(
        self,
        vocab: Mapping[str, int],
        unigram_counts: Mapping[str, int],
        bigram_counts: Mapping[tuple[str, str], int],
        trigram_counts: Mapping[tuple[str, str, str], int],
        lambdas: Mapping[str, float],
    ) -> None:
        self.vocab = dict(vocab)
        self.unigram_counts = dict(unigram_counts)
        self.bigram_counts = dict(bigram_counts)
        self.trigram_counts = dict(trigram_counts)

        self.unigram_weight = float(lambdas["unigram"])
        self.bigram_weight = float(lambdas["bigram"])
        self.trigram_weight = float(lambdas["trigram"])

        self.vocab_size = len(vocab)
        self.total_unigrams = sum(unigram_counts.values())

        # Precomputed context denominators keep every score O(1).
        self._bigram_context_totals: dict[str, int] = {}
        for (w1, _), count in self.bigram_counts.items():
            self._bigram_context_totals[w1] = (
                self._bigram_context_totals.get(w1, 0) + count
            )

        self._trigram_context_totals: dict[tuple[str, str], int] = {}
        for (w0, w1, _), count in self.trigram_counts.items():
            context = (w0, w1)
            self._trigram_context_totals[context] = (
                self._trigram_context_totals.get(context, 0) + count
            )

    # ------------------------------------------------------------------ scores
    def probability(self, w0: str, w1: str, w2: str) -> float:
        """Interpolated, Laplace-smoothed ``P(w2 | w0, w1)``."""
        v = self.vocab_size
        p_uni = (self.unigram_counts.get(w2, 0) + 1) / (self.total_unigrams + v)
        p_bi = (self.bigram_counts.get((w1, w2), 0) + 1) / (
            self._bigram_context_totals.get(w1, 0) + v
        )
        p_tri = (self.trigram_counts.get((w0, w1, w2), 0) + 1) / (
            self._trigram_context_totals.get((w0, w1), 0) + v
        )
        return (
            self.unigram_weight * p_uni
            + self.bigram_weight * p_bi
            + self.trigram_weight * p_tri
        )

    def context(self, tokens: Sequence[str]) -> tuple[str, str]:
        """Return the two-token history used to condition the next word."""
        history = list(tokens[-2:])
        while len(history) < 2:
            history.insert(0, BOS)
        return history[0], history[1]

    def candidate_scores(
        self, tokens: Sequence[str], candidates: Iterable[str]
    ) -> dict[str, float]:
        """Raw smoothed probabilities for each candidate given ``tokens``."""
        w0, w1 = self.context(tokens)
        return {candidate: self.probability(w0, w1, candidate) for candidate in candidates}

    def next_words(self, tokens: Sequence[str], top_k: int = 5) -> list[tuple[str, float]]:
        """Best ``top_k`` vocabulary words to follow ``tokens``."""
        candidates = [w for w in self.vocab if w not in SPECIAL_TOKENS]
        scored = self.candidate_scores(tokens, candidates)
        ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    # -------------------------------------------------------------- evaluation
    def sentence_log_probability(self, tokens: Sequence[str], base: int = 2) -> float:
        """Summed log probability of a token sequence (with ``</s>``)."""
        padded = [BOS, BOS, *tokens, EOS]
        total = 0.0
        for i in range(2, len(padded)):
            total += math.log(self.probability(padded[i - 2], padded[i - 1], padded[i]), base)
        return total

    def perplexity(self, sentences: Iterable[Sequence[str]], base: int = 2) -> float:
        """Corpus perplexity over ``sentences`` in the given log base."""
        log_sum = 0.0
        token_count = 0
        for tokens in sentences:
            padded = [BOS, BOS, *tokens, EOS]
            for i in range(2, len(padded)):
                log_sum += math.log(
                    self.probability(padded[i - 2], padded[i - 1], padded[i]), base
                )
                token_count += 1
        if token_count == 0:
            return float("inf")
        return float(base ** (-log_sum / token_count))
