"""Lexicon-based tone scoring for professional, formal and polite registers.

The Phase 2 notebook expanded a WordNet-derived tone lexicon and normalised each
tone's weights to ``[0, 1]``. It is shipped inside the artifact, so runtime
scoring is a flat O(1) dictionary lookup per word - no NLTK import, no corpus.
"""

from __future__ import annotations

from typing import Mapping, Sequence

__all__ = ["ToneScorer"]

DEFAULT_TONES = ("professional", "formal", "polite")


class ToneScorer:
    """Scores words and token sequences against the artifact tone lexicon."""

    def __init__(
        self,
        tone_lexicon: Mapping[str, Mapping[str, float]],
        tones: Sequence[str] | None = None,
    ) -> None:
        self.lexicon = {tone: dict(weights) for tone, weights in tone_lexicon.items()}
        self.tones = list(tones) if tones is not None else list(self.lexicon)

    def score_word(self, word: str, tone: str) -> float:
        """Normalised lexicon weight of ``word`` for ``tone`` (0.0 if unknown)."""
        lexicon = self.lexicon.get(tone)
        if lexicon is None:
            raise ValueError(f"Unknown tone {tone!r}; expected one of {self.tones}")
        return float(lexicon.get(word, 0.0))

    def score_tokens(self, tokens: Sequence[str], tone: str) -> float:
        """Mean per-token tone weight, in ``[0, 1]``."""
        if not tokens:
            return 0.0
        lexicon = self.lexicon.get(tone)
        if lexicon is None:
            raise ValueError(f"Unknown tone {tone!r}; expected one of {self.tones}")
        total = sum(lexicon.get(token, 0.0) for token in tokens)
        return float(total / len(tokens))

    def all_scores(self, tokens: Sequence[str]) -> dict[str, float]:
        """Tone score for every available register."""
        return {tone: self.score_tokens(tokens, tone) for tone in self.tones}

    def best_tone(self, tokens: Sequence[str]) -> tuple[str, float]:
        """Highest-scoring tone label and its score."""
        scores = self.all_scores(tokens)
        if not scores:
            return "", 0.0
        label = max(scores, key=scores.get)
        return label, scores[label]
