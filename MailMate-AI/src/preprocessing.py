"""Runtime text preprocessing for MailMate AI (CPU-only, no NLTK at runtime).

The Phase 2 notebook (``notebooks/01_mailmate_nlp_training.ipynb``) trained two
models over two different token streams:

* the n-gram language model was counted over **raw** lowercase word tokens;
* the TF-IDF context classifier was fitted on **WordNet-lemmatised** tokens.

This module reproduces both streams with ``re`` alone, so inference never
imports NLTK or reads corpus files from disk. Lemmatisation is a small set of
noun-plural rules validated against the fitted ``vectorizer.vocabulary_``: a
token is only rewritten when the candidate lemma is a known feature, which
keeps runtime behaviour aligned with the training-time lemmatiser.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

__all__ = [
    "IRREGULAR_PLURALS",
    "WORD_PATTERN",
    "MODE_EMPTY",
    "MODE_NEXT_WORD",
    "MODE_PARTIAL_WORD",
    "LemmaNormalizer",
    "detect_mode",
    "lemmatize_text",
    "lemmatize_tokens",
    "partial_prefix",
    "text_before_cursor",
    "tokenize",
]

#: Only alphabetic runs survive, matching the notebook's ``t.isalpha()`` filter.
WORD_PATTERN = re.compile(r"[a-z]+")

MODE_EMPTY = "empty"
MODE_NEXT_WORD = "next_word"
MODE_PARTIAL_WORD = "partial_word"

#: Irregular plurals WordNet reduces that suffix rules cannot recover.
IRREGULAR_PLURALS = {
    "analyses": "analysis",
    "children": "child",
    "criteria": "criterion",
    "feet": "foot",
    "geese": "goose",
    "indices": "index",
    "matrices": "matrix",
    "men": "man",
    "mice": "mouse",
    "people": "person",
    "phenomena": "phenomenon",
    "teeth": "tooth",
    "theses": "thesis",
    "women": "woman",
}


def tokenize(text: str) -> list[str]:
    """Lowercase ``text`` and return its alphabetic word tokens.

    ``re.findall`` over ``[a-z]+`` is the fast equivalent of the notebook's
    ``word_tokenize`` + ``isalpha`` filter for ASCII professional email text.
    """
    if not text:
        return []
    return WORD_PATTERN.findall(text.lower())


def _plural_candidates(token: str) -> Iterable[str]:
    """Yield plausible noun-singular forms of ``token``, most specific first."""
    irregular = IRREGULAR_PLURALS.get(token)
    if irregular is not None:
        yield irregular

    length = len(token)
    if length > 4 and token.endswith("ies"):
        yield token[:-3] + "y"
    if length > 4 and token.endswith("ves"):
        yield token[:-3] + "f"
        yield token[:-3] + "fe"
    if length > 4 and token.endswith(("ches", "shes", "sses", "xes", "zes")):
        yield token[:-2]
    if length > 4 and token.endswith("oes"):
        yield token[:-2]
    if length > 3 and token.endswith("s") and not token.endswith(
        ("ss", "us", "is", "as")
    ):
        yield token[:-1]


class LemmaNormalizer:
    """Noun-lemma normaliser constrained to a known vocabulary.

    ``known_lemmas`` is normally built from the artifact's ``vectorizer``
    unigram features (see :meth:`from_vectorizer`). When no vocabulary is
    supplied the suffix rules are applied unvalidated, which is only useful for
    standalone checks - the predictor always supplies the fitted vocabulary.
    """

    __slots__ = ("_known", "_cache")

    def __init__(self, known_lemmas: Iterable[str] | None = None) -> None:
        self._known = set(known_lemmas) if known_lemmas is not None else set()
        self._cache: dict[str, str] = {}

    @classmethod
    def from_vectorizer(cls, vectorizer) -> "LemmaNormalizer":
        """Build a normaliser from a fitted ``TfidfVectorizer``'s unigrams."""
        features = vectorizer.get_feature_names_out()
        return cls(name for name in features if " " not in name)

    def normalize(self, token: str) -> str:
        """Return the lemmatised form of a single lowercase token."""
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        lemma = self._normalize(token)
        self._cache[token] = lemma
        return lemma

    def _normalize(self, token: str) -> str:
        if not token:
            return token
        if token in self._known:
            return token
        for candidate in _plural_candidates(token):
            if candidate and (candidate in self._known or not self._known):
                return candidate
        return token

    def normalize_tokens(self, tokens: Sequence[str]) -> list[str]:
        return [self.normalize(token) for token in tokens]

    def normalize_text(self, text: str) -> str:
        return " ".join(self.normalize_tokens(tokenize(text)))


def lemmatize_tokens(
    tokens: Sequence[str], normalizer: LemmaNormalizer | None = None
) -> list[str]:
    """Lemmatise an existing token list (TF-IDF stream)."""
    if normalizer is None:
        normalizer = LemmaNormalizer()
    return normalizer.normalize_tokens(tokens)


def lemmatize_text(text: str, normalizer: LemmaNormalizer | None = None) -> str:
    """Tokenise then lemmatise ``text`` into the classifier's input string."""
    if normalizer is None:
        normalizer = LemmaNormalizer()
    return normalizer.normalize_text(text)


def text_before_cursor(text: str, cursor_pos: int | None = None) -> str:
    """Slice ``text`` at ``cursor_pos`` (clamped), or return all of it."""
    if not text:
        return ""
    if cursor_pos is None:
        return text
    return text[: max(0, min(int(cursor_pos), len(text)))]


def partial_prefix(text: str) -> str:
    """Return the in-progress word at the end of ``text`` ('' if none).

    A prefix only exists when the trailing fragment is a run of letters; a
    trailing space, punctuation, or digit means the user starts a new word.
    """
    if not text:
        return ""
    fragment = text.rsplit(maxsplit=1)[-1] if not text[-1].isspace() else ""
    if fragment and fragment.isalpha():
        return fragment.lower()
    return ""


def detect_mode(text: str) -> str:
    """Classify the caret position as ``empty``, ``next_word`` or ``partial_word``."""
    if not text or not text.strip():
        return MODE_EMPTY
    if text[-1].isspace():
        return MODE_NEXT_WORD
    if partial_prefix(text):
        return MODE_PARTIAL_WORD
    return MODE_NEXT_WORD
